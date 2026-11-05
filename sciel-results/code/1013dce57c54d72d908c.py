"""Matrix multiplication 3x4x5 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank1(n, m, p, entries_u, entries_v, entries_w, rank_offset):
    """
    Add rank-1 terms to U, V, W.
    entries_u: list of (idx, coeff) where idx = i*m+j
    entries_v: list of (idx, coeff) where idx = j*p+k
    entries_w: list of (idx, coeff) where idx = k*n+i
    """
    U_col = [0] * (n * m)
    V_col = [0] * (m * p)
    W_col = [0] * (p * n)
    for idx, c in entries_u:
        U_col[idx] += c
    for idx, c in entries_v:
        V_col[idx] += c
    for idx, c in entries_w:
        W_col[idx] += c
    return U_col, V_col, W_col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    # Strategy: split into blocks and use Strassen where beneficial.
    # We'll build the decomposition by enumerating all (i,j,k) triples
    # but try to find rank reductions via structured decompositions.
    
    # Start with schoolbook: rank = 3*4*5 = 60
    # Try to find a better decomposition.
    
    # Let's use a mixed approach:
    # Split j into j0={0,1} and j1={2,3}
    # Split k into k0={0,1}, k1={2,3}, k2={4}
    # Split i into i0={0,1}, i1={2}
    
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2 -> Strassen rank 7
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): 2x2x2 -> Strassen rank 7
    # Block (i in {0,1}, j in {0,1}, k in {4}): 2x2x1 -> rank 4
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): 2x2x2 -> Strassen rank 7
    # Block (i in {0,1}, j in {2,3}, k in {2,3}): 2x2x2 -> Strassen rank 7
    # Block (i in {0,1}, j in {2,3}, k in {4}): 2x2x1 -> rank 4
    # Block (i in {2}, j in {0,1}, k in {0,1}): 1x2x2 -> rank 4
    # Block (i in {2}, j in {0,1}, k in {2,3}): 1x2x2 -> rank 4
    # Block (i in {2}, j in {0,1}, k in {4}): 1x2x1 -> rank 2
    # Block (i in {2}, j in {2,3}, k in {0,1}): 1x2x2 -> rank 4
    # Block (i in {2}, j in {2,3}, k in {2,3}): 1x2x2 -> rank 4
    # Block (i in {2}, j in {2,3}, k in {4}): 1x2x1 -> rank 2
    # Total: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    # Let's implement this.
    # Strassen 2x2x2 decomposition:
    # For A (2x2), B (2x2), C = A*B (2x2)
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block at (i0,j0,k0)."""
        nonlocal rank
        # Local indices: a in {0,1} for i, b in {0,1} for j, c in {0,1} for k
        # Global: i = i0+a, j = j0+b, k = k0+c
        # U index: i*m+j = (i0+a)*m + (j0+b)
        # V index: j*p+k = (j0+b)*p + (k0+c)
        # W index: k*n+i = (k0+c)*n + (i0+a)
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        u1 = [((i0+0)*m+(j0+0), 1), ((i0+1)*m+(j0+1), 1)]
        v1 = [((j0+0)*p+(k0+0), 1), ((j0+1)*p+(k0+1), 1)]
        w1 = [((k0+0)*n+(i0+0), 1), ((k0+1)*n+(i0+1), 1)]  # C[0,0] and C[1,1]
        uc, vc, wc = _build_rank1(n, m, p, u1, v1, w1, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        u2 = [((i0+1)*m+(j0+0), 1), ((i0+1)*m+(j0+1), 1)]
        v2 = [((j0+0)*p+(k0+0), 1)]
        w2 = [((k0+0)*n+(i0+1), 1)]  # C[1,0]
        uc, vc, wc = _build_rank1(n, m, p, u2, v2, w2, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
        
        # M3 = A[0,0]*(B[1,0]-B[1,1])
        u3 = [((i0+0)*m+(j0+0), 1)]
        v3 = [((j0+1)*p+(k0+0), 1), ((j0+1)*p+(k0+1), -1)]
        w3 = [((k0+1)*n+(i0+0), 1)]  # C[0,1]
        uc, vc, wc = _build_rank1(n, m, p, u3, v3, w3, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
        
        # M4 = A[1,1]*(B[0,1]-B[0,0])
        u4 = [((i0+1)*m+(j0+1), 1)]
        v4 = [((j0+0)*p+(k0+1), 1), ((j0+0)*p+(k0+0), -1)]
        w4 = [((k0+0)*n+(i0+0), 1), ((k0+0)*n+(i0+1), 1)]  # C[0,0] and C[1,0]
        uc, vc, wc = _build_rank1(n, m, p, u4, v4, w4, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        u5 = [((i0+0)*m+(j0+0), 1), ((i0+0)*m+(j0+1), 1)]
        v5 = [((j0+1)*p+(k0+1), 1)]
        w5 = [((k0+0)*n+(i0+0), -1), ((k0+1)*n+(i0+0), 1)]  # -C[0,0] + C[0,1]
        uc, vc, wc = _build_rank1(n, m, p, u5, v5, w5, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        u6 = [((i0+1)*m+(j0+0), 1), ((i0+0)*m+(j0+0), -1)]
        v6 = [((j0+0)*p+(k0+0), 1), ((j0+0)*p+(k0+1), 1)]
        w6 = [((k0+1)*n+(i0+1), 1)]  # C[1,1]
        uc, vc, wc = _build_rank1(n, m, p, u6, v6, w6, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        u7 = [((i0+0)*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)]
        v7 = [((j0+1)*p+(k0+0), 1), ((j0+1)*p+(k0+1), 1)]
        w7 = [((k0+0)*n+(i0+0), 1), ((k0+1)*n+(i0+1), -1)]  # C[0,0] - C[1,1]
        uc, vc, wc = _build_rank1(n, m, p, u7, v7, w7, rank)
        U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
        rank += 1
    
    def add_block_2x2x1(i0, j0, k0):
        """2x2x1 block: rank 4."""
        nonlocal rank
        for a in range(2):
            for b in range(2):
                i = i0 + a
                j = j0 + b
                k = k0
                uc, vc, wc = _build_rank1(n, m, p, 
                    [(i*m+j, 1)], [(j*p+k, 1)], [(k*n+i, 1)], rank)
                U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
                rank += 1
    
    def add_block_1x2x2(i0, j0, k0):
        """1x2x2 block: rank 4."""
        nonlocal rank
        for b in range(2):
            for c in range(2):
                i = i0
                j = j0 + b
                k = k0 + c
                uc, vc, wc = _build_rank1(n, m, p, 
                    [(i*m+j, 1)], [(j*p+k, 1)], [(k*n+i, 1)], rank)
                U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
                rank += 1
    
    def add_block_1x2x1(i0, j0, k0):
        """1x2x1 block: rank 2."""
        nonlocal rank
        for b in range(2):
            i = i0
            j = j0 + b
            k = k0
            uc, vc, wc = _build_rank1(n, m, p, 
                [(i*m+j, 1)], [(j*p+k, 1)], [(k*n+i, 1)], rank)
            U_cols.append(uc); V_cols.append(vc); W_cols.append(wc)
            rank += 1
    
    # i0={0,1}, i1={2}; j0={0,1}, j1={2,3}; k0={0,1}, k1={2,3}, k2={4}
    add_strassen_2x2x2(0, 0, 0)  # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_2x2x2(0, 0, 2)  # i in {0,1}, j in {0,1}, k in {2,3}
    add_block_2x2x1(0, 0, 4)     # i in {0,1}, j in {0,1}, k in {4}
    add_strassen_2x2x2(0, 2, 0)  # i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_2x2x2(0, 2, 2)  # i in {0,1}, j in {2,3}, k in {2,3}
    add_block_2x2x1(0, 2, 4)     # i in {0,1}, j in {2,3}, k in {4}
    add_block_1x2x2(2, 0, 0)     # i in {2}, j in {0,1}, k in {0,1}
    add_block_1x2x2(2, 0, 2)     # i in {2}, j in {0,1}, k in {2,3}
    add_block_1x2x1(2, 0, 4)     # i in {2}, j in {0,1}, k in {4}
    add_block_1x2x2(2, 2, 0)     # i in {2}, j in {2,3}, k in {0,1}
    add_block_1x2x2(2, 2, 2)     # i in {2}, j in {2,3}, k in {2,3}
    add_block_1x2x1(2, 2, 4)     # i in {2}, j in {2,3}, k in {4}
    
    # Convert to list-of-lists format
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
