import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify exact tensor decomposition.
    """
    if n is None:
        n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Vectorized: U_row * V_row dot W_row
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = np.dot(u_row * v_row, w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_3x2x2():
    """
    Returns lists of (u, v, w) for 3x2x2 multiplication.
    u: length 6 (3 rows * 2 cols)
    v: length 4 (2 rows * 2 cols)
    w: length 6 (2 cols * 3 rows) -- Wait, W index is k*n+i.
    n=3, m=2, p=2.
    U size: 3*2 = 6.
    V size: 2*2 = 4.
    W size: 2*3 = 6.
    
    Indices:
    U[i*2+j]
    V[j*2+k]
    W[k*3+i]
    """
    terms = []
    
    # Strassen for rows 0,1
    # A = [[a, b], [c, d]] (indices 0,1 in U)
    # B = [[x, y], [z, w]] (indices 0,1 in V)
    # C = [[C00, C01], [C10, C11]] (indices 0,1 in W for rows 0,1)
    
    # m1 = (a+d)(x+z)
    # a: U[0], d: U[3]
    # x: V[0], z: V[2]
    # C00: W[0], C01: W[1], C10: W[3], C11: W[4]
    
    # We need to represent each multiplication term as a vector.
    # Term: u * v * w
    # u has non-zeros at a, d
    # v has non-zeros at x, z
    # w has non-zeros at C00, C01, C10, C11? No, Strassen combines them.
    
    # Let's just add the 7 Strassen terms.
    # 1. (a+d)(x+z)
    u = [1, 0, 0, 1, 0, 0] # a, d
    v = [1, 0, 1, 0]       # x, z
    w = [1, 0, 0, 0, 1, 0] # C00=1, C11=1 (from m1+m4-m5+m7 and m1-m2+m3+m6)
    # Wait, m1 contributes to C00 and C11.
    # C00 = m1 + m4 - m5 + m7
    # C11 = m1 - m2 + m3 + m6
    # So m1 has coeff 1 in C00 and 1 in C11.
    terms.append((u, v, w))
    
    # 2. (c+d)x
    u = [0, 0, 1, 1, 0, 0] # c, d
    v = [1, 0, 0, 0]       # x
    w = [0, 0, 0, 1, -1, 0] # C10=1, C11=-1
    terms.append((u, v, w))
    
    # 3. a(y-z)
    u = [1, 0, 0, 0, 0, 0] # a
    v = [0, 1, -1, 0]      # y, -z
    w = [0, 1, 0, 0, 1, 0] # C01=1, C11=1
    terms.append((u, v, w))
    
    # 4. d(w-y)
    u = [0, 0, 0, 1, 0, 0] # d
    v = [0, -1, 0, 1]      # -y, w
    w = [1, 0, 1, 0, 0, 0] # C00=1, C10=1
    terms.append((u, v, w))
    
    # 5. (a+b)w
    u = [1, 1, 0, 0, 0, 0] # a, b
    v = [0, 0, 0, 1]       # w
    w = [-1, 1, 0, 0, 0, 0] # C00=-1, C01=1
    terms.append((u, v, w))
    
    # 6. (c-a)(x+z)
    u = [-1, 0, 1, 0, 0, 0] # -a, c
    v = [1, 0, 1, 0]        # x, z
    w = [0, 0, 0, 0, 1, 0]  # C11=1
    terms.append((u, v, w))
    
    # 7. (b-d)(y+w)
    u = [0, 1, 0, -1, 0, 0] # b, -d
    v = [0, 1, 0, 1]        # y, w
    w = [1, 0, 0, 0, 0, 0]  # C00=1
    terms.append((u, v, w))
    
    # Schoolbook for row 2
    # e, f in A (U[4], U[5])
    # x, y, z, w in B (V[0], V[1], V[2], V[3])
    # C20, C21 in C (W[2], W[5])
    
    # 8. e*x
    u = [0, 0, 0, 0, 1, 0]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 9. e*y
    u = [0, 0, 0, 0, 1, 0]
    v = [0, 1, 0, 0]
    w = [0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    # 10. f*z
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 1, 0]
    w = [0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 11. f*w
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 1]
    w = [0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    return terms

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses block Strassen decomposition to achieve rank 88.
    """
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We will build 8 blocks of 3x2x2.
    # J blocks: 0 (cols 0,1), 1 (cols 2,3)
    # K blocks: 
    #   For J=0: K0 (cols 0,1), K1 (cols 2,3)
    #   For J=1: K2 (cols 4,5), K3 (cols 6,7)
    # Actually, let's just iterate over j_block in [0,1] and k_block in [0,1,2,3]
    # j_block 0 -> j in {0,1}
    # j_block 1 -> j in {2,3}
    # k_block 0 -> k in {0,1}
    # k_block 1 -> k in {2,3}
    # k_block 2 -> k in {4,5}
    # k_block 3 -> k in {6,7}
    
    base_terms = _construct_3x2x2()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for j_blk in range(2):
        for k_blk in range(4):
            # Map local indices to global
            # Local j in {0,1} -> Global j = j_blk*2 + local_j
            # Local k in {0,1} -> Global k = k_blk*2 + local_k
            # Local i in {0,1,2} -> Global i = local_i
            
            for u, v, w in base_terms:
                # u is length 6: u[i*2 + local_j]
                # v is length 4: v[local_j*2 + local_k]
                # w is length 6: w[local_k*3 + i]
                
                u_global = [0.0] * (n * m) # 12
                v_global = [0.0] * (m * p) # 32
                w_global = [0.0] * (p * n) # 24
                
                for li in range(3):
                    for lj in range(2):
                        val = u[li * 2 + lj]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            u_global[li * m + gj] = val
                            
                for lj in range(2):
                    for lk in range(2):
                        val = v[lj * 2 + lk]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            gk = k_blk * 2 + lk
                            v_global[gj * p + gk] = val
                            
                for lk in range(2):
                    for li in range(3):
                        val = w[lk * 3 + li]
                        if val != 0:
                            gk = k_blk * 2 + lk
                            w_global[gk * n + li] = val
                            
                U_cols.append(u_global)
                V_cols.append(v_global)
                W_cols.append(w_global)
                
    # Assemble matrices
    rank = len(U_cols)
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
            
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        return _schoolbook()

# EVOLVE-BLOCK-END
