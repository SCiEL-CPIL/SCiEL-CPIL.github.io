"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For all i (0..n-1), l (0..m-1), k (0..p-1):
    # sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, m, p, n, col):
    """Apply Strassen's 7-product scheme to a 2x2x2 block.
    i0,i1: the two i indices
    j0,j1: the two j (l) indices
    k0,k1: the two k indices
    Returns new col index."""
    # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    U[i0*m + j0][col] += 1
    U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    U[i1*m + j0][col] += 1
    U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + i1][col] += 1
    W[k1*n + i1][col] -= 1
    col += 1
    
    # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    U[i0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + i0][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
    U[i1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + i0][col] += 1
    W[k0*n + i1][col] += 1
    col += 1
    
    # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    U[i0*m + j0][col] += 1
    U[i0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] -= 1
    W[k1*n + i0][col] += 1
    col += 1
    
    # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    U[i1*m + j0][col] += 1
    U[i0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    U[i0*m + j1][col] += 1
    U[i1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin before 150s kill
    
    # Strategy: Use Strassen on 2x2x2 blocks where possible.
    # j indices: 0,1,2,3,4
    # k indices: 0,1,2,3,4,5
    # i indices: 0,1
    #
    # Full 2x2x2 Strassen blocks: (i=0,1) x (j pair) x (k pair)
    # j pairs: (0,1), (2,3) -> 2 pairs
    # k pairs: (0,1), (2,3), (4,5) -> 3 pairs
    # Total 2x2x2 blocks: 2*3 = 6, each saves 1 rank over 4 scalar products
    #
    # Remaining: j=4 with all k=0..5 -> 2*6 = 12 scalar products
    # 
    # Total rank: 6*7 + 12 = 42 + 12 = 54
    #
    # Can we do better? For j=4, we have 2x1x6 sub-problem.
    # The 2x1x6 tensor: entries (i,4,k) for i in {0,1}, k in {0..5}
    # This is just 12 independent rank-1 terms (each A[i,4]*B[4,k] appears in C[i,k]).
    # Minimum rank for 2x1x6 is 12 (they're all independent).
    #
    # Wait, but can we combine across different j blocks?
    # The key constraint: U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
    # Each term r contributes to ALL (i,l,k) triples simultaneously.
    # 
    # So a single term r with U[i0*m+l0, r]=a, V[l0*p+k0, r]=b, W[k0*n+i0, r]=c
    # contributes a*b*c to C[i0, l0, k0] only (since other entries in U,V,W for this r are 0).
    # Actually no - a term r can have non-zero entries in multiple U, V, W positions.
    # 
    # The Strassen trick works because a single term r can have multiple non-zero U entries,
    # multiple non-zero V entries, and multiple non-zero W entries, and the product
    # U[i,l,r]*V[l,k,r]*W[k,i,r] gives the right combination for multiple (i,l,k) triples.
    #
    # For the j=4 case: we need terms that contribute to (i,4,k) for various i,k.
    # A term with U[0*m+4,r]=a0, U[1*m+4,r]=a1, V[4*p+k0,r]=b0, V[4*p+k1,r]=b1,
    # W[k0*n+0,r]=c00, W[k0*n+1,r]=c01, W[k1*n+0,r]=c10, W[k1*n+1,r]=c11
    # contributes:
    #   to (0,4,k0): a0*b0*c00
    #   to (0,4,k1): a0*b1*c10
    #   to (1,4,k0): a1*b0*c01
    #   to (1,4,k1): a1*b1*c11
    # 
    # We need these to equal 1 for the "diagonal" and 0 for others... 
    # Actually we need the SUM over all terms to give 1 for each (i,4,k) and 0 elsewhere.
    #
    # This is exactly a 2x1x2 sub-problem embedded in the full tensor.
    # The 2x1x2 tensor T[i,4,k] = A[i,4]*B[4,k] has rank 4 (4 independent rank-1 terms).
    # So we can't save any rank here.
    #
    # Therefore rank 54 is likely optimal for this block structure.
    #
    # Let me try a different approach: can we find a rank-53 or lower decomposition?
    # This would require a more clever global structure.
    #
    # For now, let's implement the rank-54 decomposition and verify it.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(U, V, W, 0, 1, j0, j1, k0, k1, m, p, n, col)
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Verify
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, best_rank):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
