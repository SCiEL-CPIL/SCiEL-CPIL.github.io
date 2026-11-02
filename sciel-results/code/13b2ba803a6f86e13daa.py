import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the full tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m entries), j indexes B (m*p entries), k indexes C (p*n entries)
    # C index k corresponds to output position: k = col*n + row (per trace convention W[k*n+i,r] -> output row i, col k)
    # Actually the convention is W[k*n+i, r] where i is row, k is col of C
    # So C[i, k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta_{i,k} (the multiplication tensor has T[i,j,k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k])
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 (it's always 1 in the standard tensor decomposition sense)
    # The decomposition says: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    # Let's verify by checking that for all i,j,k: sum_r U[i,r]*V[j,r]*W[k,r] == 1
    # But that's only true if the tensor is the all-ones tensor, which it's not.
    # 
    # The correct check: The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # (because C[i,k] = sum_j A[i,j]*B[j,k], and each product A[i,j]*B[j,k] contributes to C[i,k])
    # 
    # Wait, that's the standard Kronecker product tensor. Let me think again.
    # 
    # For matrix multiplication A (n x m) times B (m x p) = C (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # The tensor T of shape (n*m, m*p, n*p) where T[a, b, c] = 1 if the product of A-index a and B-index b contributes to C-index c.
    # a = i*m + j (row i, col j of A)
    # b = j*p + k (row j, col k of B)  
    # c = k*n + i (col k, row i of C) -- per the trace convention W[k*n+i, r]
    # 
    # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j',k), and c corresponds to (k',i') where i=i', k=k', j=j'
    # i.e., T[a, b, c] = 1 iff the j-index of A matches the row-index of B, and the i-index of A matches the row-index of C, and the k-index of B matches the col-index of C.
    
    # Let's build the expected tensor and compare
    T_expected = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1.0
    
    # Compute T_decomposed = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # This is expensive for large rank, so do it efficiently
    T_decomposed = np.zeros((n*m, m*p, n*p))
    for r in range(rank):
        T_decomposed += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    if not np.allclose(T_decomposed, T_expected, atol=1e-10):
        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=200):
    """Try to reduce rank by finding linear dependencies among the rank-1 components."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor
    T = np.zeros((n*m, m*p, n*p))
    for r in range(rank):
        T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Try to find a lower rank decomposition by checking if any component can be removed
    # and compensated by adjusting others
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple approach: try removing each column and see if we can compensate
    # This is a heuristic - for each column r, check if the tensor contribution of that column
    # can be expressed as a linear combination of the others
    
    for r_remove in range(rank):
        # Try removing column r_remove
        U_rem = np.delete(U, r_remove, axis=1)
        V_rem = np.delete(V, r_remove, axis=1)
        W_rem = np.delete(W, r_remove, axis=1)
        
        T_rem = np.zeros((n*m, m*p, n*p))
        for r in range(rank - 1):
            T_rem += np.outer(np.outer(U_rem[:, r], V_rem[:, r]), W_rem[:, r])
        
        # The difference is the contribution of the removed column
        diff = T - T_rem
        
        # Check if diff can be expressed as a sum of rank-1 tensors from the remaining components
        # This is equivalent to checking if diff has rank <= 1 in the appropriate sense
        # For now, just check if diff is zero (which means the column was redundant)
        if np.allclose(diff, 0, atol=1e-10):
            best_rank = rank - 1
            best_U, best_V, best_W = U_rem, V_rem, W_rem
            break
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    n, m, p = 3, 4, 4
    
    # Start with the Strassen-based decomposition from the current program
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # Strassen part: 7 terms for the 2x2 block
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m2 = (c+d)e
            U[0*m + (b_r*2+0)][col] = 0
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            W[(b_c*2+1)*n + 1][col] = -1
            col += 1
            
            # m3 = a(f-h)
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m4 = d(g-e)
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            col += 1
            
            # m5 = (a+b)h
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            col += 1
            
            # m6 = (c-a)(e+f)
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m7 = (b-d)(g+h)
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    # Verify the initial decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback: schoolbook decomposition
        rank_sb = n * m * p
        U_sb = [[0]*rank_sb for _ in range(n*m)]
        V_sb = [[0]*rank_sb for _ in range(m*p)]
        W_sb = [[0]*rank_sb for _ in range(p*n)]
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j][col_sb] = 1
                    V_sb[j*p + k][col_sb] = 1
                    W_sb[k*n + i][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to reduce rank
    reduced = _try_reduce_rank(U, V, W, n, m, p)
    if reduced is not None:
        U, V, W = reduced
    
    return {"U": U, "V": V, "W": W}
