import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = 2, 4, 4
    
    # Build multiplication tensor: T[a,b,c]=1 where a=i*m+k, b=k*p+j, c=i*p+j
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, i*p+j] = 1
    
    def verify(U, V, W, r):
        """Verify that the decomposition is correct."""
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    s = 0
                    for s_idx in range(r):
                        s += U[a][s_idx] * V[b][s_idx] * W[c][s_idx]
                    if s != T[a, b, c]:
                        return False
        return True
    
    # Known rank-28 Strassen-based solution as fallback
    def strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1

            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1

            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1

            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1

            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1

            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1

            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
        X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
        X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
        X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]

        add_block(0,  X1, Y1, Z1)
        add_block(7,  X2, Y2, Z2)
        add_block(14, X3, Y3, Z3)
        add_block(21, X4, Y4, Z4)
        
        return U, V, W, rank
    
    # Attempt to find a lower-rank decomposition using linear algebra over Q
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 8 x 16 x 8
    # Flatten along mode 2: T_flat is (8*8) x 16 = 64 x 16
    # The rank of the tensor >= rank of any flattening
    # Also, the tensor can be seen as a linear map from R^{16} to R^{64}
    # via (B) -> A*B, which has rank at most 8 (since A is 2x4, output is 2x4=8 elements)
    # But we need the tensor rank, not the linear map rank
    
    # For a rank-r decomposition: T[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]
    # This means for each s, the outer product U[:,s] V[:,s] W[:,s] contributes
    # Equivalently, T can be written as sum_s (U[:,s] ⊗ V[:,s]) W[:,s]^T in some sense
    
    # Key insight: Fix W. Then for each c, the matrix M_c[a,b] = T[a,b,c] 
    # must equal sum_s U[a,s] V[b,s] W[c,s]
    # So M_c = W[c,:]^T * (U^T ⊗ I) ... no, that's not quite right
    # Actually: M_c = sum_s W[c,s] * (U[:,s] V[:,s]^T)
    # So the set {M_c : c=0..p*n-1} lies in the span of {U[:,s] V[:,s]^T : s=0..r-1}
    # Each M_c is an 8x16 matrix
    
    # Let's try: for a given rank r, pick W with random +/-1 entries
    # Then for each c, M_c must be in span of {u_s v_s^T}
    # This means the 8x16 matrices M_c all have rank <= r (as matrices)
    # AND they all lie in an r-dimensional subspace of 8x16 matrices
    
    # Better approach: use the fact that if we fix U and V, then for each c,
    # M_c = sum_s W[c,s] u_s v_s^T
    # So W[c,s] = <M_c, u_s v_s^T> / ||u_s v_s^T||^2 if we had orthonormal basis
    # But u_s v_s^T are not necessarily orthogonal
    
    # Let's try a direct search: for small r, try to find U, V, W
    # with entries in {-1, 0, 1} or {-1, -1/2, 0, 1/2, 1}
    
    def search_rank(r, max_trials=5000):
        """Search for rank-r decomposition with small integer/half-integer entries."""
        if time.time() > deadline - 5:
            return None
        
        # For each trial, randomly initialize U and V with sparse +/-1 entries
        # Then solve for W using least squares, check if exact
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(12345 + trial)
            
            # Initialize U and V with sparse structure
            U_mat = np.zeros((n*m, r), dtype=np.int64)
            V_mat = np.zeros((m*p, r), dtype=np.int64)
            
            # Each column of U and V should have a few non-zero entries
            for s in range(r):
                nz_u = np.random.randint(2, min(4, n*m))
                idx_u = np.random.choice(n*m, nz_u, replace=False)
                U_mat[idx_u, s] = np.random.choice([-1, 1], nz_u)
                
                nz_v = np.random.randint(2, min(4, m*p))
                idx_v = np.random.choice(m*p, nz_v, replace=False)
                V_mat[idx_v, s] = np.random.choice([-1, 1], nz_v)
            
            # Now for each c, M_c[a,b] = sum_s U_mat[a,s] V_mat[b,s] W[c,s]
            # This is linear in W. Let's set up the system.
            # For each (a,b,c): sum_s U_mat[a,s] V_mat[b,s] W[c,s] = T[a,b,c]
            # Group by c: for fixed c, we have an 8x16 system in W[c,:]
            # But W[c,s] appears in all (a,b) equations for that c
            # So for each c, we need: sum_s W[c,s] * (U_mat[:,s] V_mat[:,s]^T) = M_c
            
            # This is a linear system: A_c @ W[c,:] = vec(M_c)
            # where A_c is 128 x r (for each (a,b) pair)
            
            W_mat = np.zeros((p*n, r), dtype=np.int64)
            valid = True
            
            for c in range(p*n):
                # Build the 128 x r matrix A_c
                # A_c[(a,b), s] = U_mat[a,s] * V_mat[b,s]
                A_c = np.zeros((n*m * m*p, r), dtype=np.int64)
                rhs = np.zeros(n*m * m*p, dtype=np.int64)
                
                for a in range(n*m):
                    for b in range(m*p):
                        idx = a * m*p + b
                        for s in range(r):
                            A_c[idx, s] = U_mat[a, s] * V_mat[b, s]
                        rhs[idx] = T[a, b, c]
                
                # Solve A_c @ W_c = rhs using least squares
                # Since entries are small integers, try to find exact solution
                W_c, residuals, rank_c, sv = np.linalg.lstsq(A_c.astype(float), rhs.astype(float), rcond=None)
                
                # Check if solution is close to integer/half-integer
                W_c_rounded = np.round(W_c * 2) / 2
                if not np.allclose(A_c @ W_c_rounded, rhs, atol=1e-9):
                    valid = False
                    break
                
                W_mat[c, :] = W_c_rounded
            
            if not valid:
                continue
            
            # Verify the full decomposition
            U_list = [list(U_mat[a, :]) for a in range(n*m)]
            V_list = [list(V_mat[b, :]) for b in range(m*p)]
            W_list = [list(W_mat[c, :]) for c in range(p*n)]
            
            if verify(U_list, V_list, W_list, r):
                return U_list, V_list, W_list, r
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for r in range(16, 29):
        if time.time() > deadline - 10:
            break
        result = search_rank(r, max_trials=3000)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-28 Strassen solution
    U, V, W, rank = strassen_28()
    return {"U": U, "V": V, "W": W}
