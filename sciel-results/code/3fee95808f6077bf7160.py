# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    
    def find_3x3x3_decomp(target_rank, max_restarts=10, max_iters=50):
        n3, m3, p3 = 3, 3, 3
        T = np.zeros((n3*m3, m3*p3, p3*n3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T[i*m3+j, j*p3+k, k*n3+i] = 1.0
        
        best_resid = float('inf')
        best_U, best_V, best_W = None, None, None
        
        start_time = time.time()
        time_limit = 60
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            rng = np.random.RandomState(restart * 1000 + 42)
            U = rng.randn(n3*m3, target_rank)
            V = rng.randn(m3*p3, target_rank)
            W = rng.randn(p3*n3, target_rank)
            
            for it in range(max_iters):
                # Update W
                # M is (n3*m3) x (target_rank) where M[:, c] = U[:, c] * V[:, c] (element-wise? No, Kronecker)
                # Wait, T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                # For fixed k, T[:, :, k] (shape 3x3) = sum_c W[k,c] (U[:,c] V[:,c]^T)
                # Let's flatten T[:, :, k] to vector of length 9.
                # Let M_k be a matrix of size 9 x target_rank.
                # M_k[:, c] = U[:, c] * V[:, c] ? No.
                # U[:, c] is a vector of length 9 (flattened i,j? No, U is 9xR).
                # U[i, c] is scalar. V[j, c] is scalar.
                # The term is U[i,c] * V[j,c] * W[k,c].
                # So for fixed k, we have a matrix T_k (3x3).
                # T_k[i,j] = sum_c (U[i,c] V[j,c]) W[k,c].
                # This is a linear combination of rank-1 matrices E_c = U[:,c] V[:,c]^T.
                # So T_k = sum_c W[k,c] E_c.
                # Vectorize T_k to t_k (length 9).
                # Construct matrix E (9 x R) where E[:, c] = vec(E_c) = U[:,c] * V[:,c] (element-wise product of vectors? No).
                # vec(u v^T) = u \otimes v.
                # So E[:, c] = U[:, c] * V[:, c] ?
                # U is 9xR. V is 9xR.
                # U[:, c] is a vector of length 9.
                # V[:, c] is a vector of length 9.
                # The Kronecker product of two vectors of length 3 is length 9.
                # But U is already flattened.
                # Let's stick to the matrix form.
                # T_k = sum_c W[k,c] (U_col_c * V_col_c^T).
                # This is a linear least squares problem for W[k,:].
                # We can solve for all k at once.
                
                # Construct E matrix (9 x R)
                # E[i*m3+j, c] = U[i, c] * V[j, c] ?
                # No, U is indexed by i (0..2) and j (0..2)?
                # U is 9xR. Index i*m3+j.
                # So U[i*m3+j, c] is the value for (i,j).
                # But the term is U[i,c] * V[j,c].
                # This implies U and V are NOT flattened in the same way for the Kronecker.
                # In the tensor definition:
                # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
                # Wait, the indices in the problem description:
                # U[i*m+j, c]
                # V[j*p+k, c]
                # W[k*n+i, c]
                
                # So for 3x3x3:
                # U is 9xR. Index i*3+j.
                # V is 9xR. Index j*3+k.
                # W is 9xR. Index k*3+i.
                
                # The term is U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c].
                
                # This is NOT a simple Kronecker product of U and V vectors.
                # Because the indices are coupled.
                # U depends on i,j. V depends on j,k. W depends on k,i.
                
                # So ALS is more complex.
                # Fix U, V. Solve for W.
                # T[i,j,k] = sum_c U[i*3+j, c] V[j*3+k, c] W[k*3+i, c].
                # Let's fix i, j.
                # T[i,j,k] for k=0,1,2.
                # This is a linear system for W[0*3+i, c], W[1*3+i, c], W[2*3+i, c].
                # Actually, W is a 9xR matrix.
                # Let's collect terms for W.
                # For a fixed c, the coefficient of W[k*3+i, c] is U[i*3+j, c] V[j*3+k, c].
                
                # This is a linear system for the vector W_flat (length 9*R).
                # T_flat (length 27) = A W_flat.
                # A is 27 x (9*R).
                # This is solvable by least squares.
                
                # Construct A
                # A[row, col]
                # row = i*9 + j*3 + k (index in T)
                # col = k*3 + i, c (index in W) -> (k*3+i)*R + c
                # A[row, col] = U[i*3+j, c] * V[j*3+k, c]
                
                # This is O(27 * 9 * R * R) to construct?
                # R=23. 27 * 9 * 23 * 23 ~ 130k. Fast.
                
                A = np.zeros((27, 9 * target_rank))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            row = i*9 + j*3 + k
                            # For each c
                            for c in range(target_rank):
                                col = (k*3 + i) * target_rank + c
                                A[row, col] = U[i*3+j, c] * V[j*3+k, c]
                
                T_flat = T.flatten()
                # Solve A W_flat = T_flat
                W_flat, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=None)
                W = W_flat.reshape(9, target_rank)
                
                # Similarly for U and V
                # Construct A for U
                # T[i,j,k] = sum_c U[i*3+j, c] V[j*3+k, c] W[k*3+i, c]
                # Fix j, k. Solve for U[i*3+j, c] for i=0,1,2.
                # Actually, U is 9xR.
                # Let's solve for the whole U.
                # A_U is 27 x (9*R).
                # A_U[row, col] where col = (i*3+j)*R + c
                # A_U[row, col] = V[j*3+k, c] * W[k*3+i, c]
                
                A_U = np.zeros((27, 9 * target_rank))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            row = i*9 + j*3 + k
                            for c in range(target_rank):
                                col = (i*3 + j) * target_rank + c
                                A_U[row, col] = V[j*3+k, c] * W[k*3+i, c]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
                U = U_flat.reshape(9, target_rank)
                
                # Similarly for V
                A_V = np.zeros((27, 9 * target_rank))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            row = i*9 + j*3 + k
                            for c in range(target_rank):
                                col = (j*3 + k) * target_rank + c
                                A_V[row, col] = U[i*3+j, c] * W[k*3+i, c]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
                V = V_flat.reshape(9, target_rank)
                
                # Check residual
                # T_recon = sum_c U[i*3+j, c] V[j*3+k, c] W[k*3+i, c]
                # This is expensive to compute fully.
                # We can check norm of T_flat - A_W W_flat etc.
                # Or just compute once per iteration.
                
                pass # Skip residual check for speed in thought process, will add in code
            
            # Final check
            # Compute residual
            T_recon = np.zeros((27,))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        idx = i*9 + j*3 + k
                        val = 0
                        for c in range(target_rank):
                            val += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                        T_recon[idx] = val
            resid = np.linalg.norm(T_flat - T_recon)
            
            if resid < best_resid:
                best_resid = resid
                best_U, best_V, best_W = U, V, W
                
        # If best_resid is small, round and verify
        if best_U is not None and best_resid < 1e-4:
            # Round to nearest 0.5
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify
            T_recon = np.zeros((27,))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        idx = i*9 + j*3 + k
                        val = 0
                        for c in range(target_rank):
                            val += U_rounded[i*3+j, c] * V_rounded[j*3+k, c] * W_rounded[k*3+i, c]
                        T_recon[idx] = val
            if np.allclose(T_recon, T_flat, atol=1e-9):
                return U_rounded, V_rounded, W_rounded
                
        return None

    # Try to find rank 23
    decomp_3x3 = find_3x3x3_decomp(23)
    
    if decomp_3x3 is not None:
        U3, V3, W3 = decomp_3x3
        R3 = 23
        # Construct 3x3x6 decomp
        # Rank 2 * R3 = 46
        R = 2 * R3
        U = np.zeros((9, R))
        V = np.zeros((18, R))
        W = np.zeros((18, R))
        
        for c in range(R3):
            # Block 1 (k=0,1,2)
            U[:, c] = U3[:, c]
            V[0:9, c] = V3[:, c]
            V[9:18, c] = 0
            W[0:9, c] = W3[:, c]
            W[9:18, c] = 0
            
            # Block 2 (k=3,4,5)
            U[:, c + R3] = U3[:, c]
            V[0:9, c + R3] = 0
            V[9:18, c + R3] = V3[:, c]
            W[0:9, c + R3] = 0
            W[9:18, c + R3] = W3[:, c]
            
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
