import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T has T[i, j, k] = 1 for all i, j, k? 
    # No, the tensor for matrix multiplication C_ik = sum_j A_ij B_jk
    # is defined such that the bilinear form matches.
    # The seed code implies T[i, j, k] = 1 for all i, j, k is WRONG.
    # The seed code sets U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # This corresponds to the term e_ij \otimes e_jk \otimes e_ki.
    # Summing over all i, j, k gives the tensor with 1s at (i, j, k) in the (ij, jk, ki) indices.
    # This tensor T satisfies T[i, j, k] = 1.
    # Wait, if T[i, j, k] = 1, the rank is 1.
    # Why is the schoolbook rank 105?
    # Because the "matrix multiplication tensor" is NOT the all-ones tensor.
    # The all-ones tensor corresponds to the scalar product of A, B, and C?
    # No.
    # The tensor for matrix multiplication is T[i, j, k] = delta_{?}.
    # Actually, the tensor T is defined by T[i, j, k] = 1 if the indices are consistent with the multiplication.
    # For C_ik = sum_j A_ij B_jk, the tensor has T[i, j, k] = 1 for all i, j, k?
    # If so, the rank is 1.
    # There is a misunderstanding of the tensor definition in the prompt or my head.
    # Let's look at the seed code again.
    # It returns U, V, W.
    # The evaluator checks if the decomposition is exact.
    # If the rank 1 solution (all ones) worked, the score would be 105.
    # But the seed score is 1.0 (rank 105).
    # This implies the rank 1 solution is NOT valid.
    # Why?
    # Because the tensor is NOT all-ones.
    # The tensor T[i, j, k] is 1 only if the triple (i, j, k) is part of the multiplication structure.
    # But for matrix multiplication, ALL triples (i, j, k) are part of the structure.
    # C_ik depends on A_ij and B_jk for all j.
    # So T[i, j, k] = 1 for all i, j, k.
    # If T is all-ones, rank is 1.
    # Is it possible that the indices in W are different?
    # W[k*n+i, r].
    # If the tensor was T[i, j, k] = 1, then U, V, W all ones would work.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # Sum_r 1*1*1 = R.
    # This would give C_ik = R * (sum_j A_ij B_jk)?
    # No, the bilinear form is sum_{i,j,k} T[i,j,k] A_ij B_jk.
    # If T=1, sum is sum A_ij B_jk.
    # This is not C_ik.
    # The decomposition must allow recovering C_ik.
    # The standard tensor decomposition for matrix multiplication uses the tensor T[i, j, k] = delta_{?}.
    # Actually, the tensor is T[i, j, k] = 1 if we consider the identity map?
    # No.
    # The rank of the matrix multiplication tensor is the number of multiplications.
    # The tensor is T in R^{nm} \otimes R^{mp} \otimes R^{np}.
    # The entries are T[(i,j), (j,k), (i,k)] = 1?
    # If so, the bilinear form is sum_{i,j,k} A_ij B_jk C_ik.
    # This is the trace of C^T (A B)?
    # No, sum_{i,j,k} A_ij B_jk C_ik = sum_{i,k} C_ik (sum_j A_ij B_jk) = sum_{i,k} C_ik C_ik = ||C||^2.
    # This is not the multiplication.
    # The multiplication is a map A, B -> C.
    # The tensor decomposition allows computing C using R multiplications.
    # The condition is that there exist U, V, W such that:
    # C_ik = sum_r (sum_j U_ij,r A_ij) (sum_j V_jk,r B_jk) W_ki,r ?
    # No, the indices don't match.
    # The correct form is:
    # C_ik = sum_r U_{i, r} V_{k, r} ...?
    # Let's use the seed code's structure.
    # The seed code implies that the decomposition is valid for rank 105.
    # So I will assume the tensor is defined such that the schoolbook decomposition works.
    # I will use the ALS method to find a lower rank decomposition.
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Define the target tensor T
    # T[i, j, k] = 1 for all i, j, k?
    # If I use the same T as the seed, and the seed works, then T is correct.
    # The seed sets U, V, W such that the sum of rank-1 terms equals T.
    # So I will construct T as the sum of the seed's rank-1 terms.
    # T is a 3D array of shape (n*m, m*p, p*n).
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a decomposition of rank R
    best_rank = n * m * p
    best_decomp = None
    
    # Try ranks from 50 to best_rank
    for R in range(50, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Initialize U, V, W with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # ALS iterations
        for iter in range(100):
            if time.time() > deadline - 5:
                break
                
            # Fix V, W, solve for U
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # For each i, we have a system for U[i, :]
            # T[i, :, :] = U[i, :] @ (V * W)^T ?
            # Let M_VW = V * W (element-wise) -> shape (m*p, R) * (p*n, R) -> No.
            # V is (m*p, R), W is (p*n, R).
            # The term is sum_r U[i, r] V[j, r] W[k, r].
            # This is U[i, r] * (V[j, r] * W[k, r]).
            # Let Z[j, k, r] = V[j, r] * W[k, r].
            # Then T[i, j, k] = sum_r U[i, r] Z[j, k, r].
            # This is a linear system for U[i, r].
            # Flatten T[i, :, :] to (m*p*p*n,).
            # Flatten Z to (m*p*p*n, R).
            # Solve U[i, :] = T[i, :, :].flatten() @ Z.flatten().pinv()?
            
            # This is too large: 15 * 35 * 21 = 11025.
            # R is ~50.
            # 11025 x 50 matrix.
            # Solving 3 times (for U, V, W) per iteration.
            # 100 iterations.
            # 300 solves.
            # Each solve is small.
            
            # Construct Z
            Z = np.zeros((m*p*p*n, R))
            for r in range(R):
                Z[:, r] = (V[:, r] * W[:, r]).reshape(m*p, p*n).flatten() # Wrong shape?
                # V is (m*p, R). W is (p*n, R).
                # We need V[j, r] * W[k, r] for all j, k.
                # This is an outer product of V[:, r] and W[:, r]?
                # No, V index is j, W index is k.
                # So yes, outer product.
                Z[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve for U
            # T[i, :, :] is (m*p, p*n). Flatten to (m*p*p*n,).
            for i in range(n*m):
                rhs = T[i, :, :].flatten()
                # U[i, :] = Z.T @ U? No.
                # rhs = Z @ U[i, :]
                # U[i, :] = np.linalg.lstsq(Z, rhs, rcond=None)[0]
                # This is 3 times (n*m) solves.
                # n*m = 15.
                # 15 solves of 11025 x R.
                
                U[i, :] = np.linalg.lstsq(Z, rhs, rcond=None)[0]
                
            if time.time() > deadline - 5:
                break
                
            # Similarly for V
            # T[i, j, k] = sum_r V[j, r] U[i, r] W[k, r]
            # For each j, solve for V[j, :]
            Z2 = np.zeros((n*m*p*n, R))
            for r in range(R):
                Z2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            for j in range(m*p):
                rhs = T[:, j, :].flatten() # Shape (n*m, p*n) -> (n*m*p*n,)
                V[j, :] = np.linalg.lstsq(Z2, rhs, rcond=None)[0]
                
            if time.time() > deadline - 5:
                break
                
            # Similarly for W
            # T[i, j, k] = sum_r W[k, r] U[i, r] V[j, r]
            # For each k, solve for W[k, :]
            Z3 = np.zeros((n*m*m*p, R))
            for r in range(R):
                Z3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            for k in range(p*n):
                rhs = T[:, :, k].flatten() # Shape (n*m, m*p) -> (n*m*m*p,)
                W[k, :] = np.linalg.lstsq(Z3, rhs, rcond=None)[0]
                
        # Check error
        # Compute the reconstructed tensor
        # T_rec[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # This is expensive.
        # Instead, check the Frobenius norm of T - T_rec.
        # We can compute the error on a subset or use the residual.
        # For simplicity, compute the full error if time permits.
        
        # Efficient error calculation:
        # err = 0
        # for i in range(n*m):
        #   for j in range(m*p):
        #     for k in range(p*n):
        #       val = np.dot(U[i, :], V[j, :] * W[k, :])
        #       err += (T[i, j, k] - val)**2
        # This is 11025 * R operations.
        # 11025 * 50 = 551250. Fast.
        
        err = 0.0
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = np.dot(U[i, :], V[j, :] * W[k, :])
                    err += (T[i, j, k] - val)**2
        
        if err < 1e-6:
            best_rank = R
            best_decomp = (U, V, W)
            break
    if best_decomp is not None:
        U, V, W = best_decomp
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
