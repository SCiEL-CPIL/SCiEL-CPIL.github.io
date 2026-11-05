import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition (rank = n*m*p = 36)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Verify schoolbook
    if not verify_decomposition(U_school, V_school, W_school, n, m, p, rank_school):
        # This should never happen
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find a lower rank decomposition
    # Known lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Known upper bound: schoolbook = 36
    # Strassen for 3x3x3 is 21, but we have 3x3x4
    # Let's try ranks from 12 to 35
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try random search for lower ranks
    # Use a simple approach: random initialization + local optimization
    
    for target_rank in range(12, 30):
        if time.time() - start_time > 110:
            break
        
        # Try a few random restarts
        for attempt in range(3):
            if time.time() - start_time > 115:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Optimize to minimize error
            # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            #        + sum_{other entries} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            
            # For matrix multiplication, the tensor has 1s where i,j,k are valid indices
            # and 0s elsewhere. But for n*m*p dimensional tensor, all entries where
            # i in [0,n), j in [0,m), k in [0,p) should be 1 if we're doing A*B.
            # Wait, no. The matrix multiplication tensor T satisfies:
            # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
            # And all other entries of the full tensor (if we embed in a larger space) are 0.
            # But here the tensor is exactly n*m*p dimensional in each mode, so all entries
            # that correspond to valid i,j,k are 1, and there are no "other" entries.
            # Actually, the tensor is defined on {0,...,n-1} x {0,...,m-1} x {0,...,p-1}
            # and T[i,j,k] = 1 for all i,j,k. That's not right either.
            
            # Let me re-read the task. "Matrix multiplication 3,3,4"
            # A is 3x3, B is 3x4, C is 3x4.
            # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
            # The tensor is T[i, j, k] = 1 for all i in [0,3), j in [0,3), k in [0,4)
            # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # U index: i*m + j, V index: j*p + k, W index: k*n + i
            
            # So for all i in [0,n), j in [0,m), k in [0,p):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a system of n*m*p equations.
            
            # Let's use gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() - start_time > 118:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            # Compute current value
                            val = 0.0
                            for r in range(target_rank):
                                val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                            
                            error = val - 1.0
                            
                            # Gradient with respect to U[idx_u, r]
                            # d/dU[idx_u,r] (U[idx_u,r]*V[idx_v,r]*W[idx_w,r]) = V[idx_v,r]*W[idx_w,r]
                            for r in range(target_rank):
                                grad_U[idx_u, r] += 2 * error * V[idx_v, r] * W[idx_w, r]
                                grad_V[idx_v, r] += 2 * error * U[idx_u, r] * W[idx_w, r]
                                grad_W[idx_w, r] += 2 * error * U[idx_u, r] * V[idx_v, r]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 100 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-6:
                        # Try to round to half-integers and verify
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                            best_rank = target_rank
                            best_U = U_rounded.copy()
                            best_V = V_rounded.copy()
                            best_W = W_rounded.copy()
                        break
            
            # Check if we found a valid decomposition
            if best_rank <= target_rank:
                break
    
    # Return the best decomposition found
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
