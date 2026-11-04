import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial))
    V_trivial = np.zeros((mp, rank_trivial))
    W_trivial = np.zeros((pn, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    
    # Target ranks to try, starting from lower (better) to higher
    # We know 90 works. Let's try to beat it.
    # Based on literature, 3x5x6 might be around 50-60.
    target_ranks = [49, 50, 52, 55, 60, 65, 70, 75, 80]
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s for safety/verification
    
    # Target tensor
    T = np.ones((n, m, p))
    
    for R in target_ranks:
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R + attempt) # Reproducibility for debugging, but random is fine
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            lr = 0.05
            # Use a simple gradient descent with momentum? Or just plain.
            # Vectorized gradient computation
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current tensor: sum_r U_r o V_r o W_r
                # U is (nm, R), V is (mp, R), W is (pn, R)
                # Reshape to (n, m, R), (m, p, R), (p, n, R)
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R)
                
                # Tensor product sum
                # current[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                current = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                error = T - current
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-6:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness with rounded values
                    U_3d_r = U_r.reshape(n, m, R)
                    V_3d_r = V_r.reshape(m, p, R)
                    W_3d_r = W_r.reshape(p, n, R)
                    current_r = np.einsum('ijr,jkr,kir->ijk', U_3d_r, V_3d_r, W_3d_r)
                    
                    if np.allclose(current_r, T, atol=1e-9):
                        # Success!
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        break # Break out of step loop
                    else:
                        # Rounding failed, continue optimizing
                        pass
                
                # Compute gradients
                # dU[i,j,r] = sum_k error[i,j,k] * V[j,k,r] * W[k,i,r]
                # dV[j,k,r] = sum_i error[i,j,k] * U[i,j,r] * W[k,i,r]
                # dW[k,i,r] = sum_j error[i,j,k] * U[i,j,r] * V[j,k,r]
                
                dU = np.einsum('ijk,jkr,kir->ijr', error, V_3d, W_3d)
                dV = np.einsum('ijk,ijr,kir->jkr', error, U_3d, W_3d)
                dW = np.einsum('ijk,ijr,jkr->kir', error, U_3d, V_3d)
                
                # Reshape gradients back to 2D
                dU_2d = dU.reshape(nm, R)
                dV_2d = dV.reshape(mp, R)
                dW_2d = dW.reshape(pn, R)
                
                # Update
                U -= lr * dU_2d
                V -= lr * dV_2d
                W -= lr * dW_2d
                
                # Optional: Normalize to prevent explosion
                # U /= np.linalg.norm(U, axis=0, keepdims=True)
                
            if best_rank == R:
                break # Found a solution for this rank, try lower rank next
        
        if time.time() > deadline - 5:
            break
    
    # Convert best result to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
