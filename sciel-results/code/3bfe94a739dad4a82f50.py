"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
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

    start_time = time.time()
    deadline = start_time + 140.0

    # Search for lower rank
    # Known lower bounds suggest rank might be around 21-25 for 3x3x5
    # We search from 21 upwards
    for R in range(21, 45):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts
        # Increase restarts for lower ranks as they are harder
        restarts = 50 if R < 25 else 20
        
        for restart in range(restarts):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + R * 1000 + restart)
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Reshape for easier tensor operations
            # U: (n, m, R), V: (m, p, R), W: (p, n, R)
            U_t = U.reshape(n, m, R)
            V_t = V.reshape(m, p, R)
            W_t = W.reshape(p, n, R)
            
            # Target tensor T: (n, m, p) with T[i,j,k] = 1
            T = np.ones((n, m, p))
            
            converged = False
            for it in range(200):
                # Update W
                # W_t[k,i,r] = sum_{j} T[i,j,k] * U_t[i,j,r] * V_t[j,k,r] / sum_{j} (U_t[i,j,r] * V_t[j,k,r])^2
                # Numerator: sum_j T[i,j,k] U_t[i,j,r] V_t[j,k,r]
                # Since T=1, Num = sum_j U_t[i,j,r] V_t[j,k,r]
                # Denom = sum_j (U_t[i,j,r] V_t[j,k,r])^2
                
                # Efficient computation using einsum
                # Num_W[k,i,r] = sum_j U_t[i,j,r] * V_t[j,k,r]
                Num_W = np.einsum('ijr,jkr->kir', U_t, V_t)
                Den_W = np.einsum('ijr,jkr->kir', U_t**2, V_t**2)
                
                # Avoid division by zero
                mask = Den_W > 1e-10
                W_t_new = np.zeros_like(W_t)
                W_t_new[mask] = Num_W[mask] / Den_W[mask]
                W_t = W_t_new
                
                # Update U
                # U_t[i,j,r] = sum_k T[i,j,k] W_t[k,i,r] V_t[j,k,r] / sum_k (W_t[k,i,r] V_t[j,k,r])^2
                Num_U = np.einsum('kir,jkr->ijr', W_t, V_t)
                Den_U = np.einsum('kir,jkr->ijr', W_t**2, V_t**2)
                
                U_t_new = np.zeros_like(U_t)
                mask = Den_U > 1e-10
                U_t_new[mask] = Num_U[mask] / Den_U[mask]
                U_t = U_t_new
                
                # Update V
                # V_t[j,k,r] = sum_i T[i,j,k] U_t[i,j,r] W_t[k,i,r] / sum_i (U_t[i,j,r] W_t[k,i,r])^2
                Num_V = np.einsum('ijr,kir->jkr', U_t, W_t)
                Den_V = np.einsum('ijr,kir->jkr', U_t**2, W_t**2)
                
                V_t_new = np.zeros_like(V_t)
                mask = Den_V > 1e-10
                V_t_new[mask] = Num_V[mask] / Den_V[mask]
                V_t = V_t_new
                
                # Check residual
                # Reconstruct tensor
                T_rec = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                err = np.max(np.abs(T_rec - T))
                
                if err < 1e-8:
                    # Try to round to half-integers
                    U_flat = U_t.reshape(-1)
                    V_flat = V_t.reshape(-1)
                    W_flat = W_t.reshape(-1)
                    
                    U_int = np.round(U_flat * 2) / 2
                    V_int = np.round(V_flat * 2) / 2
                    W_int = np.round(W_flat * 2) / 2
                    
                    # Verify exactness with integers/half-integers
                    U_int_t = U_int.reshape(n, m, R)
                    V_int_t = V_int.reshape(m, p, R)
                    W_int_t = W_int.reshape(p, n, R)
                    
                    T_rec_int = np.einsum('ijr,jkr,kir->ijk', U_int_t, V_int_t, W_int_t)
                    err_int = np.max(np.abs(T_rec_int - T))
                    
                    if err_int < 1e-9:
                        return {
                            "U": U_int.reshape(n*m, R).tolist(),
                            "V": V_int.reshape(m*p, R).tolist(),
                            "W": W_int.reshape(p*n, R).tolist()
                        }
                    else:
                        # If rounding failed, keep searching or try other restarts
                        # But if we found a real solution, it's a good sign
                        # We can break and return real? No, must be half-integer.
                        # Continue to next restart or rank
                        break
                
                if time.time() > deadline - 5.0:
                    break

    return schoolbook()

# EVOLVE-BLOCK-END
