import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def _find_low_rank_233(target_rank, max_time=30.0):
    """
    Tries to find a rank-`target_rank` decomposition for M_{2,3,3}.
    Returns (U, V, W) if found and valid, else None.
    """
    n, m, p = 2, 3, 3
    T = np.ones((n, m, p))
    
    start_time = time.time()
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() - start_time > max_time:
            break
            
        np.random.seed(restart * 100 + target_rank)
        # Initialize
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
            U_curr = U_flat.reshape(n*m, target_rank)
            V_curr = V_flat.reshape(m*p, target_rank)
            W_curr = W_flat.reshape(p*n, target_rank)
            
            # Compute tensor reconstruction
            # T_recon[i, l, k] = sum_c U[i*m+l, c] V[l*p+k, c] W[k*n+i, c]
            # We can compute this efficiently
            # Sum over c of (U_c * V_c) * W_c? No.
            # T_recon = np.zeros((n, m, p))
            # for c in range(target_rank):
            #    u_c = U_curr[:, c].reshape(n, m)
            #    v_c = V_curr[:, c].reshape(m, p)
            #    w_c = W_curr[:, c].reshape(p, n)
            #    # T_recon += u_c[i,l] * v_c[l,k] * w_c[k,i]
            #    # This is a 3-tensor product.
            #    # We can use einsum or loops.
            #    # For small sizes, loops are fine.
            #    for i in range(n):
            #        for l in range(m):
            #            for k in range(p):
            #                T_recon[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
            #                
            # # Faster:
            # # T_recon[i,l,k] = sum_c U[i*m+l, c] V[l*p+k, c] W[k*n+i, c]
            # # Let M[i,l, c] = U[i*m+l, c]
            # # Let N[l,k, c] = V[l*p+k, c]
            # # Let P[k,i, c] = W[k*n+i, c]
            # # T_recon[i,l,k] = sum_c M[i,l,c] N[l,k,c] P[k,i,c]
            
            # Vectorized approach:
            # U_curr is (n*m, r)
            # V_curr is (m*p, r)
            # W_curr is (p*n, r)
            
            # Create index arrays
            i_idx = np.repeat(np.arange(n), m*p)
            l_idx = np.tile(np.repeat(np.arange(m), p), n)
            k_idx = np.tile(np.arange(p), n*m)
            
            # Indices for U, V, W
            u_idx = i_idx * m + l_idx
            v_idx = l_idx * p + k_idx
            w_idx = k_idx * n + i_idx
            
            # x is concatenated.
            # U_part = x[0 : n*m*r]
            # V_part = x[n*m*r : n*m*r + m*p*r]
            # W_part = x[n*m*r + m*p*r : ]
            
            # Reshape parts to (n*m, r), etc?
            # Actually, x is flat.
            # U_flat[c, idx] = x[idx * r + c]
            # So U_val = U_flat[u_idx * r + np.arange(r)[:, None]]?
            
            # Let's just use the loop for simplicity and correctness in the objective,
            # as n*m*p is small (18).
            T_recon = np.zeros((n, m, p))
            for c in range(target_rank):
                u_c = U_curr[:, c].reshape(n, m)
                v_c = V_curr[:, c].reshape(m, p)
                w_c = W_curr[:, c].reshape(p, n)
                # T_recon[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
                # Use einsum for speed
                # T_recon += np.einsum('il,lk,ki->ilk', u_c, v_c, w_c)
                T_recon += np.einsum('il,lk,ki->ilk', u_c, v_c, w_c)
                
            diff = T - T_recon
            return np.sum(diff * diff)
            
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success and res.fun < 1e-6:
            # Extract solution
            U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_sol = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Round to half-integers
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            # Verify exactness
            T_exact = np.zeros((n, m, p))
            for c in range(target_rank):
                u_c = U_round[:, c].reshape(n, m)
                v_c = V_round[:, c].reshape(m, p)
                w_c = W_round[:, c].reshape(p, n)
                T_exact += np.einsum('il,lk,ki->ilk', u_c, v_c, w_c)
                
            if np.allclose(T, T_exact, atol=1e-9):
                return U_round, V_round, W_round
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U_school[i * m + l, col] = 1
                V_school[l * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find a better solution using the block structure
    # M_{2,3,9} can be decomposed into 3 blocks of M_{2,3,3}
    # If we find rank r for M_{2,3,3}, we get rank 3r for M_{2,3,9}
    
    for r in range(15, 19): # Try ranks 15, 16, 17, 18
        if time.time() > deadline - 10:
            break
            
        print(f"Trying rank {r} for M_{2,3,3}...")
        factors = _find_low_rank_233(r, max_time=20.0)
        
        if factors is not None:
            U_3, V_3, W_3 = factors
            new_rank = 3 * r
            print(f"Found rank {new_rank} for M_{2,3,9} via M_{2,3,3} rank {r}")
            
            # Construct M_{2,3,9} factors
            U = np.zeros((n * m, new_rank))
            V = np.zeros((m * p, new_rank))
            W = np.zeros((p * n, new_rank))
            
            # U is repeated 3 times
            for j in range(3):
                U[:, j*r : (j+1)*r] = U_3
                
            # V and W are block-diagonal in a sense
            # V rows: l * p + k. p=9.
            # V_3 rows: l * 3 + k'.
            # For block j, k = 3*j + k'.
            # V[l * 9 + (3*j + k'), j*r + c] = V_3[l * 3 + k', c]
            for l in range(m):
                for k_prime in range(3):
                    for j in range(3):
                        k = 3 * j + k_prime
                        v_idx = l * p + k
                        v3_idx = l * 3 + k_prime
                        V[v_idx, j*r : (j+1)*r] = V_3[v3_idx, :]
                        
            # W rows: k * n + i. p=9, n=2.
            # W_3 rows: k' * 2 + i.
            # For block j, k = 3*j + k'.
            # W[(3*j + k') * 2 + i, j*r + c] = W_3[k' * 2 + i, c]
            for i in range(n):
                for k_prime in range(3):
                    for j in range(3):
                        k = 3 * j + k_prime
                        w_idx = k * n + i
                        w3_idx = k_prime * 2 + i
                        W[w_idx, j*r : (j+1)*r] = W_3[w3_idx, :]
                        
            # Verify the constructed solution
            T_target = np.ones((n, m, p))
            T_recon = np.zeros((n, m, p))
            for c in range(new_rank):
                u_c = U[:, c].reshape(n, m)
                v_c = V[:, c].reshape(m, p)
                w_c = W[:, c].reshape(p, n)
                T_recon += np.einsum('il,lk,ki->ilk', u_c, v_c, w_c)
                
            if np.allclose(T_target, T_recon, atol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                print("Verification failed for constructed solution")
                
    return best_result
