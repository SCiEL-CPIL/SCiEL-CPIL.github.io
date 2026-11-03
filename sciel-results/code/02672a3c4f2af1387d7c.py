import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # Decomposition: sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1
    
    # Try ranks from a lower bound up to 80
    # Lower bound for M(4,4,5) is likely around 48-52
    # Let's try from 48 to 80
    
    best_rank = 80
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute indices for faster access if needed, but NumPy is fine
    # T is just ones
    T = np.ones((n, m, p), dtype=np.float64)
    
    start_time = time.time()
    time_limit = 110  # 10s margin
    
    for r in range(48, 81):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a decomposition of rank r
        # Use random restarts
        found = False
        for restart in range(100):  # 100 restarts per rank
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W with random values in {0, 1, -1}
            U = np.random.choice([0, 1, -1], size=(n, m, r)).astype(np.float64)
            V = np.random.choice([0, 1, -1], size=(m, p, r)).astype(np.float64)
            W = np.random.choice([0, 1, -1], size=(p, n, r)).astype(np.float64)
            
            # Compute residual
            # sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # We can compute this efficiently using tensor contractions
            # Residual[i, j, k] = 1 - sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            
            # Let's compute the product tensor P
            # P[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            
            # We can loop over c and accumulate
            P = np.zeros((n, m, p), dtype=np.float64)
            for c in range(r):
                # U[:, :, c] is (n, m)
                # V[:, :, c] is (m, p)
                # W[:, :, c] is (p, n)
                # Term = U[:, :, c] * V[:, :, c] * W[:, :, c]
                # But indices must match: U[i, j], V[j, k], W[k, i]
                # So we need to broadcast correctly.
                # U[i, j, c] * V[j, k, c] * W[k, i, c]
                # This is a sum over c.
                # We can do this with einsum
                # np.einsum('ijc,jkc,kic->ijk', U, V, W)
                pass
            
            # Using einsum is cleaner
            P = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            R = T - P
            
            if np.allclose(R, 0, atol=1e-9):
                found = True
                break
            
            # Local search
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random element in U, V, or W
                if np.random.random() < 0.33:
                    idx = (np.random.randint(n), np.random.randint(m), np.random.randint(r))
                    old_val = U[idx]
                    new_val = np.random.choice([0, 1, -1])
                    if old_val == new_val:
                        continue
                    U[idx] = new_val
                    P_new = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    R_new = T - P_new
                    if np.linalg.norm(R_new) < np.linalg.norm(R) - 1e-12:
                        R = R_new
                        P = P_new
                    else:
                        U[idx] = old_val
                elif np.random.random() < 0.5:
                    idx = (np.random.randint(m), np.random.randint(p), np.random.randint(r))
                    old_val = V[idx]
                    new_val = np.random.choice([0, 1, -1])
                    if old_val == new_val:
                        continue
                    V[idx] = new_val
                    P_new = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    R_new = T - P_new
                    if np.linalg.norm(R_new) < np.linalg.norm(R) - 1e-12:
                        R = R_new
                        P = P_new
                    else:
                        V[idx] = old_val
                else:
                    idx = (np.random.randint(p), np.random.randint(n), np.random.randint(r))
                    old_val = W[idx]
                    new_val = np.random.choice([0, 1, -1])
                    if old_val == new_val:
                        continue
                    W[idx] = new_val
                    P_new = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    R_new = T - P_new
                    if np.linalg.norm(R_new) < np.linalg.norm(R) - 1e-12:
                        R = R_new
                        P = P_new
                    else:
                        W[idx] = old_val
                        
                if np.allclose(R, 0, atol=1e-9):
                    found = True
                    break
            
            if found:
                break
        
        if found:
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break # Found minimal rank in this range, stop searching higher ranks
    
    if best_U is None:
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
    else:
        # Convert to list of lists as required
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        
        # Reshape to match required interface shapes
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        U_flat = np.array(U_list).reshape(n * m, r)
        V_flat = np.array(V_list).reshape(m * p, r)
        W_flat = np.array(W_list).reshape(p * n, r)
        
        return {
            "U": U_flat.tolist(),
            "V": V_flat.tolist(),
            "W": W_flat.tolist()
        }

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
