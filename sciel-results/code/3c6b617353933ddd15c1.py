"""Matrix multiplication 3x4x7 decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    # Start with schoolbook decomposition
    rank_school = n * m * p  # 84
    best_U = [[0.0] * rank_school for _ in range(n * m)]
    best_V = [[0.0] * rank_school for _ in range(m * p)]
    best_W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1.0
                best_V[j * p + k][col] = 1.0
                best_W[k * n + i][col] = 1.0
                col += 1
    best_rank = rank_school

    def verify(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition satisfies the tensor equation."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank(rank):
        """Attempt to find a decomposition of the given rank."""
        if time.time() > deadline - 10:
            return None
        
        # We'll use a structured approach with random initialization and iterative refinement
        # The key insight: for each term r, U[i*m+j,r], V[j*p+k,r], W[k*n+i,r] are free variables
        # subject to the constraint sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
        
        # Strategy: Use random linear combinations of schoolbook terms
        # Each term r is a linear combination of the 84 schoolbook basis terms
        # U[i*m+j,r] = sum_{i',j',k'} alpha_r[i',j',k'] * delta_{i,i'} * delta_{j,j'}
        #              = alpha_r[i,j,*]... no, this doesn't work directly.
        
        # Better approach: parameterize each term as a "superposition" of schoolbook terms
        # Term r corresponds to a vector in R^{n*m*p} that is the sum of contributions
        # For term r, define:
        #   u_r[i,j] = sum_k a_r[i,j,k]  (contribution to U[i*m+j,r])
        #   v_r[j,k] = sum_i a_r[i,j,k]  (contribution to V[j*p+k,r])  
        #   w_r[k,i] = sum_j a_r[i,j,k]  (contribution to W[k*n+i,r])
        # Then sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] should equal 1 for all (i,j,k)
        # But this is not guaranteed by construction.
        
        # Alternative: Direct random search with Newton-like updates
        # Initialize U, V, W randomly, then iteratively correct
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
                
            random.seed(hash((rank, attempt)) % (2**31))
            np.random.seed(hash((rank, attempt)) % (2**31))
            
            U = np.random.uniform(-0.5, 0.5, (n * m, rank))
            V = np.random.uniform(-0.5, 0.5, (m * p, rank))
            W = np.random.uniform(-0.5, 0.5, (p * n, rank))
            
            # Iterative correction
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                    
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = U[i * m + j]
                            vk = V[j * p + k]
                            wi = W[k * n + i]
                            s = np.dot(ui * vk, wi)
                            err = 1.0 - s
                            if abs(err) > 1e-10:
                                max_err = max(max_err, abs(err))
                                # Find the term r that can best correct this error
                                # We want to adjust W[k*n+i, r] for some r
                                # new_w = (1 - (s - ui[r]*vk[r]*wi[r])) / (ui[r]*vk[r])
                                # Pick r with largest |ui[r]*vk[r]|
                                prod = ui * vk
                                # Avoid zero products
                                mask = np.abs(prod) > 1e-10
                                if not np.any(mask):
                                    # Try adjusting U or V instead
                                    # Adjust U[i*m+j, r]
                                    prod_w = vk * wi
                                    mask_u = np.abs(prod_w) > 1e-10
                                    if np.any(mask_u):
                                        idx = np.where(mask_u)[0]
                                        r = idx[random.randint(0, len(idx)-1)]
                                        new_u = (1.0 - (s - ui[r]*prod_w[r])) / prod_w[r]
                                        ui[r] = new_u
                                    else:
                                        # Adjust V
                                        prod_uw = ui * wi
                                        mask_v = np.abs(prod_uw) > 1e-10
                                        if np.any(mask_v):
                                            idx = np.where(mask_v)[0]
                                            r = idx[random.randint(0, len(idx)-1)]
                                            new_v = (1.0 - (s - vk[r]*prod_uw[r])) / prod_uw[r]
                                            vk[r] = new_v
                                    continue
                                idx = np.where(mask)[0]
                                # Pick r with largest product magnitude
                                r = idx[np.argmax(np.abs(prod[idx]))]
                                new_w = (1.0 - (s - ui[r]*vk[r]*wi[r])) / (ui[r]*vk[r])
                                wi[r] = new_w
                
                if max_err < 1e-8:
                    # Success! Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
        return None

    # Try ranks from low to high
    # Known lower bounds: for 3x4x7, the rank is at least 3*4 + 4*7 + 3*7 = 12+28+21 = 61
    # But better bounds exist. Let's try from a reasonable range.
    for rk in range(30, 84):
        if time.time() > deadline - 15:
            break
        if rk >= best_rank:
            break
        result = try_rank(rk)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rk
            print(f"Found rank {rk}")
            # Once we find a rank, try to improve
            break

    # Use the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
