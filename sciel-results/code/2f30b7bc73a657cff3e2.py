import numpy as np
import time
import random

def _verify_decomp(U, V, W, n, m, p, trials=10, seed=42):
    """
    Verify the bilinear identity:
    C[i, k] = sum_j A[i, j] B[j, k]
    Reconstructed:
    C[i, k] = sum_{j, r} U[i, j, r] V[j, k, r] W[k, i, r] A[i, j] B[j, k]
    Interface indices:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    Condition:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k
    """
    U_t = np.array(U).reshape(n, m, -1)
    V_t = np.array(V).reshape(m, p, -1)
    W_t = np.array(W).reshape(p, n, -1)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception as e:
        return False
        
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    n, m, p = 2, 2, 11
    
    # Target ranks to try, from low to high
    target_ranks = [39, 40, 41, 42, 43, 44]
    
    best_result = None
    best_rank = 44 # Fallback to schoolbook
    
    # First, ensure we have a valid schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, 44))
    V_sb = np.zeros((m * p, 44))
    W_sb = np.zeros((p * n, 44))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    if _verify_decomp(U_sb.tolist(), V_sb.tolist(), W_sb.tolist(), n, m, p):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        best_rank = 44
    else:
        # Should not happen
        return {"U": [[0]*44 for _ in range(n*m)], "V": [[0]*44 for _ in range(m*p)], "W": [[0]*44 for _ in range(p*n)]}

    # Strategy: ALS (Alternating Least Squares) with Rounding
    # We try to find a decomposition of rank R.
    
    for R in target_ranks:
        if R >= best_rank:
            break
            
        # Try multiple random initializations
        found = False
        max_attempts = 200 # Adjust based on time
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers/half-integers
            # Using integers for simplicity, then rounding
            # Scale: values should be small, e.g., -2 to 2
            np.random.seed(attempt)
            U = np.random.randint(-2, 3, (n * m, R)).astype(float)
            V = np.random.randint(-2, 3, (m * p, R)).astype(float)
            W = np.random.randint(-2, 3, (p * n, R)).astype(float)
            
            # ALS iterations
            # We want to minimize sum_{i,j,k} (sum_r U_ijr V_jkr W_kir - 1)^2
            # This is non-linear. 
            # Linearize by fixing two matrices.
            # Fix V, W -> solve for U.
            # The term is sum_r U_ijr (V_jkr W_kir).
            # Let M_ij = V_jkr W_kir? No, k is summed over in the error?
            # The error is E = sum_{i,j,k} (S_ijk - 1)^2 where S_ijk = sum_r U_ijr V_jkr W_kir.
            # This is not linear in U because S_ijk depends on U, and we sum over k.
            # Wait, S_ijk is linear in U.
            # E = sum_{i,j,k} (sum_r U_ijr V_jkr W_kir - 1)^2
            # This is a quadratic form in U.
            # We can solve this using normal equations.
            
            # However, solving the full system for U (4*R variables) for each step might be slow if done naively.
            # Let's use a simpler gradient step or just iterate.
            
            # Actually, we can use the fact that the system is small.
            # Variables: U (4*R), V (22*R), W (22*R).
            # Equations: 44.
            
            # Let's try a different approach: 
            # Since the target is small integers, we can use a stochastic local search.
            # Start with random integers.
            # Calculate error.
            # Pick a random element in U, V, or W.
            # Change it by +/- 0.5.
            # If error decreases, keep it.
            # If error is 0, we found a solution.
            
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            def calc_error(U, V, W):
                U_t = U.reshape(n, m, -1)
                V_t = V.reshape(m, p, -1)
                W_t = W.reshape(p, n, -1)
                S = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                return np.sum((S - 1.0)**2)
            
            err = calc_error(U_curr, V_curr, W_curr)
            
            # Local search
            steps = 500
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and index
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U_curr
                    dims = (n * m, R)
                elif mat_idx == 1:
                    mat = V_curr
                    dims = (m * p, R)
                else:
                    mat = W_curr
                    dims = (p * n, R)
                
                row = np.random.randint(dims[0])
                col = np.random.randint(dims[1])
                
                old_val = mat[row, col]
                # Try +/- 0.5
                delta = 0.5 * (1 if np.random.rand() > 0.5 else -1)
                new_val = old_val + delta
                mat[row, col] = new_val
                
                new_err = calc_error(U_curr, V_curr, W_curr)
                
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        # Found a solution!
                        # Round to nearest 0.5 to ensure exactness if floating point noise exists
                        U_final = np.round(U_curr * 2) / 2
                        V_final = np.round(V_curr * 2) / 2
                        W_final = np.round(W_curr * 2) / 2
                        
                        if _verify_decomp(U_final.tolist(), V_final.tolist(), W_final.tolist(), n, m, p):
                            best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                            best_rank = R
                            found = True
                            break
                else:
                    mat[row, col] = old_val
                    
            if found:
                break

    if best_result is not None:
        return best_result
    else:
        return {"U": [[0]*best_rank for _ in range(n*m)], "V": [[0]*best_rank for _ in range(m*p)], "W": [[0]*best_rank for _ in range(p*n)]}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
