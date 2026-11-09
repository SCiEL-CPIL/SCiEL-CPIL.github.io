"""Combined program with linear system search for lower rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    if time.time() - start_time > 5.0:
        return {"U": U_base, "V": V_base, "W": W_base}

    best_solution = None
    best_rank = rank_baseline
    deadline = start_time + time_limit - 5.0
    
    # Search for lower rank
    # We iterate R from 59 down to 40
    for R in range(rank_baseline - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # Start with more trials for higher ranks, decrease for lower
        trials = 1000 if R > 55 else 500
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Random U, V with entries from {-1, 0, 1}
            # Use a seed for reproducibility in debugging, but random for search
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            W_candidate = np.zeros((p * n, R), dtype=float)
            valid = True
            
            # Optimization: Pre-compute M[c, i, k] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # M has shape (R, n, p)
            M = np.zeros((R, n, p), dtype=int)
            for i in range(n):
                for k in range(p):
                    for c in range(R):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j, c] * V[j * p + k, c]
                        M[c, i, k] = s
            
            for i in range(n):
                for k in range(p):
                    # We need sum_c W[k*n+i, c] * M[c, i, k] = 1
                    # M_vec is M[:, i, k]
                    M_vec = M[:, i, k]
                    
                    # First, try sparse solution (single non-zero in W row)
                    # Look for c where M_vec[c] is in {1, -1, 2, -2}
                    found = False
                    for c in range(R):
                        v = M_vec[c]
                        if v == 1:
                            W_candidate[k * n + i, c] = 1.0
                            found = True
                            break
                        elif v == -1:
                            W_candidate[k * n + i, c] = -1.0
                            found = True
                            break
                        elif v == 2:
                            W_candidate[k * n + i, c] = 0.5
                            found = True
                            break
                        elif v == -2:
                            W_candidate[k * n + i, c] = -0.5
                            found = True
                            break
                    
                    if not found:
                        # Try to solve a small linear system for this (i,k)
                        # A = M_vec (shape R,)
                        # We want w (shape R,) such that A . w = 1
                        # And w entries should be half-integers (ideally sparse)
                        
                        # Check if 1 is in the span of A. Since A is 1D, this just means A has non-zero elements.
                        # If A is all zeros, impossible.
                        if np.all(M_vec == 0):
                            valid = False
                            break
                            
                        # Use least squares to find a solution
                        # Since A is 1D, the "matrix" is just the vector.
                        # lstsq on a vector isn't standard in the same way, but we can treat it as a system.
                        # Actually, for a single equation sum(A_c * w_c) = 1, there are infinitely many solutions.
                        # We want a "nice" one (half-integers).
                        
                        # Heuristic: Try to find a subset of indices that sums to 1 (with signs).
                        # This is a subset sum problem variant.
                        # Given small R (<= 60) and small integer values, we can try a few random combinations?
                        # Or just use the least squares solution and round?
                        
                        w_ls, residuals, rank, sv = np.linalg.lstsq(M_vec.reshape(1, -1), np.ones(1), rcond=None)
                        w_ls = w_ls.flatten()
                        
                        # Check if w_ls is close to half-integers
                        w_candidate = np.round(w_ls * 2) / 2.0
                        
                        # Verify A * w_candidate == 1
                        if np.allclose(M_vec.dot(w_candidate), 1.0, atol=1e-9):
                            W_candidate[k * n + i, :] = w_candidate
                        else:
                            # Try a few random perturbations or different rounding strategies?
                            # For now, mark invalid to keep speed.
                            valid = False
                            break
                            
                if not valid:
                    break
            
            if valid:
                # Verify the full tensor equation
                # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
                valid_full = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(R):
                                s += U[i * m + j, c] * V[j * p + k, c] * W_candidate[k * n + i, c]
                            if abs(s - 1) > 1e-9:
                                valid_full = False
                                break
                        if not valid_full:
                            break
                    if not valid_full:
                        break
                
                if valid_full:
                    # Convert to lists
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_candidate.tolist()
                    }
                    best_rank = R
                    # If we find a rank significantly lower, we can break early
                    # but we want the minimum, so we continue to lower R
                    # However, if R is very low, we might break to save time for verification?
                    # No, we just continue the outer loop to try R-1.
                    # But if we found R=50, we should check R=49 etc.
                    # The loop structure handles this.
                    # We can break the inner trial loop to try next R?
                    # No, we might find a better solution for the same R?
                    # Rank is the score, so any solution for R is equal to any other for R.
                    # So we can break the trial loop for this R.
                    break

    if best_solution is not None:
        return best_solution
    else:
        return {"U": U_base, "V": V_base, "W": W_base}
