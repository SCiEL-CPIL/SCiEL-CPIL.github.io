import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: 2x3x9
    # Strategy: Find min rank for 2x3x3, then replicate 3 times.
    
    # Helper to solve for n x m x p
    def solve_decomposition(n, m, p, max_R, time_limit=100):
        start_time = time.time()
        
        # Generate indices
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        indices_nonzero = []
        indices_zero = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    indices_nonzero.append((a, b, c))
                    
        # All possible indices
        all_indices = []
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    all_indices.append((a, b, c))
                    
        # We only need to check zeros that are not in nonzero
        # But for optimization, it's safer to include some zeros or all?
        # Including all zeros (306 for 2x3x3) is fine.
        
        # Try ranks from lower bound to max_R
        # Lower bound for 2x3x3?
        # Try 15, 16, 17, 18
        for R in range(15, max_R + 1):
            if time.time() - start_time > time_limit:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            def residuals(x):
                U = x[:size_U].reshape(n * m, R)
                V = x[size_U:size_U+size_V].reshape(m * p, R)
                W = x[size_U+size_V:].reshape(p * n, R)
                
                res = np.zeros(len(all_indices))
                # Non-zero
                for idx, (a, b, c) in enumerate(indices_nonzero):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    res[idx] = val - 1.0
                    
                # Zero
                # We can't easily skip nonzero in all_indices loop without a set
                # Let's just compute all and subtract the target
                # Target is 1 if in nonzero, 0 otherwise
                # We can create a target vector
                # But this is slower.
                # Let's just add zero constraints for a subset of zeros to keep it fast
                # Or just check nonzero in optimization and verify zeros at end.
                # If we only check nonzero, the solution might not have zeros.
                # But if we verify at the end, we can reject it.
                # To improve success rate, let's include a random subset of zeros in the residual.
                return res

            # Better: Use a function that computes the full tensor contraction and compares to target
            # But that's expensive.
            
            # Let's stick to checking nonzero in optimization, and verify zeros later.
            # This is what the original program did.
            
            np.random.seed(42)
            success = False
            best_sol = None
            best_cost = np.inf
            
            # Try a few random restarts
            for restart in range(5):
                if time.time() - start_time > time_limit:
                    break
                    
                x0 = np.random.rand(total_size) * 0.1
                try:
                    sol = opt.least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                    if sol.cost < 1e-10 and sol.cost < best_cost:
                        best_cost = sol.cost
                        best_sol = sol.x
                        
                        # Verify full tensor
                        U = best_sol[:size_U].reshape(n * m, R)
                        V = best_sol[size_U:size_U+size_V].reshape(m * p, R)
                        W = best_sol[size_U+size_V:].reshape(p * n, R)
                        
                        is_valid = True
                        for a, b, c in all_indices:
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            i, j = divmod(a, m)
                            j2, k = divmod(b, p)
                            k2, i2 = divmod(c, n)
                            if j == j2 and i == i2:
                                if abs(val - 1.0) > 1e-6:
                                    is_valid = False
                                    break
                            else:
                                if abs(val) > 1e-6:
                                    is_valid = False
                                    break
                        if is_valid:
                            # Round to 0.5
                            U = np.round(U * 2) / 2
                            V = np.round(V * 2) / 2
                            W = np.round(W * 2) / 2
                            
                            # Verify rounded
                            valid_rounded = True
                            for a, b, c in all_indices:
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                i, j = divmod(a, m)
                                j2, k = divmod(b, p)
                                k2, i2 = divmod(c, n)
                                if j == j2 and i == i2:
                                    if abs(val - 1.0) > 1e-9:
                                        valid_rounded = False
                                        break
                                else:
                                    if abs(val) > 1e-9:
                                        valid_rounded = False
                                        break
                            if valid_rounded:
                                return U, V, W, R
                                # Note: We return the 2x3x3 solution
                except Exception:
                    pass
                    
            if best_sol is not None:
                # If we found a low cost but failed verification, maybe try next R
                pass
                
        return None

    # Solve for 2x3x3
    sol_233 = solve_decomposition(2, 3, 3, max_R=18, time_limit=90)
    
    if sol_233:
        U_233, V_233, W_233, R_233 = sol_233
        R_total = 3 * R_233
        
        # Construct 2x3x9
        n, m, p = 2, 3, 9
        U = np.zeros((n * m, R_total))
        V = np.zeros((m * p, R_total))
        W = np.zeros((p * n, R_total))
        
        for ell in range(3):
            # U is same for all blocks
            U[:, ell * R_233 : (ell+1) * R_233] = U_233
            
            # V: j*p + k. k = 3*ell + k'
            for j in range(m):
                for k_prime in range(3):
                    k = 3 * ell + k_prime
                    b_idx = j * p + k
                    b_idx_233 = j * 3 + k_prime
                    V[b_idx, ell * R_233 : (ell+1) * R_233] = V_233[b_idx_233]
                    
            # W: k*n + i. k = 3*ell + k'
            for k_prime in range(3):
                k = 3 * ell + k_prime
                for i in range(n):
                    c_idx = k * n + i
                    c_idx_233 = k_prime * 2 + i
                    W[c_idx, ell * R_233 : (ell+1) * R_233] = W_233[c_idx_233]
                    
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback: Schoolbook 54
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
