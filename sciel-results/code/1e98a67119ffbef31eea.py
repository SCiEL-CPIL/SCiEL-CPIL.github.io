"""Matrix multiplication 2x5x5 rank-minimal decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]
DEADLINE = 110.0  # safety margin before 120s kill

def _compute_residual(U, V, W):
    """Compute the residual vector for the tensor equation."""
    n, m, p = DIMENSIONS
    # Residual for each (i, j, k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
    # We only check the "on-diagonal" entries that should be 1.
    # The "off-diagonal" entries should be 0, but checking them is expensive.
    # For optimization, we focus on making the on-diagonal entries 1.
    # A valid solution must also have off-diagonal entries 0.
    
    n_constraints = n * m * p
    res = np.zeros(n_constraints)
    idx = 0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                res[idx] = val - 1.0
                idx += 1
    return res

def _verify_full(U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check on-diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries (should be 0)
    # This is the expensive part. We check all (a, b, c) that are not valid triples.
    # Valid triple: a=i*m+j, b=j*p+k, c=k*n+i
    # Let's iterate over all a, b, c
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                # j mismatch, so c must be anything? No, the structure is fixed.
                # If j_a != j_b, then for any c, the term is 0 in the target tensor.
                # So we must check that sum_r U[a,r] V[b,r] W[c,r] == 0 for all c.
                for c in range(p * n):
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
            else:
                # j matches. Now check c.
                # Valid c is k_b * n + i_a.
                # For other c, it must be 0.
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if k_c == k_b and i_c == i_a:
                        continue # Valid entry, already checked
                    else:
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > tol:
                            return False
    return True

def _check_half_integer(vals, tol=1e-8):
    """Check if all values are integer or half-integer."""
    for v in vals:
        if abs(v * 2 - round(v * 2)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook solution as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def check_time():
        return time.time() < DEADLINE
    
    # Try to find lower rank decompositions
    # We will try rank 49, 48, ... down to 40 (or time limit)
    # Using ALS (Alternating Least Squares)
    
    def als_optimize(R, seed, max_iters=50):
        """
        Alternating Least Squares to find U, V, W of rank R.
        """
        if not check_time():
            return None
            
        np.random.seed(seed)
        
        # Initialization
        # Random Gaussian
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Normalize to avoid scaling issues
        # U = U / np.linalg.norm(U, axis=1, keepdims=True)
        # V = V / np.linalg.norm(V, axis=1, keepdims=True)
        # W = W / np.linalg.norm(W, axis=1, keepdims=True)
        
        best_local_err = np.inf
        best_local_sol = None
        
        for it in range(max_iters):
            if not check_time():
                break
                
            # Solve for U
            # For each (i, j), we want U[i*m+j, :] such that
            # sum_k (U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) = 1
            # This is a linear system for U[i*m+j, :]
            # Let u = U[i*m+j, :]. Then sum_k u . (V[j*p+k, :] * W[k*n+i, :]) = 1
            # This is a system of p equations for R unknowns (u).
            # A_u * u = 1
            # A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            # We can solve this for all i, j simultaneously by block solving or just looping.
            # Since n*m is small (10), we can loop.
            
            for i in range(n):
                for j in range(m):
                    A_u = np.zeros((p, R))
                    for k in range(p):
                        A_u[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    # Solve A_u * u = 1
                    try:
                        # Use lstsq for robustness
                        u_sol, _, _, _ = np.linalg.lstsq(A_u, np.ones(p), rcond=None)
                        U[i * m + j, :] = u_sol
                    except:
                        pass
            
            # Solve for V
            # For each (j, k), we want V[j*p+k, :] such that
            # sum_i (U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) = 1
            # Let v = V[j*p+k, :]. Then sum_i v . (U[i*m+j, :] * W[k*n+i, :]) = 1
            # A_v * v = 1
            # A_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
            
            for j in range(m):
                for k in range(p):
                    A_v = np.zeros((n, R))
                    for i in range(n):
                        A_v[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A_v, np.ones(n), rcond=None)
                        V[j * p + k, :] = v_sol
                    except:
                        pass
            
            # Solve for W
            # For each (k, i), we want W[k*n+i, :] such that
            # sum_j (U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) = 1
            # Let w = W[k*n+i, :]. Then sum_j w . (U[i*m+j, :] * V[j*p+k, :]) = 1
            # A_w * w = 1
            # A_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            for k in range(p):
                for i in range(n):
                    A_w = np.zeros((m, R))
                    for j in range(m):
                        A_w[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                        W[k * n + i, :] = w_sol
                    except:
                        pass
            
            # Check convergence
            # Compute residual
            res = _compute_residual(U, V, W)
            err = np.max(np.abs(res))
            
            if err < best_local_err:
                best_local_err = err
                best_local_sol = (U.copy(), V.copy(), W.copy())
                
            if err < 1e-9:
                # Check if it's a full solution (off-diagonals are 0)
                if _verify_full(U, V, W):
                    # Check half-integer constraint?
                    # The problem asks for integer/half-integer entries.
                    # ALS might find real solutions.
                    # We might need to snap to half-integers or find a different solution.
                    # For now, let's see if we can find any solution.
                    # If we find a real solution, we might try to snap it.
                    # But snapping might break the equations.
                    # Let's assume for now that we just need to minimize rank.
                    # The prompt says "integer/half-integer entries".
                    # So we MUST return such a solution.
                    # This makes it harder.
                    # Let's check if the current solution is close to half-integers.
                    all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    if _check_half_integer(all_vals):
                        return U, V, W
                    else:
                        # It's a valid real solution, but not half-integer.
                        # We can try to snap it.
                        U_snapped = np.round(U * 2) / 2
                        V_snapped = np.round(V * 2) / 2
                        W_snapped = np.round(W * 2) / 2
                        if _verify_full(U_snapped, V_snapped, W_snapped):
                            return U_snapped, V_snapped, W_snapped
                        # If snapping fails, we keep looking or return None
                        # Returning None means we didn't find a valid half-integer solution
                        return None

        if best_local_sol is not None:
            # Try polishing the best local solution
            U0, V0, W0 = best_local_sol
            # ... (polishing logic if needed)
            pass
            
        return None

    # Try ranks from 49 down
    # Given the difficulty, we might only try a few.
    # Rank 49 is the most likely to be found if it exists.
    
    # Note: The schoolbook rank is 50.
    # If we find rank 49, that's an improvement.
    
    # Let's try rank 49 with a few seeds.
    # Time limit is tight.
    
    for R in range(49, 39, -1):
        if not check_time():
            break
            
        # Try a few seeds for this rank
        seeds_to_try = 10 if R == 49 else 5
        
        for seed in range(seeds_to_try):
            if not check_time():
                break
                
            sol = als_optimize(R, seed, max_iters=20)
            if sol is not None:
                U_found, V_found, W_found = sol
                best_rank = R
                best_U = U_found
                best_V = V_found
                best_W = W_found
                # Found a lower rank! We can stop or try even lower.
                # Let's try to go lower.
                break
        
        # If we found rank R, we can try R-1
        # But we might run out of time.
        
    # Convert best solution to lists
    # The interface expects lists of lists.
    # And the values should be integer/half-integer.
    
    # If we didn't find a better solution, return schoolbook.
    # Schoolbook has integer entries (0 and 1), so it's valid.
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
