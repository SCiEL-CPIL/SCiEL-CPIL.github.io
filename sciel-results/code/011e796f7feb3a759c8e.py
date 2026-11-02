import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook()

    def fast_error(U, V, W):
        R = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(R):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat, optimize=True)
        return np.sum((T - 1.0) ** 2)

    def verify(U, V, W, tol=1e-10):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def solve_U_given_VW(V, W, r):
        """For fixed V, W, solve linear system for U using least squares per i-block."""
        U = np.zeros((n * m, r))
        for i in range(n):
            # Build linear system for U[i*m:(i+1)*m, :]
            # Unknowns: m*r entries
            # Equations: m*p entries (for each j, k)
            A_mat = np.zeros((m * p, m * r))
            b_vec = np.ones(m * p)
            for j in range(m):
                for k in range(p):
                    eq_idx = j * p + k
                    # Coefficient for U[i*m+j, c] is V[j*p+k, c] * W[k*n+i, c]
                    A_mat[eq_idx, j*r:(j+1)*r] = V[j*p+k, :] * W[k*n+i, :]
            # Solve least squares
            sol, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U[i*m:(i+1)*m, :] = sol.reshape(m, r)
        return U

    def try_rank(r, max_attempts=200, seed_offset=0):
        """Try to find a decomposition of rank r by random V, W and solving for U."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed_offset)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random V and W with small integer/half-integer entries
            # Use values from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
            vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            V = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(m*p)], dtype=float)
            W = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(p*n)], dtype=float)
            
            # Solve for U
            U = solve_U_given_VW(V, W, r)
            
            # Round U to nearest half-integer
            U_rounded = np.round(U * 2) / 2.0
            
            # Check error with rounded U
            err = fast_error(U_rounded, V, W)
            if err < 1e-8:
                # Verify exactly
                if verify(U_rounded, V, W, tol=1e-10):
                    return U_rounded, V, W
            
            # Also try with unrounded U to see if it's close
            err_unrounded = fast_error(U, V, W)
            if err_unrounded < 1e-8:
                if verify(U, V, W, tol=1e-10):
                    return U, V, W
            
            # Try local refinement: adjust V and W entries slightly
            # and re-solve for U
            if err < 1.0:  # Only refine if we're getting close
                for refine_iter in range(20):
                    if time.time() > deadline:
                        return None
                    
                    # Randomly perturb one entry in V or W
                    mat_choice = rng.choice(2)
                    if mat_choice == 0:
                        mat = V
                        rows, cols = V.shape
                    else:
                        mat = W
                        rows, cols = W.shape
                    
                    i_idx = rng.randint(rows)
                    j_idx = rng.randint(cols)
                    old_val = mat[i_idx, j_idx]
                    
                    # Try small perturbations
                    best_err = err
                    best_val = old_val
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_val = old_val + delta
                        # Keep to half-integers
                        new_val = round(new_val * 2) / 2.0
                        if new_val == old_val:
                            continue
                        mat[i_idx, j_idx] = new_val
                        U_new = solve_U_given_VW(V, W, r)
                        U_new_rounded = np.round(U_new * 2) / 2.0
                        new_err = fast_error(U_new_rounded, V, W)
                        if new_err < best_err:
                            best_err = new_err
                            best_val = new_val
                        mat[i_idx, j_idx] = old_val
                    
                    if best_val != old_val:
                        mat[i_idx, j_idx] = best_val
                        U = solve_U_given_VW(V, W, r)
                        U_rounded = np.round(U * 2) / 2.0
                        err = fast_error(U_rounded, V, W)
                        if err < 1e-8:
                            if verify(U_rounded, V, W, tol=1e-10):
                                return U_rounded, V, W
                    else:
                        break
        
        return None

    # Start with schoolbook as fallback
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = target_rank

    # Try to find lower rank decompositions
    # Start from rank 50 and go up, trying to find the minimum
    # Known: M(3,5,7) should be significantly less than 105
    
    # First, try a range of ranks
    for r in range(50, min(target_rank, 105)):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_attempts=50, seed_offset=r * 1000)
        if result is not None:
            U_found, V_found, W_found = result
            best_U = U_found
            best_V = V_found
            best_W = W_found
            best_rank = r
            break  # Found a valid decomposition at this rank
    
    # If we found something, try to go even lower
    if best_rank < target_rank:
        for r in range(max(40, best_rank - 15), best_rank):
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_attempts=30, seed_offset=r * 2000 + 500)
            if result is not None:
                U_found, V_found, W_found = result
                best_U = U_found
                best_V = V_found
                best_W = W_found
                best_rank = r

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-9):
        # Fall back to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = target_rank

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
