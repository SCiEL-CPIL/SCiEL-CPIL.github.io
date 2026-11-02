import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118  # Safety margin before 120s kill

    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    def verify_solution(U_list, V_list, W_list, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U_list[0]) if U_list else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U_list[i * m + j][t] * V_list[j * p + k][t] * W_list[k * n + i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_solve_for_W(U_arr, V_arr, R):
        """
        Given U (n*m x R) and V (m*p x R), solve the linear system for W (p*n x R).
        Returns W as a list if a valid solution exists, else None.
        """
        # Build the linear system: A @ w = b
        # Each equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        num_eqs = n * m * p
        num_unk = p * n * R
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * (m * p) + j * p + k
                    for r in range(R):
                        coeff = U_arr[i * m + j, r] * V_arr[j * p + k, r]
                        if coeff != 0:
                            unk_idx = (k * n + i) * R + r
                            A[eq_idx, unk_idx] = coeff

        # Solve using least squares
        try:
            w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None

        # Check residual
        if len(residuals) > 0:
            if residuals[0] > 1e-8:
                return None
        else:
            # Underdetermined or square: check directly
            if not np.allclose(A @ w_sol, b, atol=1e-8):
                return None

        # Check if solution is close to integer or half-integer values
        w_rounded = np.round(w_sol * 2) / 2
        if not np.allclose(w_sol, w_rounded, atol=1e-7):
            # Try rounding to integers
            w_rounded_int = np.round(w_sol)
            if np.allclose(w_sol, w_rounded_int, atol=1e-7):
                w_rounded = w_rounded_int
            else:
                return None

        # Construct W list
        W_list = [[0] * R for _ in range(p * n)]
        for idx in range(len(w_sol)):
            k_val = idx // (n * R)
            rem = idx % (n * R)
            i_val = rem // R
            r_val = rem % R
            W_list[k_val * n + i_val][r_val] = float(w_rounded[idx])

        return W_list

    def generate_sparse_matrix(rows, cols, density, values=(-1, 0, 1)):
        """Generate a sparse random matrix with entries from values."""
        mat = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.random.random() < density:
                    mat[i, j] = np.random.choice(values)
        return mat

    # Strategy 1: Try ranks from 35 (lower bound) upward
    # For each rank, try random sparse U and V, solve for W
    lower_bound = max(n * m, m * p, p * n)  # 35

    # We'll search from lower_bound to best_rank - 1
    # But we need to be efficient. Let's try a few ranks with multiple trials.
    
    # Pre-compute time budget per rank
    # We have about 110 seconds of budget
    # Let's allocate more time to lower ranks
    
    # Try ranks in increasing order
    ranks_to_try = list(range(lower_bound, min(best_rank, 105)))
    
    # Shuffle to try different ranks, but prioritize lower ones
    # Actually, let's just go in order since lower rank is better
    
    for R in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Number of trials based on rank
        # Lower ranks get more trials
        if R <= 40:
            trials = 50
        elif R <= 50:
            trials = 30
        elif R <= 60:
            trials = 20
        else:
            trials = 10
        
        for t in range(trials):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse U and V
            # Density: higher density for lower ranks (need more structure)
            density = 0.3 if R <= 40 else 0.4
            
            U_arr = generate_sparse_matrix(n * m, R, density)
            V_arr = generate_sparse_matrix(m * p, R, density)
            
            # Solve for W
            W_list = try_solve_for_W(U_arr, V_arr, R)
            
            if W_list is not None:
                # Verify the solution
                U_list = U_arr.tolist()
                V_list = V_arr.tolist()
                
                if verify_solution(U_list, V_list, W_list):
                    # Found a valid solution!
                    best_rank = R
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    # Continue searching for even lower rank
                    # But break out of trials for this R
                    break
        
        # If we found a solution at this rank, continue to lower ranks
        # (we're going upward, so we'd stop here if we found the minimum)
        # But we want the MINIMUM, so we should stop once we find a valid solution
        # at the lowest possible rank
        if best_rank == R:
            break

    # Strategy 2: If no solution found above, try a different approach
    # Use random real-valued U and V, solve for W, then check if W is close to rational
    if best_rank == rank_sb:
        for R in range(lower_bound, min(best_rank, 60)):
            if time.time() > deadline - 5:
                break
            
            trials = 20
            for t in range(trials):
                if time.time() > deadline - 5:
                    break
                
                # Random real-valued matrices
                U_arr = np.random.randn(n * m, R)
                V_arr = np.random.randn(m * p, R)
                
                # Solve for W
                num_eqs = n * m * p
                num_unk = p * n * R
                A = np.zeros((num_eqs, num_unk))
                b = np.ones(num_eqs)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * (m * p) + j * p + k
                            for r in range(R):
                                coeff = U_arr[i * m + j, r] * V_arr[j * p + k, r]
                                A[eq_idx, (k * n + i) * R + r] = coeff
                
                try:
                    w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    continue
                
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    if not np.allclose(A @ w_sol, b, atol=1e-8):
                        continue
                
                # Check if W is close to rational numbers with small denominators
                # Try denominators 1, 2, 3, 4, 5, 6, 7, 8
                found_rational = False
                for denom in range(1, 9):
                    w_scaled = w_sol * denom
                    w_rounded = np.round(w_scaled)
                    if np.allclose(w_scaled, w_rounded, atol=1e-6):
                        # W entries are multiples of 1/denom
                        W_list = [[0] * R for _ in range(p * n)]
                        for idx in range(len(w_sol)):
                            k_val = idx // (n * R)
                            rem = idx % (n * R)
                            i_val = rem // R
                            r_val = rem % R
                            W_list[k_val * n + i_val][r_val] = float(w_rounded[idx]) / denom
                        
                        U_list = U_arr.tolist()
                        V_list = V_arr.tolist()
                        
                        if verify_solution(U_list, V_list, W_list):
                            best_rank = R
                            best_solution = {"U": U_list, "V": V_list, "W": W_list}
                            found_rational = True
                            break
                
                if found_rational:
                    break
        
        if best_rank != rank_sb:
            break

    return best_solution
# EVOLVE-BLOCK-END
