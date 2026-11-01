import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb

    def verify(U, V, W, R, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(mat, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    def try_discrete_structured(R):
        """Try structured discrete decompositions with integer/half-integer entries."""
        # Try Strassen-like block structure
        # Split A (2x3) into A1 (2x2) and A2 (2x1)
        # Split B (3x7) into B1 (2x7) and B2 (1x7)
        # C = A1*B1 + A2*B2
        
        # For A1*B1: use Strassen on 2x2 * 2x7 by splitting B1 into 2x2 blocks
        # B1 = [B11(2x2) B12(2x2) B13(2x2) B14(2x1)]
        # Apply Strassen to each 2x2 block separately, but share factors where possible
        
        # Alternative: use a direct rank-16 construction
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # For 2x3x7, we can try to find a decomposition with rank < 42
        
        # Try a block Strassen approach:
        # Group columns of B: B = [B_0 | B_1 | B_2 | B_3 | B_4 | B_5 | B_6]
        # where each B_l is 3x1
        # C[:, l] = A * B_l
        
        # For each column l, A*B_l is a 2x1 vector
        # We can use the fact that A is 2x3 to find efficient decompositions
        
        # Let's try a specific structured decomposition
        # Use the identity: for 2x3 * 3x1, rank is at most 6 (schoolbook)
        # But we can do better by sharing structure across columns
        
        # Try: express each C[i,k] as sum over r of U[i,k,r]*V[j,k,r]*W[k,i,r]
        # with small integer/half-integer coefficients
        
        # Construct a candidate using block Strassen on 2x2 submatrices
        # A = [A11 A12 A13; A21 A22 A23]
        # B = [B11 ... B17; B21 ... B27; B31 ... B37]
        
        # Use Strassen-like formulas for 2x2 blocks
        # M1 = (A11+A22)*(B11+B21)  -- but B is 3x7, so this doesn't directly apply
        
        # Instead, let's use a different approach:
        # For 2x3 * 3x7, we can view it as 2x(3*7) * (3*7)x7 after reshaping
        # Or use the fact that rank(2x3x7) <= rank(2x3x3) + rank(2x3x4) etc.
        
        # Let's try to construct a decomposition with rank 24 or so
        # using a mixture of Strassen and schoolbook on blocks
        
        # Block decomposition:
        # A = [A_left (2x2) | A_right (2x1)]
        # B = [B_top (2x7); B_bottom (1x7)]
        
        # C = A_left * B_top + A_right * B_bottom
        
        # For A_left (2x2) * B_top (2x7):
        # B_top = [B_t1 (2x2) B_t2 (2x2) B_t3 (2x2) B_t4 (2x1)]
        # Apply Strassen to each 2x2 block: 7 multiplications each, but we can share
        # Actually, for 2x2 * 2xk, we can use a modified Strassen with 7k/2 multiplications
        
        # Simpler: use schoolbook for small blocks
        # A_left * B_top: 2*2*7 = 28 rank-1 terms
        # A_right * B_bottom: 2*1*7 = 14 rank-1 terms
        # Total: 42 (schoolbook)
        
        # To improve, use Strassen for the 2x2 * 2x7 part:
        # For 2x2 * 2x2, Strassen uses 7 multiplications
        # For 2x2 * 2x7, we can split the 2x7 into four 2x2 blocks (last is 2x1)
        # and apply Strassen to each, but this gives 7*3 + 7*1 = 28 for the 2x2*2x7 part
        # Same as schoolbook.
        
        # Better approach: use the fact that for 2x2 * 2xk, the rank is at most 7k/2
        # For k=7, that's 24.5, so at most 25.
        
        # Let's try to construct a rank-25 decomposition for 2x2 * 2x7
        # Then add 14 for the 2x1 * 1x7 part, total 39.
        
        # Actually, let's just try the continuous optimization with better initialization
        
        return None

    def continuous_optimize(R, attempt, deadline):
        """Optimize a rank-R decomposition using least_squares."""
        rng = np.random.default_rng(attempt * 7919 + R * 104729)
        
        # Initialize with structured random values
        vals = [-1, -0.5, 0, 0.5, 1]
        U = rng.choice(vals, size=(n * m, R)).astype(float)
        V = rng.choice(vals, size=(m * p, R)).astype(float)
        W = rng.choice(vals, size=(p * n, R)).astype(float)
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            u_size = n * m * R
            v_size = m * p * R
            U = x[:u_size].reshape(n * m, R)
            V = x[u_size:u_size+v_size].reshape(m * p, R)
            W = x[u_size+v_size:].reshape(p * n, R)
            return U, V, W
        
        def residuals(x):
            U, V, W = unflatten(x)
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        res[idx] = s - 1.0
                        idx += 1
            return res
        
        x0 = flatten(U, V, W)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=3000, ftol=1e-14, xtol=1e-14)
        except Exception:
            return None
        
        if sol.cost < 1e-12:
            U_sol, V_sol, W_sol = unflatten(sol.x)
            
            # Try to snap to half-integers
            U_snapped = snap_to_half_integer(U_sol, tol=1e-5)
            V_snapped = snap_to_half_integer(V_sol, tol=1e-5)
            W_snapped = snap_to_half_integer(W_sol, tol=1e-5)
            
            if verify(U_snapped, V_snapped, W_snapped, R):
                return U_snapped, V_snapped, W_snapped
            
            # If snapping doesn't work, check if raw solution is close to half-integers
            all_close = True
            for mat in [U_sol, V_sol, W_sol]:
                snapped = snap_to_half_integer(mat, tol=1e-4)
                if np.max(np.abs(mat - snapped)) > 1e-4:
                    all_close = False
                    break
            
            if all_close:
                return U_snapped, V_snapped, W_snapped
        
        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for matrix multiplication tensor rank:
    # For n x m x p, rank >= max(n*m, m*p, n*p) is not strict
    # But rank >= n*p = 14 is a lower bound from the output dimension
    # Actually, a better lower bound is from the slice rank: rank >= max(n*m, m*p, n*p)
    # For 2x3x7: max(6, 21, 14) = 21
    
    min_rank_to_try = 21
    max_attempts_per_rank = 6
    
    for R in range(min_rank_to_try, min(R_sb + 1, 42)):
        if time.time() > deadline - 20:
            break
        
        # First try discrete structured search
        result = try_discrete_structured(R)
        if result is not None:
            U_sol, V_sol, W_sol = result
            if verify(U_sol, V_sol, W_sol, R):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        
        # Then try continuous optimization
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 20:
                break
            
            result = continuous_optimize(R, attempt, deadline)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if verify(U_sol, V_sol, W_sol, R):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
