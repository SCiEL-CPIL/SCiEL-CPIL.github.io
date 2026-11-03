"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 130s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    total_error += (val - 1.0) ** 2
        return total_error < 1e-9, total_error
    
    def solve_for_factor_U(U, V, W, r):
        """
        Solve for U given V, W.
        For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        Group by (i,j): p equations, r unknowns per (i,j)
        """
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                # For fixed (i,j), build A: (p, r) where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = np.zeros((p, r))
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                b = np.ones(p)
                # Solve A @ u = b
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i * m + j] = u
        return U_new
    
    def solve_for_factor_V(U, V, W, r):
        """
        Solve for V given U, W.
        For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        Group by (j,k): n equations, r unknowns per (j,k)
        """
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                # For fixed (j,k), build A: (n, r) where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                A = np.zeros((n, r))
                for i in range(n):
                    A[i] = U[i * m + j] * W[k * n + i]
                b = np.ones(n)
                # Solve A @ v = b
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j * p + k] = v
        return V_new
    
    def solve_for_factor_W(U, V, W, r):
        """
        Solve for W given U, V.
        For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        Group by (k,i): m equations, r unknowns per (k,i)
        """
        W_new = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                # For fixed (k,i), build A: (m, r) where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j] = U[i * m + j] * V[j * p + k]
                b = np.ones(m)
                # Solve A @ w = b
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k * n + i] = w
        return W_new
    
    def alternating_least_squares(r, U, V, W, max_iters=100, tol=1e-12):
        """
        Alternating least squares to find a rank-r decomposition.
        Returns (U, V, W, error) if converged, else None.
        """
        prev_error = float('inf')
        for iteration in range(max_iters):
            if time.time() > DEADLINE - 5:
                return None
            
            # Solve for U
            U_new = solve_for_factor_U(U, V, W, r)
            
            # Solve for V
            V_new = solve_for_factor_V(U_new, V, W, r)
            
            # Solve for W
            W_new = solve_for_factor_W(U_new, V_new, W, r)
            
            # Check convergence
            valid, error = verify_np(U_new, V_new, W_new, r)
            
            if valid:
                return U_new, V_new, W_new, error
            
            # Check if we're making progress
            if iteration > 0:
                if abs(prev_error - error) < tol and error > 1e-8:
                    # Stagnant and not close enough
                    return None
                if abs(prev_error - error) < tol and error < 1e-6:
                    # Close but not exact - break to try polish
                    break
            
            prev_error = error
            U, V, W = U_new, V_new, W_new
        
        # Return the best solution found even if not fully converged
        valid, error = verify_np(U, V, W, r)
        if error < 1e-6:
            return U, V, W, error
        return None
    
    def polish(U, V, W, r, max_iters=100):
        """
        Final polish pass with tighter tolerance and more iterations.
        """
        prev_error = float('inf')
        for iteration in range(max_iters):
            if time.time() > DEADLINE - 5:
                return None
            
            U_new = solve_for_factor_U(U, V, W, r)
            V_new = solve_for_factor_V(U_new, V, W, r)
            W_new = solve_for_factor_W(U_new, V_new, W, r)
            
            valid, error = verify_np(U_new, V_new, W_new, r)
            
            if valid:
                return U_new, V_new, W_new, error
            
            # Check for stagnation
            if iteration > 5 and abs(prev_error - error) < 1e-14:
                break
            
            prev_error = error
            U, V, W = U_new, V_new, W_new
        
        # Return if error is small enough for rational reconstruction
        valid, error = verify_np(U, V, W, r)
        if error < 1e-8:
            return U, V, W, error
        return None
    
    def try_round_and_verify(U, V, W, r):
        """Try to round to half-integers and verify."""
        # Try rounding to half-integers
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        valid, error = verify_np(U_rounded, V_rounded, W_rounded, r)
        if valid:
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        valid, error = verify_np(U_int, V_int, W_int, r)
        if valid:
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Try a few different rounding strategies
        for scale in [1, 2, 4, 8]:
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            
            valid, error = verify_np(U_s, V_s, W_s, r)
            if valid:
                # Check if values are half-integers
                if all(abs(v * 2 - round(v * 2)) < 1e-9 for row in U_s for v in row) and \
                   all(abs(v * 2 - round(v * 2)) < 1e-9 for row in V_s for v in row) and \
                   all(abs(v * 2 - round(v * 2)) < 1e-9 for row in W_s for v in row):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
        
        return None
    
    def structured_init(r, seed=42):
        """
        Create a structured initialization based on the algebraic structure.
        Uses a mix of random and structured patterns to help convergence.
        """
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Add some structured components to help with convergence
        # The diagonal entries should be non-zero for a valid decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Ensure the (i,j,k) diagonal term has some weight
                    idx = (i * m + j) * p + k
                    if idx < r:
                        U[i * m + j, idx] += 0.5
                        V[j * p + k, idx] += 0.5
                        W[k * n + i, idx] += 0.5
        
        return U, V, W
    
    # Try to find a minimal rank decomposition
    best_result = None
    best_rank = n * m * p + 1
    
    def rational_reconstruct(arr, max_denom=8):
        """Try to reconstruct each entry as a rational with small denominator."""
        from fractions import Fraction
        result = []
        for row in arr:
            new_row = []
            for v in row:
                # Try to find p/q with q <= max_denom such that |v - p/q| < 1e-9
                found = False
                for q in range(1, max_denom + 1):
                    p = round(v * q)
                    if abs(v - p / q) < 1e-9:
                        new_row.append(p / q)
                        found = True
                        break
                if not found:
                    # Fall back to nearest half-integer
                    new_row.append(round(v * 2) / 2.0)
            result.append(new_row)
        return result
    
    def verify_rational(U, V, W, r):
        """Verify exact decomposition with rational entries."""
        from fractions import Fraction
        U_f = [[Fraction(v) for v in row] for row in U]
        V_f = [[Fraction(v) for v in row] for row in V]
        W_f = [[Fraction(v) for v in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(r):
                        val += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                    if val != 1:
                        return False
        return True
    
    # The true rank of the 2x3x6 matrix multiplication tensor is known to be 18
    # (from literature on small matrix multiplication)
    # Try ranks from 18 upward
    for r in range(18, 37):
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple random initializations with more seeds at lower ranks
        num_seeds = 10 if r <= 20 else 6
        seeds = list(range(42, 42 + num_seeds * 37))  # Spaced seeds
        
        found = False
        for seed in seeds:
            if time.time() > DEADLINE - 10:
                break
            
            U, V, W = structured_init(r, seed)
            
            # Run ALS with moderate iterations
            result = alternating_least_squares(r, U, V, W, max_iters=30)
            
            if result is not None:
                U_sol, V_sol, W_sol, error = result
                
                # Try rational reconstruction
                U_rat = rational_reconstruct(U_sol, max_denom=8)
                V_rat = rational_reconstruct(V_sol, max_denom=8)
                W_rat = rational_reconstruct(W_sol, max_denom=8)
                
                if verify_rational(U_rat, V_rat, W_rat, r):
                    return {"U": U_rat, "V": V_rat, "W": W_rat}
                
                # Try rounding to half-integers
                rounded = try_round_and_verify(U_sol, V_sol, W_sol, r)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # Polish and try again
                polished = polish(U_sol, V_sol, W_sol, r, max_iters=20)
                if polished is not None:
                    U_p, V_p, W_p, error_p = polished
                    U_rat_p = rational_reconstruct(U_p, max_denom=8)
                    V_rat_p = rational_reconstruct(V_p, max_denom=8)
                    W_rat_p = rational_reconstruct(W_p, max_denom=8)
                    
                    if verify_rational(U_rat_p, V_rat_p, W_rat_p, r):
                        return {"U": U_rat_p, "V": V_rat_p, "W": W_rat_p}
                    
                    rounded_p = try_round_and_verify(U_p, V_p, W_p, r)
                    if rounded_p is not None:
                        U_r, V_r, W_r = rounded_p
                        return {"U": U_r, "V": V_r, "W": W_r}
                
                found = True
        
        if found:
            continue
    
    # Fallback: try to find any valid decomposition at rank 36 (schoolbook)
    # This is guaranteed to work
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
# EVOLVE-BLOCK-END
