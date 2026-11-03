"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 130s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def round_to_half(x):
        return round(x * 2) / 2.0
    
    def verify(U, V, W, r):
        """Verify that U, V, W form a valid rank-r decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    total_error += (val - 1.0) ** 2
        return total_error < 1e-9, total_error
    
    def solve_for_factor(factor_name, U, V, W, r):
        """
        Solve for one factor given the other two using least squares.
        Returns the updated factor.
        """
        # Build the linear system: for each (i,j,k), sum_c other1[i,j,k,c] * X[relevant,c] = 1
        # The structure depends on which factor we're solving for
        
        if factor_name == 'U':
            # U[i*m+j, c] * sum over k of V[j*p+k,c] * W[k*n+i,c] = 1 for each (i,j,k)
            # Actually: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is linear in U: for each (i,j,k), the equation is:
            # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # Group by (i,j): for fixed (i,j), we have p equations in (n*m*r) unknowns... 
            # No, U[i*m+j, :] is a vector of length r, and we have p equations for each (i,j)
            # Actually all (i,j,k) give equations. Let's build the full system.
            
            A_mat = np.zeros((n * m * p, n * m * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            col = (i * m + j) * r + c
                            A_mat[e, col] = V[j * p + k, c] * W[k * n + i, c]
            
            u_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return u_flat.reshape(n * m, r)
        
        elif factor_name == 'V':
            A_mat = np.zeros((n * m * p, m * p * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            col = (j * p + k) * r + c
                            A_mat[e, col] = U[i * m + j, c] * W[k * n + i, c]
            
            v_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return v_flat.reshape(m * p, r)
        
        elif factor_name == 'W':
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A_mat[e, col] = U[i * m + j, c] * V[j * p + k, c]
            
            w_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return w_flat.reshape(p * n, r)
        
        return None
    
    def alternating_least_squares(r, U, V, W, max_iters=100, tol=1e-10):
        """
        Alternating least squares to find a rank-r decomposition.
        Returns (U, V, W, error) if converged, else None.
        """
        for iteration in range(max_iters):
            if time.time() > DEADLINE:
                return None
            
            # Solve for U
            U_new = solve_for_factor('U', U, V, W, r)
            if U_new is None:
                return None
            
            # Solve for V
            V_new = solve_for_factor('V', U_new, V, W, r)
            if V_new is None:
                return None
            
            # Solve for W
            W_new = solve_for_factor('W', U_new, V_new, W, r)
            if W_new is None:
                return None
            
            # Check convergence
            valid, error = verify_np(U_new, V_new, W_new, r)
            
            if valid:
                return U_new, V_new, W_new, error
            
            # Check if we're making progress
            if iteration > 0:
                prev_error = verify_np(U, V, W, r)[1]
                if abs(prev_error - error) < tol:
                    # Stagnant
                    if error < 1e-6:
                        # Close but not exact - try polish
                        break
                    else:
                        return None
            
            U, V, W = U_new, V_new, W_new
        
        return None
    
    def polish(U, V, W, r, max_iters=50):
        """
        Final polish pass with tighter tolerance.
        """
        for iteration in range(max_iters):
            if time.time() > DEADLINE:
                return None
            
            U_new = solve_for_factor('U', U, V, W, r)
            V_new = solve_for_factor('V', U_new, V, W, r)
            W_new = solve_for_factor('W', U_new, V_new, W, r)
            
            valid, error = verify_np(U_new, V_new, W_new, r)
            
            if valid:
                return U_new, V_new, W_new, error
            
            U, V, W = U_new, V_new, W_new
        
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
        For matrix multiplication, we can use the fact that:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        A natural rank-r decomposition uses r terms where each term corresponds
        to a specific pattern of (i,j,k) interactions.
        """
        np.random.seed(seed)
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        return U, V, W
    
    # Try to find a minimal rank decomposition
    # Start from lower ranks and work up
    
    # The theoretical minimum rank for 2x3x6 is at least max(2*3, 3*6, 2*6) = 18
    # But due to tensor structure, it's likely higher. Let's try from 18 up.
    
    best_result = None
    best_rank = n * m * p + 1
    
    for r in range(18, 37):
        if time.time() > DEADLINE - 5:
            break
        
        # Try multiple random initializations
        found = False
        for seed in [42, 123, 456, 789, 101, 202]:
            if time.time() > DEADLINE - 5:
                break
            
            U, V, W = structured_init(r, seed)
            
            result = alternating_least_squares(r, U, V, W, max_iters=50)
            
            if result is not None:
                U_sol, V_sol, W_sol, error = result
                
                # Try to round
                rounded = try_round_and_verify(U_sol, V_sol, W_sol, r)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # If we have an exact solution but can't round, try polish
                polished = polish(U_sol, V_sol, W_sol, r, max_iters=20)
                if polished is not None:
                    U_p, V_p, W_p, error_p = polished
                    rounded = try_round_and_verify(U_p, V_p, W_p, r)
                    if rounded is not None:
                        U_r, V_r, W_r = rounded
                        return {"U": U_r, "V": V_r, "W": W_r}
                
                found = True
        
        if found:
            # We found an exact decomposition at this rank but couldn't round
            # Continue to see if we can find one that rounds nicely
            continue
    
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
# EVOLVE-BLOCK-END
