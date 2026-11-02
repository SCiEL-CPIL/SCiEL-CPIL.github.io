"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def validate(U, V, W, rank_val):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Strategy: Fix U and V with small integer entries, solve linear system for W
    # The constraint is: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # This is linear in W!
    
    def try_solve_for_W(U_list, V_list, target_rank):
        """
        Given U (n*m x rank) and V (m*p x rank), solve for W (p*n x rank).
        The constraints are linear in W.
        """
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Group by (k,i): for fixed (k,i), we have n*m equations in rank unknowns W[k*n+i, :]
        # But the equations couple through the same W row
        
        # Actually, let's set up the full linear system
        # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(rank)
        # That's p*n*rank unknowns
        
        # Equations: for each (i,j,k), one equation
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        num_eqs = n * m * p
        num_unknowns = p * n * target_rank
        
        # Build A matrix and b vector
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        # W index: k*n+i, column c
                        w_row = k * n + i
                        w_col = c
                        unknown_idx = w_row * target_rank + w_col
                        A[eq_idx, unknown_idx] = U_arr[i*m+j, c] * V_arr[j*p+k, c]
                    eq_idx += 1
        
        # Solve the underdetermined system
        # Use least squares, then check if solution is close to half-integers
        try:
            # Use pseudo-inverse
            W_sol = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            return None
        
        # Check if solution is close to half-integers
        W_rounded = np.round(W_sol * 2) / 2.0
        W_rounded = np.round(W_rounded * 2) / 2.0  # double round for safety
        
        # Verify
        W_check = W_rounded.reshape(p*n, target_rank)
        residual = np.zeros(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    residual[eq_idx] = np.dot(U_arr[i*m+j] * V_arr[j*p+k], W_check[w_row]) - 1.0
                    eq_idx += 1
        
        if np.max(np.abs(residual)) < 1e-9:
            # Convert to list of lists
            W_list = W_check.tolist()
            return W_list
        return None
    
    # Try random U and V with entries in {0, ±1}, solve for W
    def random_search_solve_for_W(target_rank, max_attempts=200):
        """Randomly generate U and V, solve for W, check if W is half-integer."""
        for attempt in range(max_attempts):
            if elapsed() > deadline - 5:
                return None, None, None
            
            # Generate sparse random U and V
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            
            # Fill U with sparse ±1 entries
            for row in range(n*m):
                num_nonzero = random.randint(1, max(1, target_rank // 4))
                cols = random.sample(range(target_rank), num_nonzero)
                for c in cols:
                    U[row][c] = random.choice([1, -1])
            
            # Fill V with sparse ±1 entries
            for row in range(m*p):
                num_nonzero = random.randint(1, max(1, target_rank // 4))
                cols = random.sample(range(target_rank), num_nonzero)
                for c in cols:
                    V[row][c] = random.choice([1, -1])
            
            # Solve for W
            W = try_solve_for_W(U, V, target_rank)
            if W is not None:
                return U, V, W
        
        return None, None, None
    
    # Try ranks from 40 to 71
    for target_rank in range(40, 72):
        if elapsed() > deadline - 10:
            break
        
        U_r, V_r, W_r = random_search_solve_for_W(target_rank, max_attempts=100)
        
        if U_r is not None:
            # Verify
            if validate(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                # Continue trying lower ranks
                continue
    
    # Also try structured approach: use block structure
    # View B as [B1 | B2] where B1, B2 are 3x4
    # Use shared A structure
    
    # Try a more targeted construction based on known results
    # For 3x3*3x3, Laderman gives rank 20
    # We can potentially use 20 + something for the extra 4 columns
    
    # Try rank 60 with more structured random search
    if elapsed() < deadline - 15:
        for target_rank in [60, 58, 56, 54, 52, 50]:
            if elapsed() > deadline - 10:
                break
            U_r, V_r, W_r = random_search_solve_for_W(target_rank, max_attempts=200)
            if U_r is not None and validate(U_r, V_r, W_r, target_rank):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
