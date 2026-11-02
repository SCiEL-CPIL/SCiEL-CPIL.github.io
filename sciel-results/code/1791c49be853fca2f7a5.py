import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_solution_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m+j]
            for k in range(p):
                v_row = Va[j*p+k]
                w_row = Wa[k*n+i]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1) > 1e-8:
                    return False
    return True

def _verify_half_integer(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if v is not None:
                    v2 = v * 2
                    if abs(v2 - round(v2)) > 1e-9:
                        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin within 150s
    
    # Schoolbook decomposition (Rank 105)
    best_solution = _build_schoolbook()
    best_rank = n * m * p  # 105
    
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # 35
    
    n_eq = n * m * p  # 105 equations
    n_unk = p * n  # 21 W-rows
    
    # Precompute equation indices
    # For equation (i,j,k): eq_idx = i*m*p + j*p + k
    # For unknown (k,i,r): unk_idx = (k*n + i) * R + r
    
    # Strategy 1: Try to find rank R solutions by fixing U,V and solving for W
    # Use various random initialization strategies
    
    def try_solve_for_W(U_rand, V_rand, R):
        """Given U and V, solve for W using least squares."""
        n_unk_total = p * n * R
        A = np.zeros((n_eq, n_unk_total))
        
        for i in range(n):
            for j in range(m):
                u_row = U_rand[i * m + j]  # (R,)
                for k in range(p):
                    eq_idx = i * (m * p) + j * p + k
                    v_row = V_rand[j * p + k]  # (R,)
                    coeff = u_row * v_row  # (R,)
                    nz = np.nonzero(coeff)[0]
                    if len(nz) > 0:
                        for r in nz:
                            unk_idx = (k * n + i) * R + r
                            A[eq_idx, unk_idx] = coeff[r]
        
        b = np.ones(n_eq)
        try:
            w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Round to nearest half-integer
            w_rounded = np.round(w_sol * 2) / 2
            
            # Verify: A @ w_rounded should equal b
            if np.allclose(A @ w_rounded, b, atol=1e-8):
                # Construct W from w_rounded
                W_list = [[0] * R for _ in range(p * n)]
                for idx in range(n_unk_total):
                    ki = idx // R
                    r = idx % R
                    k_val = ki // n
                    i_val = ki % n
                    W_list[k_val * n + i_val][r] = w_rounded[idx]
                return W_list
        except Exception:
            pass
        return None
    
    def try_random_search(R, num_trials, entry_set, time_limit):
        """Try random U,V with given entry set, solve for W."""
        for t in range(num_trials):
            if time.time() > time_limit:
                break
            
            # Generate random U and V
            U_rand = np.array([[np.random.choice(entry_set) for _ in range(R)] for _ in range(n * m)], dtype=np.float64)
            V_rand = np.array([[np.random.choice(entry_set) for _ in range(R)] for _ in range(m * p)], dtype=np.float64)
            
            # Solve for W
            W_list = try_solve_for_W(U_rand, V_rand, R)
            
            if W_list is not None:
                # Final verification
                if _verify_solution_fast(U_rand.tolist(), V_rand.tolist(), W_list) and \
                   _verify_half_integer(U_rand.tolist(), V_rand.tolist(), W_list):
                    return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_list}
        return None
    
    # Try different entry sets and ranks
    entry_sets = [
        [-1, 0, 1],
        [-1, 1],
        [-2, -1, 0, 1, 2],
        [-0.5, 0, 0.5],
        [-1, -0.5, 0, 0.5, 1],
    ]
    
    # Try ranks from 104 down to lower_bound
    # Focus on higher ranks first (more likely to find solutions)
    ranks_to_try = list(range(min(104, best_rank - 1), lower_bound - 1, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Adaptive number of trials based on rank
        if R >= 100:
            max_trials = 30
        elif R >= 90:
            max_trials = 20
        elif R >= 70:
            max_trials = 15
        else:
            max_trials = 10
        
        # Try different entry sets
        for entry_set in entry_sets:
            if time.time() > deadline - 15:
                break
            
            result = try_random_search(R, max_trials, entry_set, deadline - 15)
            if result is not None:
                return result
    
    return best_solution
# EVOLVE-BLOCK-END
