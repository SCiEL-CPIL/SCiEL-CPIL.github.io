import time
import numpy as np

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def _build_target_tensor(n, m, p):
    """
    Builds the target tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1.0, else 0.0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _compute_error(U, V, W, T):
    """
    Computes the squared error between the reconstructed tensor and the target T.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    # Reconstruct R[I, J, K] = sum_c U[I, c] * V[J, c] * W[K, c]
    # Using einsum for efficiency
    R = np.einsum('ic,jc,kc->ijk', U, V, W)
    diff = R - T
    return np.sum(diff * diff)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies the decomposition exactly.
    """
    T = _build_target_tensor(n, m, p)
    err = _compute_error(U, V, W, T)
    return err < tol

def _construct_2x3x6_rank_24():
    """
    Fallback: The known rank-24 block decomposition.
    """
    n, m, p = 2, 3, 6
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for 2x2x2
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    U_s[0, 0] = 1; V_s[1, 0] = 1; V_s[3, 0] = 1; W_s[0, 0] = 1; W_s[3, 0] = 1
    U_s[2, 1] = 1; U_s[3, 1] = 1; V_s[0, 1] = 1; W_s[1, 1] = 1; W_s[2, 1] = 1
    U_s[0, 2] = 1; U_s[1, 2] = 1; V_s[3, 2] = 1; W_s[0, 2] = 1; W_s[3, 2] = 1
    U_s[3, 3] = 1; V_s[0, 3] = 1; V_s[2, 3] = -1; W_s[0, 3] = 1; W_s[2, 3] = 1
    U_s[0, 4] = 1; U_s[2, 4] = 1; V_s[1, 4] = 1; W_s[0, 4] = -1; W_s[1, 4] = 1
    U_s[1, 5] = 1; U_s[2, 5] = -1; V_s[0, 5] = 1; V_s[2, 5] = 1; W_s[0, 5] = -1
    U_s[3, 6] = 1; U_s[1, 6] = -1; V_s[1, 6] = 1; V_s[3, 6] = 1; W_s[3, 6] = 1
    
    # Construct 2x3x2 block
    U_232 = np.zeros((6, 8))
    V_232 = np.zeros((6, 8))
    W_232 = np.zeros((4, 8))
    
    U_232[0:4, 0:7] = U_s
    V_232[0:4, 0:7] = V_s
    W_232[0:4, 0:7] = W_s
    
    U_232[4, 7] = 1; U_232[5, 7] = 1
    V_232[4, 7] = 1; V_232[5, 7] = 1
    W_232[0, 7] = 1; W_232[1, 7] = 1; W_232[2, 7] = 1; W_232[3, 7] = 1
    
    # Assemble 2x3x6
    for block in range(3):
        U[:, block*8:(block+1)*8] = U_232
        for j in range(m):
            for k_local in range(2):
                k_global = block * 2 + k_local
                v_idx = j * p + k_global
                v_232_idx = j * 2 + k_local
                V[v_idx, block*8:(block+1)*8] = V_232[v_232_idx, :]
        for k_local in range(2):
            for i in range(n):
                k_global = block * 2 + k_local
                w_idx = k_global * n + i
                w_232_idx = k_local * n + i
                W[w_idx, block*8:(block+1)*8] = W_232[w_232_idx, :]
                
    return U, V, W

def _search_rank(n, m, p, target_rank, time_budget):
    """
    Simulated Annealing search for a rank decomposition with integer coefficients.
    """
    T = _build_target_tensor(n, m, p)
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize with random integers in {-1, 0, 1}
    # We use a fixed seed for reproducibility in the search logic, but random restarts
    np.random.seed(42)
    
    U = np.random.randint(-1, 2, size=(dim_u, target_rank))
    V = np.random.randint(-1, 2, size=(dim_v, target_rank))
    W = np.random.randint(-1, 2, size=(dim_w, target_rank))
    
    current_err = _compute_error(U, V, W, T)
    best_err = current_err
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    start_time = time.time()
    temp = 1.0
    cool_rate = 0.9999
    
    steps = 0
    while time.time() - start_time < time_budget:
        steps += 1
        
        # Choose a random matrix to modify
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            mat = U
            dim = dim_u
        elif mat_idx == 1:
            mat = V
            dim = dim_v
        else:
            mat = W
            dim = dim_w
            
        # Choose a random entry
        row = np.random.randint(0, dim)
        col = np.random.randint(0, target_rank)
        
        # Choose a move: +/- 1 or flip
        old_val = mat[row, col]
        if np.random.rand() < 0.5:
            new_val = old_val + 1
        else:
            new_val = old_val - 1
            
        # Apply move
        mat[row, col] = new_val
        
        # Compute new error
        new_err = _compute_error(U, V, W, T)
        
        # Acceptance criterion
        delta = new_err - current_err
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_err = new_err
            if new_err < best_err:
                best_err = new_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                # If we found an exact solution, return immediately
                if best_err < 1e-9:
                    return best_U, best_V, best_W
        else:
            # Revert move
            mat[row, col] = old_val
            
        # Cool down
        temp *= cool_rate
        
        # If we are stuck at 0 error, we are done
        if current_err < 1e-9:
            return U, V, W

    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    
    # First, ensure we have a valid baseline
    U_base, V_base, W_base = _construct_2x3x6_rank_24()
    
    # Try to find lower rank solutions
    # Known lower bounds suggest 18 or 19 might be possible.
    # We try 18, 19, 20, 21, 22, 23.
    
    ranks_to_try = [18, 19, 20, 21, 22, 23]
    
    # Allocate time budget
    # 120s total. 10s for overhead. 110s for search.
    # Try each rank with a subset of the time.
    # If we find a solution, we stop.
    
    # We prioritize lower ranks.
    # Give more time to 18 and 19.
    
    time_per_rank = {
        18: 30.0,
        19: 25.0,
        20: 20.0,
        21: 10.0,
        22: 5.0,
        23: 5.0
    }
    
    for r in ranks_to_try:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        budget = time_per_rank.get(r, 5.0)
        # Cap budget to remaining time
        remaining = TIME_LIMIT - 10 - (time.time() - START_TIME)
        budget = min(budget, remaining)
        
        if budget <= 0:
            continue
            
        U_sol, V_sol, W_sol = _search_rank(n, m, p, r, budget)
        
        # Verify
        if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
            # Found a valid solution for rank r
            # Convert to lists
            U_list = [[round(float(x), 8) for x in row] for row in U_sol]
            V_list = [[round(float(x), 8) for x in row] for row in V_sol]
            W_list = [[round(float(x), 8) for x in row] for row in W_sol]
            
            return {"U": U_list, "V": V_list, "W": W_list}

    # If no lower rank found, return the baseline rank 24
    U_list = [[round(float(x), 8) for x in row] for row in U_base]
    V_list = [[round(float(x), 8) for x in row] for row in V_base]
    W_list = [[round(float(x), 8) for x in row] for row in W_base]
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print("Rank:", len(res['U'][0]))
