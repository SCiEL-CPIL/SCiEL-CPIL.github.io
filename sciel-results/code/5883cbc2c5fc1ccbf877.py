"""Combined approach for matrix multiplication 3x4x8 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 96

# EVOLVE-BLOCK-START

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    Convention: W index is k*n+i.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    n, m, p = N, M, P
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is integer)."""
    return abs(2.0 * x - round(2.0 * x)) < 1e-9

def _all_half_integers(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _schoolbook(n, m, p):
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

def _random_candidate(n, m, p, rank, seed):
    """Generate a random candidate with small integer/half-integer entries."""
    rng = random.Random(seed)
    vals = [0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0, -2.0]
    U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}

def _local_search(U, V, W, n, m, p, max_iters=500, deadline=None):
    """Try to reduce rank by zeroing columns that don't contribute.
    Returns (new_U, new_V, new_W, new_rank) or None if no improvement.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Try removing columns one at a time (greedy)
    active = list(range(rank))
    current_U = U_arr[:, active]
    current_V = V_arr[:, active]
    current_W = W_arr[:, active]
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for idx in range(len(active)):
            # Try removing this column
            new_active = active[:idx] + active[idx+1:]
            if not new_active:
                break
            test_U = U_arr[:, new_active]
            test_V = V_arr[:, new_active]
            test_W = W_arr[:, new_active]
            
            # Quick check: verify
            if _verify(test_U.tolist(), test_V.tolist(), test_W.tolist()):
                active = new_active
                current_U = test_U
                current_V = test_V
                current_W = test_W
                improved = True
                break
    
    if len(active) < rank:
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), len(active)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = N, M, P
    schoolbook_rank = n * m * p  # 96
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # First, try to reduce the schoolbook decomposition via column elimination
    elim_result = _local_search(best["U"], best["V"], best["W"], n, m, p, 
                                max_iters=2000, deadline=deadline - 5.0)
    if elim_result is not None:
        best["U"], best["V"], best["W"], best_rank = elim_result
    
    # Now try randomized search for even lower ranks
    seed_counter = 0
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 10.0:
            break
        
        remaining_time = deadline - 10.0 - time.time()
        if remaining_time <= 0:
            break
        
        # More trials for ranks close to current best
        max_trials = min(100, max(10, int(remaining_time * 5)))
        
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
            
            seed_counter += 1
            candidate = _random_candidate(n, m, p, rank, seed_counter)
            
            if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                if _all_half_integers(candidate["U"], candidate["V"], candidate["W"]):
                    # Try to further reduce via elimination
                    elim = _local_search(candidate["U"], candidate["V"], candidate["W"],
                                         n, m, p, max_iters=500, deadline=deadline - 5.0)
                    if elim is not None:
                        candidate["U"], candidate["V"], candidate["W"], candidate_rank = elim
                    else:
                        candidate_rank = rank
                    
                    if candidate_rank < best_rank:
                        best = candidate
                        best_rank = candidate_rank
                        found = True
                        break
        
        if found:
            continue
    
    # Clean entries: convert to int if possible, round half-integers
    def _clean(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if _is_half_integer(val):
                    cleaned = round(val * 2) / 2.0
                    if abs(cleaned - round(cleaned)) < 1e-9:
                        new_row.append(int(round(cleaned)))
                    else:
                        new_row.append(cleaned)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    best["U"] = _clean(best["U"])
    best["V"] = _clean(best["V"])
    best["W"] = _clean(best["W"])
    
    return best

# EVOLVE-BLOCK-END
