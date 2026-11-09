"""Search-based program for matrix multiplication 3x4x8."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, tol=1e-9):
    """Fast verification using numpy einsum."""
    n, m, p = 3, 4, 8
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _is_half_integer(x, tol=1e-9):
    return abs(2.0 * x - round(2.0 * x)) < tol

def _all_half_integers(U, V, W, tol=1e-9):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
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

def _try_structured_block(n, m, p, deadline):
    """Try block structure exploiting 8 = 2*4.
    
    View B as [B1, B2] where B1, B2 are 4x4.
    Then C = A * [B1, B2] = [A*B1, A*B2].
    This doesn't directly reduce rank, but we can try to share some terms.
    """
    n, m, p = 3, 4, 8
    # Try rank = n*m*p - d for small d
    # The idea: if two terms can be combined, we save 1 rank
    
    # Actually, let's try a different approach: 
    # For each pair of (i,j,k) and (i',j',k') that share structure,
    # see if we can merge them.
    
    # This is complex, so let's try random perturbation of schoolbook
    return None

def _local_search(n, m, p, start_rank, deadline, max_trials=500):
    """Local search: start from schoolbook, try to reduce rank by 
    randomly zeroing out entries and checking if decomposition still works."""
    n, m, p = 3, 4, 8
    best = None
    best_rank = start_rank
    
    for rank in range(start_rank - 1, max(32, start_rank - 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            rng = random.Random(trial * 1000 + rank)
            
            # Start with schoolbook of full rank, then try to reduce
            # by setting some entries to 0 in a pattern
            
            # Alternative: generate random sparse matrices
            density = 0.3  # 30% of entries are non-zero
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Randomly place 1s and -1s
            for i in range(n * m):
                for c in range(rank):
                    if rng.random() < density:
                        U[i][c] = rng.choice([1, -1])
            for j in range(m * p):
                for c in range(rank):
                    if rng.random() < density:
                        V[j][c] = rng.choice([1, -1])
            for k in range(p * n):
                for c in range(rank):
                    if rng.random() < density:
                        W[k][c] = rng.choice([1, -1])
            
            if _verify_fast(U, V, W):
                if _all_half_integers(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
    
    return best, best_rank

def _random_search(n, m, p, start_rank, end_rank, deadline, max_trials_per_rank=100):
    """Random search over ranks from start_rank down to end_rank."""
    n, m, p = 3, 4, 8
    best = None
    best_rank = start_rank
    
    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Adjust trials based on remaining time
        trials = min(max_trials_per_rank, max(10, int(remaining * 2)))
        
        found = False
        for trial in range(trials):
            if time.time() > deadline:
                break
            
            seed = rank * 10000 + trial
            rng = random.Random(seed)
            
            # Try different sparsity patterns
            density = rng.choice([0.2, 0.3, 0.4, 0.5])
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for i in range(n * m):
                for c in range(rank):
                    if rng.random() < density:
                        U[i][c] = rng.choice([1, -1, 0.5, -0.5])
            for j in range(m * p):
                for c in range(rank):
                    if rng.random() < density:
                        V[j][c] = rng.choice([1, -1, 0.5, -0.5])
            for k in range(p * n):
                for c in range(rank):
                    if rng.random() < density:
                        W[k][c] = rng.choice([1, -1, 0.5, -0.5])
            
            if _verify_fast(U, V, W):
                if _all_half_integers(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
    
    return best, best_rank

def _perturb_schoolbook(n, m, p, deadline):
    """Start from schoolbook and try to merge terms by perturbing."""
    n, m, p = 3, 4, 8
    schoolbook = _schoolbook(n, m, p)
    full_rank = n * m * p
    
    # Try to reduce by 1, 2, 3... by finding if any term can be removed
    # or if two terms can be merged
    
    best = schoolbook
    best_rank = full_rank
    
    # For each possible reduction d, try to find a valid decomposition
    for d in range(1, 15):
        if time.time() > deadline:
            break
        
        target_rank = full_rank - d
        
        # Try random perturbations of the schoolbook structure
        for trial in range(20):
            if time.time() > deadline:
                break
            
            rng = random.Random(d * 1000 + trial)
            
            U = [row[:] for row in schoolbook["U"]]
            V = [row[:] for row in schoolbook["V"]]
            W = [row[:] for row in schoolbook["W"]]
            
            # Randomly modify some entries to try to create dependencies
            # that allow rank reduction
            for _ in range(d * 2):
                mat = rng.choice([U, V, W])
                row = rng.randrange(len(mat))
                col = rng.randrange(full_rank)
                mat[row][col] += rng.choice([1, -1, 0.5, -0.5])
            
            # Truncate to target_rank columns
            U_t = [row[:target_rank] for row in U]
            V_t = [row[:target_rank] for row in V]
            W_t = [row[:target_rank] for row in W]
            
            if _verify_fast(U_t, V_t, W_t):
                if _all_half_integers(U_t, V_t, W_t):
                    best = {"U": U_t, "V": V_t, "W": W_t}
                    best_rank = target_rank
                    break
    
    return best, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Strategy 1: Perturb schoolbook to find small reductions
    if time.time() < deadline - 30:
        s, sr = _perturb_schoolbook(n, m, p, deadline - 30)
        if s is not None and sr < best_rank:
            best = s
            best_rank = sr
    
    # Strategy 2: Local search with sparse random matrices
    if time.time() < deadline - 15:
        s, sr = _local_search(n, m, p, best_rank, deadline - 15, max_trials=300)
        if s is not None and sr < best_rank:
            best = s
            best_rank = sr
    
    # Strategy 3: Random search over a range of ranks
    if time.time() < deadline - 5:
        s, sr = _random_search(n, m, p, best_rank, max(40, best_rank - 30), deadline - 5, max_trials_per_rank=50)
        if s is not None and sr < best_rank:
            best = s
            best_rank = sr
    
    # Clean up entries
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
