"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Use float for speed, assume exactness if close
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact verification using Fractions. Returns True if exact."""
    rank = U.shape[1]
    # Convert to Fraction matrices
    # Optimization: if inputs are integers, use Fraction(int)
    # We assume inputs are integers or half-integers (multiples of 0.5)
    # Let's convert to Fraction for safety
    
    # To avoid overhead, we can check if floats are effectively integers
    # But for correctness, we use Fraction.
    # Note: This is slow, so only use if rank is small or as final check.
    
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for r in range(rank):
        for u_idx in range(n*m):
            val = U[u_idx, r]
            if abs(val - round(val*2)/2) < 1e-9:
                U_frac[u_idx, r] = Fraction(round(val*2), 2)
            else:
                return False # Not integer/half-integer
        
        for v_idx in range(m*p):
            val = V[v_idx, r]
            if abs(val - round(val*2)/2) < 1e-9:
                V_frac[v_idx, r] = Fraction(round(val*2), 2)
            else:
                return False
        
        for w_idx in range(p*n):
            val = W[w_idx, r]
            if abs(val - round(val*2)/2) < 1e-9:
                W_frac[w_idx, r] = Fraction(round(val*2), 2)
            else:
                return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[i*m+j, r] * V_frac[j*p+k, r] * W_frac[k*n+i, r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_sparse_candidate(n, m, p, rank, max_nonzeros_per_col=2, seed=None):
    """
    Generate a random candidate with sparse structure.
    Each column in U, V, W has at most `max_nonzeros_per_col` non-zero entries.
    Values are chosen from {0, 0.5, 1, 1.5, ...} or small integers.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Possible values for entries
    # Let's try integers and half-integers in range [-3, 3]
    vals = [i/2.0 for i in range(-6, 7)]
    
    for r in range(rank):
        # Pick indices for U
        u_idxs = random.sample(range(n*m), min(max_nonzeros_per_col, n*m))
        for idx in u_idxs:
            U[idx, r] = random.choice(vals)
        
        # Pick indices for V
        v_idxs = random.sample(range(m*p), min(max_nonzeros_per_col, m*p))
        for idx in v_idxs:
            V[idx, r] = random.choice(vals)
        
        # Pick indices for W
        w_idxs = random.sample(range(p*n), min(max_nonzeros_per_col, p*n))
        for idx in w_idxs:
            W[idx, r] = random.choice(vals)
            
    return U, V, W

def _calculate_error(U, V, W, n, m, p):
    """Calculate sum of squared errors."""
    err = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # s = dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                # Vectorized dot product
                term = V[j*p+k, :] * W[k*n+i, :]
                s = np.dot(U[i*m+j, :], term)
                diff = 1.0 - s
                err += diff * diff
    return err

def _local_search_integer(n, m, p, rank, time_limit, seed=None):
    """
    Perform local search on integer/half-integer grid.
    Start from a random sparse candidate.
    Mutate single entries.
    Accept if error decreases.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Initial candidate
    # Try different sparsity levels
    for sparsity in [1, 2, 3]:
        U, V, W = _random_sparse_candidate(n, m, p, rank, max_nonzeros_per_col=sparsity, seed=seed)
        current_err = _calculate_error(U, V, W, n, m, p)
        
        if current_err < 1e-9:
            return {"U": U, "V": V, "W": W}
            
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = current_err
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 5000 # Reset if stuck
        
        while time.time() - start_time < time_limit:
            # Choose which matrix and entry to mutate
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                rows = n*m
            elif mat_idx == 1:
                mat = V
                rows = m*p
            else:
                mat = W
                rows = p*n
            
            row_idx = random.randint(0, rows - 1)
            col_idx = random.randint(0, rank - 1)
            
            # Choose a new value
            # Perturb current value or pick random from grid
            # Let's try small perturbations first
            if random.random() < 0.7:
                # Perturb by +/- 0.5 or +/- 1
                delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = mat[row_idx, col_idx] + delta
            else:
                # Random new value from grid
                new_val = random.choice([i/2.0 for i in range(-6, 7)])
                
            # Keep values bounded to avoid overflow/instability
            if abs(new_val) > 10:
                continue
            
            # Store original
            old_val = mat[row_idx, col_idx]
            mat[row_idx, col_idx] = new_val
            
            # Calculate new error
            new_err = _calculate_error(U, V, W, n, m, p)
            
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve_count = 0
                if new_err < 1e-9:
                    return {"U": U, "V": V, "W": W}
            else:
                # Revert
                mat[row_idx, col_idx] = old_val
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                # Restart with new random candidate of same sparsity
                U, V, W = _random_sparse_candidate(n, m, p, rank, max_nonzeros_per_col=sparsity, seed=random.randint(0, 10000))
                current_err = _calculate_error(U, V, W, n, m, p)
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_err = current_err
                no_improve_count = 0
                if best_err < 1e-9:
                    return {"U": U, "V": V, "W": W}
                
    return None

def _try_rank(n, m, p, rank, time_budget, seed_base=0):
    """Try to find a decomposition of given rank within time budget."""
    best_result = None
    start = time.time()
    
    # Try a few random restarts
    attempts = 0
    while time.time() - start < time_budget and attempts < 10:
        seed = seed_base + attempts
        result = _local_search_integer(n, m, p, rank, time_limit=time_budget/10.0, seed=seed)
        if result is not None:
            return result
        attempts += 1
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank = 90)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Known results:
    # 3x3x3: 23
    # 3x4x4: 30?
    # 3x5x6: ?
    # Let's try ranks from 30 up to 89.
    # We prioritize lower ranks first, but spend more time on promising ones.
    # Given the difficulty, let's start from 30 and go up.
    # If we find one, we stop.
    
    # To be efficient, we can test a few random ranks first, then sweep.
    # But since we need MINIMAL rank, we must find the absolute minimum.
    # However, the task says "Find a rank-minimal exact tensor decomposition".
    # This implies we need the BEST possible rank.
    # If we don't find the absolute minimum, we should return the best we found.
    # But the score is `rank`, so lower is better.
    
    # Strategy:
    # 1. Try ranks 30-40 with significant time.
    # 2. If none found, try 40-50.
    # 3. If none found, try 50-60.
    # ...
    # 4. Return best found (or schoolbook).
    
    ranks_to_try = list(range(30, 90))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Allocate time proportional to likelihood?
        # Early ranks are harder. Let's give each rank a chunk of time.
        # Total time available: ~140s.
        # Ranks: 60.
        # ~2.3s per rank. Might be too little for hard ones.
        # Let's give more time to lower ranks.
        
        if rank < 40:
            budget = 5.0
        elif rank < 60:
            budget = 3.0
        else:
            budget = 1.0
            
        if time.time() + budget > deadline - 5:
             break
             
        result = _try_rank(n, m, p, rank, time_budget=budget, seed_base=rank*100)
        
        if result is not None:
            best_result = result
            best_rank = rank
            # We found a rank. Can we do better?
            # We need to check lower ranks. But we are iterating from low to high.
            # So if we find rank R, we should check R-1, R-2...
            # Wait, the loop is `for rank in ranks_to_try` which is 30, 31, ...
            # So if we find 50, we have already checked 30-49.
            # So 50 is the minimum found so far.
            # We can break if we want to assume this is minimal?
            # No, we should continue checking lower? No, we started from 30.
            # So if we find rank 50, it means 30-49 failed.
            # So 50 is the minimal found.
            # We can break and return this?
            # Or continue to see if we find 49 later? No, 49 was already tried.
            # So we can break.
            break
            
    # If no rank < 90 was found, best_result is still schoolbook.
    
    # Final verification of the best result to be safe
    if best_result is not None:
        if not _verify_decomposition_fast(best_result["U"], best_result["V"], best_result["W"], n, m, p):
             # Fallback to schoolbook if verification fails (shouldn't happen)
             best_result = _schoolbook(n, m, p)
             best_rank = n * m * p

    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
# EVOLVE-BLOCK-END
