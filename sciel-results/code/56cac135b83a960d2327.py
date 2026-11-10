"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_max_discrepancy(signs):
    """Compute maximum absolute signed sum over all arithmetic progressions in [1, n]."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons have abs sum 1
    
    # Iterate over all possible steps
    for step in range(1, n + 1):
        # For a given step, the arithmetic progressions are determined by their start index r.
        # The valid starts are r such that r < n.
        # The sequence for start r is arr[r], arr[r+step], arr[r+2*step], ...
        # These sequences for different r (0 <= r < step) are disjoint and cover all elements.
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            # Compute cumulative sums
            cumsums = np.cumsum(subseq)
            # Find max absolute value
            max_abs = np.max(np.abs(cumsums))
            if max_abs > max_disc:
                max_disc = max_abs
    return max_disc

def _local_search(signs, deadline, rng):
    """Perform local search to minimize max discrepancy."""
    n = len(signs)
    current = list(signs)
    current_disc = _compute_max_discrepancy(current)
    best = list(current)
    best_disc = current_disc
    
    start_time = time.time()
    iterations = 0
    
    while time.time() < deadline:
        # Try flipping a random position
        i = rng.randint(0, n - 1)
        old_val = current[i]
        current[i] = -old_val
        new_disc = _compute_max_discrepancy(current)
        
        if new_disc <= current_disc:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best = list(current)
            iterations += 1
        else:
            current[i] = old_val
    
    return best, best_disc

def construct():
    # Use a deadline that is safely within the 60s limit (e.g., 50s)
    # The evaluator kills at 90s, but the task says 60s limit.
    deadline = time.time() + 50
    rng = random.Random(171)
    
    # Start with random initialization
    best_signs = [rng.choice((-1, 1)) for _ in range(256)]
    best_disc = _compute_max_discrepancy(best_signs)
    
    # Local search from initial random
    best_signs, best_disc = _local_search(best_signs, deadline, rng)
    
    # Try multiple restarts if time permits
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        candidate = [rng.choice((-1, 1)) for _ in range(256)]
        cand_disc = _compute_max_discrepancy(candidate)
        if cand_disc < best_disc:
            best_signs = candidate
            best_disc = cand_disc
        # Local search from this candidate
        candidate, cand_disc = _local_search(candidate, deadline, rng)
        if cand_disc < best_disc:
            best_signs = candidate
            best_disc = cand_disc
        # Break if we've found a very good solution
        if best_disc <= 1:
            break
    
    return best_signs
# EVOLVE-BLOCK-END
