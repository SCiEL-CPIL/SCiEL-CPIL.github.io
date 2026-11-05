"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import random

def construct():
    n = 128
    target_low = 128
    target_high = 256
    
    # Precompute indices for each sum s
    # For s in [128, 256], valid a are in [max(0, s-128), min(128, s//2)]
    # We store (a_min, a_max, s) for each s
    sum_params = []
    for s in range(target_low, target_high + 1):
        a_min = max(0, s - 128)
        a_max = min(128, s // 2)
        sum_params.append((a_min, a_max, s))
        
    def compute_counts(x_binary):
        """
        x_binary: numpy array of 0s and 1s of length 129
        Returns: numpy array of counts for s in [128, 256]
        """
        counts = np.zeros(target_high - target_low + 1, dtype=np.int32)
        for i, (a_min, a_max, s) in enumerate(sum_params):
            # v1 = x[a_min : a_max+1]
            # v2 = x[s-a_max : s-a_min+1] reversed
            # Note: s-a_min >= s-a_max
            # Indices in x for v2: from s-a_max to s-a_min
            # We need x[s-a_min] to correspond to x[a_min]
            # So v2_rev = x[s-a_min : s-a_max-1 : -1]
            v1 = x_binary[a_min:a_max+1]
            v2 = x_binary[s-a_min:s-a_max-1:-1]
            counts[i] = np.dot(v1, v2)
        return counts

    def objective(x):
        # x is continuous [0, 1]
        x_binary = (x > 0.5).astype(np.int32)
        counts = compute_counts(x_binary)
        
        # Check coverage
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            # Heavy penalty for not covering
            return 10000 + uncovered
            
        max_count = np.max(counts)
        return max_count

    # Differential Evolution
    bounds = [(0, 1)] * 129
    
    # Use a smaller population and fewer iterations to fit in time
    # 129 dimensions is high for DE, so we need to be careful with popsize
    # popsize=15, maxiter=100 -> ~1500 evaluations
    # Each evaluation is ~129 dot products of length ~60 -> very fast
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6, 
            polish=False # We will do our own polishing
        )
        best_x = result.x
    except Exception:
        # Fallback to a simple heuristic if DE fails or times out
        # A = [64, 128] covers [128, 256]?
        # No, [64, 128] is a set.
        # Let's use the seed from the original program as fallback
        best_x = np.zeros(129)
        best_x[64:129] = 1.0

    # Convert to binary set
    A_set = set(np.where(best_x > 0.5)[0])
    
    # Local Search Polish
    # 1. Try to remove elements that are not critical for coverage and reduce max count
    # 2. Try to add elements to reduce max count if possible (though adding increases counts usually)
    # Actually, removing elements is the primary way to reduce max count if coverage allows.
    
    def get_stats(A_set):
        x_binary = np.zeros(129, dtype=np.int32)
        for a in A_set:
            x_binary[a] = 1
        counts = compute_counts(x_binary)
        max_count = np.max(counts)
        uncovered = np.sum(counts == 0)
        return max_count, uncovered

    current_max, current_uncovered = get_stats(A_set)
    
    # If DE failed to cover, try to fix coverage first with a greedy add
    if current_uncovered > 0:
        # Greedy add to cover
        while current_uncovered > 0 and time.time() < 55:
            # Find uncovered sums
            x_binary = np.zeros(129, dtype=np.int32)
            for a in A_set:
                x_binary[a] = 1
            counts = compute_counts(x_binary)
            uncovered_sums = [target_low + i for i in range(len(counts)) if counts[i] == 0]
            if not uncovered_sums:
                break
            
            # Try adding a random element
            candidates = [i for i in range(129) if i not in A_set]
            if not candidates:
                break
            added = False
            for _ in range(100): # Try a few random adds
                cand = random.choice(candidates)
                new_A = A_set | {cand}
                new_max, new_unc = get_stats(new_A)
                if new_unc < current_uncovered:
                    A_set = new_A
                    current_max = new_max
                    current_uncovered = new_unc
                    added = True
                    break
            if not added:
                # Stuck, break to avoid infinite loop
                break

    # Now minimize max count by removing elements
    start_time = time.time()
    time_limit = 55
    
    while time.time() - start_time < time_limit:
        # Try removing a random element
        if len(A_set) <= 1:
            break
        elem = random.choice(list(A_set))
        new_A = A_set - {elem}
        new_max, new_uncovered = get_stats(new_A)
        
        if new_uncovered == 0 and new_max <= current_max:
            A_set = new_A
            current_max = new_max
        else:
            # Try swapping? Or just continue
            pass
            
        # Occasionally try to add an element to see if it allows further removals?
        # Or just stick to removal for simplicity and speed.
        
    return list(A_set)
# EVOLVE-BLOCK-END
