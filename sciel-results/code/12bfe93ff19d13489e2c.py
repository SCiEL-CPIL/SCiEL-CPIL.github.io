"""Improved program for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 50  # safety margin before 60s kill

    # Precompute all arithmetic progressions as index lists
    # An arithmetic progression is (start, step, length) with start in [0, n-1],
    # step >= 1, and start + (len-1)*step < n
    progressions = []
    for step in range(1, n):
        for start in range(step):
            # length: start + (k-1)*step < n  =>  k <= (n - 1 - start) // step + 1
            max_len = (n - 1 - start) // step + 1
            if max_len < 1:
                continue
            # For each length from 1 to max_len, the progression is the first 'length' elements
            # Actually, we need ALL contiguous sub-progressions? No, the task says "any arithmetic progression contained in [1,256]"
            # This means any sequence a, a+d, a+2d, ..., a+(k-1)d where all terms are in [1,256].
            # So for a given (start, step), the possible lengths are 1 to max_len.
            # But wait, "contained in [1,256]" just means the indices are valid.
            # So for each (start, step), we have progressions of length 1, 2, ..., max_len.
            # However, a progression of length L is a subset of a progression of length L+1 (if we extend it).
            # But the sum is different. We need to check ALL of them.
            # Total number of progressions:
            # Sum over step of (sum over start of max_len)
            # This is O(n^2 log n) roughly?
            # Let's estimate: for step=1, sum is n(n+1)/2 ~ 32k. For step=2, ~16k. Sum is ~ n^2/2 * H_n ~ 32k * 6 ~ 190k?
            # Actually, number of APs is sum_{d=1}^{n-1} sum_{a=0}^{d-1} floor((n-1-a)/d) + 1
            # This is approximately n^2/2. For n=256, ~32k * something.
            # Let's just generate them. It might be ~ 100k-200k progressions.
            # Storing 200k lists of average length ~2 is fine.
            # But computing the sum for 200k progressions on every flip is expensive.
            # 200k * 256 flips * 2 ops = 100M ops per flip? No, 200k progressions * (length) ops.
            # Average length is small.
            # Let's try to be smarter.
            # We can precompute for each index i, which progressions it belongs to.
            # Then when we flip index i, we only update the sums of those progressions.
            # Number of progressions containing a fixed index i:
            # For each step d, and each start a such that a <= i < a + k*d, i is in the progression.
            # This is still potentially large but much less than total progressions.
            
            # Let's try a different approach: local search with direct evaluation.
            # But 200k progressions is too many to evaluate naively for each flip.
            # We need a faster way to compute the max absolute sum.
            
            # Alternative: Use a heuristic that focuses on the most "constrained" progressions.
            # Or, use a simulated annealing approach with a smarter cost function.
            
            # Let's first count how many progressions there are.
            pass

    # Let's count progressions to see if it's feasible
    count = 0
    for step in range(1, n):
        for start in range(step):
            max_len = (n - 1 - start) // step + 1
            count += max_len
    # print(f"Total progressions: {count}")  # ~ 100k-200k
    
    # This is too many to check all for each flip.
    # We need a different strategy.
    
    # Strategy: Use a greedy approach or a smarter local search.
    # Idea: Start with a random assignment, then find the worst progression and try to fix it.
    # Or, use a simulated annealing where the cost is the max over a random subset of progressions.
    
    # Let's try: 
    # 1. Generate all progressions.
    # 2. For each progression, compute its sum.
    # 3. Find the max absolute sum.
    # 4. Pick a random index from the worst progression and flip it.
    # 5. Update only the progressions that contain that index.
    
    # To make this efficient, we need to know which progressions contain each index.
    # Let's build that mapping.
    
    # progressions_by_index: list of lists, where progressions_by_index[i] is a list of (prog_id, sign)
    # sign is +1 if i is in the progression (which it always is, so sign is always +1)
    # Actually, we just need the list of progression IDs that contain index i.
    
    progressions = []  # list of (indices, sum)
    prog_by_index = [[] for _ in range(n)]
    
    pid = 0
    for step in range(1, n):
        for start in range(step):
            max_len = (n - 1 - start) // step + 1
            for length in range(1, max_len + 1):
                indices = tuple(start + k * step for k in range(length))
                progressions.append(indices)
                for idx in indices:
                    prog_by_index[idx].append(pid)
                pid += 1
    
    num_progs = len(progressions)
    # print(f"Number of progressions: {num_progs}")
    
    # Initialize signs
    rng = random.Random(42)
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    
    # Compute initial sums
    sums = np.zeros(num_progs, dtype=np.int16)
    for pid, indices in enumerate(progressions):
        s = 0
        for idx in indices:
            s += signs[idx]
        sums[pid] = s
    
    max_abs = np.max(np.abs(sums))
    
    # Local search
    best_signs = signs.copy()
    best_max_abs = max_abs
    
    # Precompute for each index, the list of progression IDs
    # This is already in prog_by_index
    
    # For speed, use numpy where possible, but the update is sparse.
    
    t_start = time.time()
    iterations = 0
    
    while time.time() - t_start < deadline - t0 - 5:  # leave 5s margin
        iterations += 1
        
        # Find the worst progression
        worst_pid = np.argmax(np.abs(sums))
        worst_sum = sums[worst_pid]
        
        # Pick a random index from the worst progression
        worst_indices = progressions[worst_pid]
        if len(worst_indices) == 0:
            continue
        flip_idx = rng.choice(worst_indices)
        
        # Try flipping flip_idx
        old_sign = signs[flip_idx]
        signs[flip_idx] = -old_sign
        
        # Update sums for all progressions containing flip_idx
        for p in prog_by_index[flip_idx]:
            sums[p] += 2 * (-old_sign)  # change is -2*old_sign
        
        new_max_abs = np.max(np.abs(sums))
        
        if new_max_abs < best_max_abs:
            best_max_abs = new_max_abs
            best_signs = signs.copy()
        else:
            # Revert
            signs[flip_idx] = old_sign
            for p in prog_by_index[flip_idx]:
                sums[p] -= 2 * (-old_sign)
        
        # Occasionally, do a random flip to escape local minima
        if iterations % 1000 == 0:
            flip_idx = rng.randint(0, n - 1)
            old_sign = signs[flip_idx]
            signs[flip_idx] = -old_sign
            for p in prog_by_index[flip_idx]:
                sums[p] += 2 * (-old_sign)
            new_max_abs = np.max(np.abs(sums))
            if new_max_abs < best_max_abs:
                best_max_abs = new_max_abs
                best_signs = signs.copy()
            else:
                signs[flip_idx] = old_sign
                for p in prog_by_index[flip_idx]:
                    sums[p] -= 2 * (-old_sign)
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
