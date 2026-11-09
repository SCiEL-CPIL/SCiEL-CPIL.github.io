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
            # For each length from 1 to max_len, the progression is the first `length` terms
            # But we can just store the full list of indices for this (start, step) and
            # consider all prefixes? No, the task says "any start, positive step and length".
            # So each (start, step, length) is a distinct progression.
            # However, note that for a fixed (start, step), the progressions of different lengths
            # are nested. But the sum over a prefix is what matters.
            # Actually, the sum over the progression of length L is sum of signs at positions
            # start, start+step, ..., start+(L-1)*step.
            # So we can just iterate over all (start, step, length) triples.
            # But that's O(n^3) which is too many. Let's count:
            # sum over step=1 to n-1 of (sum over start=0 to step-1 of ((n-1-start)//step + 1))
            # This is roughly n^2 / 2 * harmonic, which is about n^2 log n ~ 256^2 * 6 ~ 400k.
            # That's manageable if we do it smartly.
            # But actually, we can just store the indices for each (start, step) and then
            # for each prefix length, compute the sum. But that's still a lot.
            # Alternative: for each (start, step), we have a list of positions.
            # We can compute the prefix sums of the signs at those positions.
            # Then the max absolute sum for this (start, step) is the max over all prefix sums.
            # So we only need to store the list of positions for each (start, step).
            positions = list(range(start, n, step))
            if len(positions) > 0:
                progressions.append(positions)

    # Convert to numpy array for faster sum computation
    # progressions is a list of lists of ints
    # We'll create a 2D array? No, variable length.
    # Instead, we'll just keep the list of lists.

    def compute_cost(signs):
        """Compute max absolute sum over all arithmetic progressions."""
        signs_arr = np.array(signs)
        max_abs = 0
        for pos in progressions:
            # Compute sum of signs at these positions
            s = signs_arr[pos].sum()
            abs_s = abs(s)
            if abs_s > max_abs:
                max_abs = abs_s
        return int(max_abs)

    def local_search(signs, current_cost, time_limit):
        """Simple local search: flip one sign at a time if it improves cost."""
        signs = signs[:]
        current_max = current_cost
        n_local = n
        rng_local = random.Random(123)
        
        while time.time() < time_limit:
            improved = False
            # Shuffle positions to avoid bias
            positions = list(range(n_local))
            rng_local.shuffle(positions)
            
            for pos in positions:
                if time.time() >= time_limit:
                    break
                old_val = signs[pos]
                signs[pos] = -old_val
                new_cost = compute_cost(signs)
                if new_cost < current_max:
                    current_max = new_cost
                    improved = True
                else:
                    signs[pos] = old_val
            if not improved:
                break
        return signs, current_max

    # Initial random signs
    rng = random.Random(99)
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_cost = compute_cost(best_signs)
    
    # Local search on initial
    phase1_end = min(t0 + 10, deadline - 5)
    best_signs, best_cost = local_search(best_signs, best_cost, phase1_end)
    
    # Random restarts
    while time.time() < deadline - 3:
        cand = [rng.choice((-1, 1)) for _ in range(n)]
        cand_cost = compute_cost(cand)
        if cand_cost < best_cost:
            best_signs = cand[:]
            best_cost = cand_cost
        
        # Brief local search on candidate
        restart_end = min(time.time() + 2.0, deadline - 3)
        if time.time() < restart_end:
            cand, cand_cost = local_search(cand, cand_cost, restart_end)
            if cand_cost < best_cost:
                best_signs = cand[:]
                best_cost = cand_cost
    
    return best_signs
# EVOLVE-BLOCK-END
