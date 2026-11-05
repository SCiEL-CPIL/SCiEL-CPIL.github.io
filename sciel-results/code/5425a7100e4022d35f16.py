import numpy as np
import time

def _compute_max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [0, n-1] with any start, positive step, and length >= 1.
    signs: numpy array of int8 with values -1 or +1, length n.
    Returns: max absolute sum (int).
    """
    n = len(signs)
    max_abs = 0
    # For each step s from 1 to n//2 (and also s up to n-1 but that gives only length 1 or 2)
    # Actually step can be up to n-1. For step s, the progression is a, a+s, a+2s, ...
    # where a is the start (0-indexed) and the indices must be < n.
    # For a fixed step s, the possible starts are 0, 1, ..., s-1 (since if start >= s, 
    # we could have started earlier). Wait, no. The start can be any index such that
    # there's at least one element. For step s, the residue classes mod s are independent.
    # For each residue r in [0, s-1], the progression is r, r+s, r+2s, ...
    # We need to check all prefixes of these progressions? No, the problem says "any ... length".
    # So for a fixed (start, step), we consider ALL lengths >= 1.
    # That means for a fixed (start, step), we need the max abs prefix sum of that progression.
    
    # Optimization: For step s, iterate over residues r=0..s-1.
    # For each residue, extract the subsequence signs[r], signs[r+s], signs[r+2s], ...
    # Compute prefix sums and find max abs.
    
    # But we can do better. Note that for large s, the progressions are short.
    # For s > n//2, the maximum length is 2 (indices r and r+s, where r < n-s).
    # Actually if s > n/2, then r+s < n implies r < n-s < s. So each residue class has at most 2 elements.
    # For s > n/2, the max abs sum is just max over all pairs (signs[i] + signs[i+s]) for i < n-s, and also singletons.
    # Singletons always give abs sum 1. So max_abs is at least 1.
    
    # Let's iterate s from 1 to n-1.
    # For efficiency, we can break early if s is large enough that no progression can beat current max_abs.
    # Max possible sum for step s is the length of the longest progression with that step.
    # Longest progression length for step s starting at r is floor((n-1-r)/s) + 1.
    # The max over r is floor((n-1)/s) + 1.
    # If floor((n-1)/s) + 1 <= max_abs, then no progression with step s can exceed max_abs.
    # So we can break when floor((n-1)/s) + 1 <= max_abs, i.e., s >= (n-1)/max_abs.
    
    for s in range(1, n):
        max_len = (n - 1) // s + 1
        if max_len <= max_abs:
            break
        for r in range(s):
            # Progression: r, r+s, r+2s, ...
            # Compute prefix sums
            cur = 0
            for idx in range(r, n, s):
                cur += signs[idx]
                a = abs(cur)
                if a > max_abs:
                    max_abs = a
    return max_abs

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # Start with alternating pattern
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs = signs.copy()
    best_score = _compute_max_abs_ap_sum(best_signs)
    
    # Local search: flip one sign at a time, keep if it improves
    # Use random order for flips
    rng = np.random.RandomState(42)
    idx_order = np.arange(n)
    
    improved = True
    while improved:
        improved = False
        rng.shuffle(idx_order)
        for i in idx_order:
            if time.time() > deadline:
                break
            # Try flipping sign at position i
            old_val = signs[i]
            signs[i] = -old_val
            new_score = _compute_max_abs_ap_sum(signs)
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_val
        if time.time() > deadline:
            break
    
    # If we still have time, try a few random restarts
    while time.time() < deadline - 5:
        # Generate a random starting point
        random_signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        cur_signs = random_signs.copy()
        cur_score = _compute_max_abs_ap_sum(cur_signs)
        
        # Local search from this point
        for _ in range(3):  # Max 3 passes
            if time.time() > deadline - 2:
                break
            idx_order = np.arange(n)
            rng.shuffle(idx_order)
            for i in idx_order:
                if time.time() > deadline - 2:
                    break
                old_val = cur_signs[i]
                cur_signs[i] = -old_val
                new_score = _compute_max_abs_ap_sum(cur_signs)
                if new_score < cur_score:
                    cur_score = new_score
                else:
                    cur_signs[i] = old_val
        
        if cur_score < best_score:
            best_score = cur_score
            best_signs = cur_signs.copy()
    
    return [int(x) for x in best_signs]
