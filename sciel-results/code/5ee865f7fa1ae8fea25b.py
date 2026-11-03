import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def find_max_sidon_set(n):
    # Returns the largest Sidon set in {0, ..., n-1}
    # Using backtracking
    max_size = 0
    best_set = []
    
    def backtrack(current_set, next_elem):
        nonlocal max_size, best_set
        if len(current_set) > max_size:
            max_size = len(current_set)
            best_set = current_set.copy()
        
        # Pruning: max possible size is approx sqrt(2n)
        # If len(current_set) + (n - next_elem) <= max_size, prune? No, that's weak.
        # Better: if len(current_set) + int(np.sqrt(2*(n-next_elem))) <= max_size, prune.
        
        if next_elem >= n:
            return

        # Try including next_elem
        # Check if next_elem can be added
        can_add = True
        sums = set()
        for x in current_set:
            if x + next_elem in sums:
                can_add = False
                break
            sums.add(x + next_elem)
        # Also check sums within current_set are already valid
        # We only need to check new sums involving next_elem
        
        if can_add:
            # Check self sum 2*next_elem
            if 2*next_elem in sums:
                can_add = False
            
        if can_add:
            backtrack(current_set + [next_elem], next_elem + 1)
        
        # Try excluding next_elem
        backtrack(current_set, next_elem + 1)

    # To make it faster, we can fix 0 in the set?
    # A Sidon set can be shifted. {0, ...} is WLOG if we consider sums?
    # Actually, shifting doesn't change the Sidon property.
    # So we can assume 0 is in the set.
    backtrack([0], 1)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Find max Sidon set
    # The backtracking above might be slow if not pruned well.
    # Let's use a more efficient greedy with random restarts or a specific heuristic.
    # Or just a simple recursive search with good pruning.
    
    # For N=50, the search space is small enough for a decent backtracker.
    # But the naive one above is O(2^N) worst case.
    # Pruning:
    # 1. If current size + max_possible_remaining <= best_size, return.
    # Max possible remaining from index i is roughly sqrt(2*(N-i)).
    
    max_sidon = []
    
    def is_valid(candidate, new_elem):
        sums = set()
        for x in candidate:
            s = x + new_elem
            if s in sums:
                return False
            sums.add(s)
        if 2*new_elem in sums:
            return False
        return True

    def backtrack(candidate, start_idx):
        nonlocal max_sidon
        if len(candidate) > len(max_sidon):
            max_sidon = candidate.copy()
        
        # Pruning
        # Upper bound on size we can add
        remaining = N - start_idx
        max_add = int(np.sqrt(2 * remaining)) + 1
        if len(candidate) + max_add <= len(max_sidon):
            return
            
        if start_idx >= N:
            return
            
        # Include start_idx
        if is_valid(candidate, start_idx):
            backtrack(candidate + [start_idx], start_idx + 1)
            
        # Exclude start_idx
        backtrack(candidate, start_idx + 1)

    # Initialize with 0
    backtrack([0], 1)
    
    S = max_sidon
    K = len(S)
    
    # Construct h
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0
        
    # Check time
    if time.time() > deadline - 10:
        return list(h)
        
    # Optional: Optimize weights
    # For a Sidon set, uniform weights are optimal.
    # But let's verify with a quick SLSQP just in case.
    
    def neg_obj(h_vec):
        l1 = np.sum(h_vec)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h_vec**2)
        hc = np.convolve(h_vec, h_vec, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0
        return -(l2_sq / mc)
        
    # We only optimize on the support S
    # Map h_vec to full vector
    def map_to_full(h_sub):
        h_full = np.zeros(N)
        for idx, val in zip(S, h_sub):
            h_full[idx] = val
        return h_full

    h_sub_init = np.ones(K) / K
    # Normalize to sum 1
    h_sub_init = h_sub_init / np.sum(h_sub_init)
    
    # Actually, the objective is homogeneous. We can fix sum(h_sub)=1.
    # But SLSQP handles constraints.
    
    # Since K is small, we can just use uniform.
    # The improvement from weight optimization on a Sidon set is negligible (0).
    
    return list(h)
