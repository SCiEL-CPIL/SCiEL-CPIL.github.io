N = 256

import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Leave 5s buffer for safety

    def greedy_randomized():
        # Shuffle candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            # Check if x can be added
            added_sums = [x + a for a in selected] + [2 * x]
            # Optimization: check intersection first
            if sums.intersection(added_sums):
                continue
            if len(set(added_sums)) != len(added_sums):
                continue
            
            selected.append(x)
            sums.update(added_sums)
            
        return selected

    def is_sidon(s):
        if not s:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def is_maximal(s):
        if not is_sidon(s):
            return False
        s_set = set(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if adding x preserves Sidon property
            new_sums = [x + a for a in s] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if sums.intersection(new_sums):
                continue
            # x can be added, so not maximal
            return False
        return True

    def local_search_remove(s):
        # Try to remove elements while maintaining maximality
        # This is tricky: removing an element may break maximality
        # We only remove if the set remains maximal after removal
        # But removing an element makes it easier to be non-maximal
        # So this is not straightforward. Instead, we try swaps.
        pass

    def swap_search(s, max_iters=100):
        # Try swapping elements to reduce size or maintain size with better structure
        # This is complex for Sidon sets. Let's stick to greedy restarts with different strategies.
        return s

    best_set = None
    best_size = float('inf')
    
    # Run several restarts with different random seeds implicitly via shuffle
    while time.time() < deadline:
        current_set = greedy_randomized()
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Fallback to deterministic if time ran out or no result (unlikely)
    if best_set is None:
        # Deterministic ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected

    return best_set
# EVOLVE-BLOCK-END
