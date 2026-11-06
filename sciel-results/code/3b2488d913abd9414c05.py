"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N  # Maximum possible sum
    
    def compute_sum_counts(chosen_set):
        """Compute sum counts using FFT convolution. Returns dict of sum -> count."""
        if not chosen_set:
            return {}
        # Create boolean vector
        vec = np.zeros(MAX_SUM + 1, dtype=np.float64)
        for x in chosen_set:
            vec[x] = 1.0
        # Convolution gives counts of pairs (a,b) with a+b=s
        # But we need a<=b, so we need to be careful
        # FFT convolution: (vec * vec)[s] = sum over a+b=s of vec[a]*vec[b]
        # This counts ordered pairs. For a<b, each pair counted twice.
        # For a==b, counted once.
        # We want: for a<b, count once; for a==b, count once.
        # So: total_ordered = conv[s]; diagonal = 1 if s is even and s/2 in set
        # count_a_le_b = (total_ordered + diagonal) / 2
        conv = np.convolve(vec, vec, mode='full')
        counts = {}
        for s in range(2, MAX_SUM + 1):
            if conv[s] > 0.5:
                ordered = int(round(conv[s]))
                diag = 1 if (s % 2 == 0 and vec[s // 2] > 0.5) else 0
                count = (ordered + diag) // 2
                if count > 0:
                    counts[s] = count
        return counts
    
    def is_valid_add(chosen_set, counts, x):
        """Check if x can be added to chosen. O(k) check."""
        for y in chosen_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True
    
    def get_tight_count(chosen_set, counts, x):
        """Get total tightness for adding x. Returns (tightness, num_new_sums)."""
        tightness = 0
        for y in chosen_set:
            s = x + y
            tightness += counts.get(s, 0)
        tightness += counts.get(2 * x, 0)
        num_new_sums = len(chosen_set) + 1
        return tightness, num_new_sums
    
    def greedy(order, chosen_set, counts):
        """Greedy construction given an order."""
        for x in order:
            if x not in chosen_set and is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        return chosen_set, counts
    
    def greedy_with_spectral_heuristic(order, chosen_set, counts):
        """Greedy with spectral entropy preference."""
        for x in order:
            if x in chosen_set:
                continue
            if not is_valid_add(chosen_set, counts, x):
                continue
            # Compute how "spread out" the new sums would be
            # Prefer additions that create sums with lower existing counts
            tightness, num_new_sums = get_tight_count(chosen_set, counts, x)
            
            # Accept if not too tight (allow some flexibility)
            # Use a threshold: accept if average tightness per new sum is < 1.5
            avg_tight = tightness / num_new_sums if num_new_sums > 0 else 0
            
            if avg_tight <= 1.5:
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        return chosen_set, counts
    
    def local_search(chosen_set, counts, max_iter=500):
        """Local search: try swaps to improve size."""
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not chosen_set:
                break
            
            # Pick a random element to potentially remove
            old_x = random.choice(list(chosen_set))
            
            # Temporarily remove it
            for y in chosen_set:
                if y == old_x:
                    continue
                s = old_x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            counts[2 * old_x] -= 1
            if counts[2 * old_x] == 0:
                del counts[2 * old_x]
            chosen_set.discard(old_x)
            
            # Try to add a different element
            found = False
            candidates = [x for x in range(1, N + 1) if x not in chosen_set]
            random.shuffle(candidates)
            
            for x in candidates[:300]:
                if is_valid_add(chosen_set, counts, x):
                    for y in chosen_set:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    chosen_set.add(x)
                    found = True
                    break
            
            if not found:
                # Put old_x back
                for y in chosen_set:
                    s = old_x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * old_x] = counts.get(2 * old_x, 0) + 1
                chosen_set.add(old_x)
        
        return chosen_set, counts
    
    def try_add_remaining(chosen_set, counts):
        """Try to add any remaining elements that can fit."""
        added = True
        while added and time.time() < deadline - 3:
            added = False
            for x in range(1, N + 1):
                if x not in chosen_set and is_valid_add(chosen_set, counts, x):
                    for y in chosen_set:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    chosen_set.add(x)
                    added = True
                    break
        return chosen_set, counts
    
    # Phase 1: Multiple greedy strategies
    best_chosen = set()
    best_size = 0
    
    # Forward greedy
    chosen_set = set()
    counts = {}
    chosen_set, counts = greedy(range(1, N + 1), chosen_set, counts)
    best_chosen = chosen_set.copy()
    best_size = len(chosen_set)
    
    # Backward greedy
    if time.time() < deadline - 20:
        chosen_set = set()
        counts = {}
        chosen_set, counts = greedy(range(N, 0, -1), chosen_set, counts)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Spectral heuristic greedy (forward)
    if time.time() < deadline - 20:
        chosen_set = set()
        counts = {}
        chosen_set, counts = greedy_with_spectral_heuristic(range(1, N + 1), chosen_set, counts)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Spectral heuristic greedy (backward)
    if time.time() < deadline - 20:
        chosen_set = set()
        counts = {}
        chosen_set, counts = greedy_with_spectral_heuristic(range(N, 0, -1), chosen_set, counts)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Random greedy passes
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        chosen_set = set()
        counts = {}
        chosen_set, counts = greedy(order, chosen_set, counts)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Random spectral heuristic passes
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        chosen_set = set()
        counts = {}
        chosen_set, counts = greedy_with_spectral_heuristic(order, chosen_set, counts)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Phase 2: Try to add more elements to the best solution
    chosen_set = best_chosen.copy()
    counts = compute_sum_counts(chosen_set)
    chosen_set, counts = try_add_remaining(chosen_set, counts)
    
    if len(chosen_set) > best_size:
        best_size = len(chosen_set)
        best_chosen = chosen_set.copy()
    
    # Phase 3: Local search - try to improve by swapping
    if time.time() < deadline - 10:
        counts = compute_sum_counts(best_chosen)
        best_chosen, counts = local_search(best_chosen, counts, max_iter=1500)
        
        if len(best_chosen) > best_size:
            best_size = len(best_chosen)
    
    # Phase 4: Final attempt to add remaining (multiple passes)
    if time.time() < deadline - 5:
        counts = compute_sum_counts(best_chosen)
        best_chosen, counts = try_add_remaining(best_chosen, counts)
        
        # One more pass after local search
        if time.time() < deadline - 3:
            counts = compute_sum_counts(best_chosen)
            best_chosen, counts = try_add_remaining(best_chosen, counts)
    
    return list(best_chosen)
# EVOLVE-BLOCK-END
