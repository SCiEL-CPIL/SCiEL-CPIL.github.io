import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer for 60s limit
    
    best_set = []
    best_size = 0
    
    def is_valid(set_list):
        """Check if the set has unique triple sums."""
        sums = set()
        n = len(set_list)
        # Use indices to avoid sorting overhead if already sorted? 
        # The set_list might not be sorted. 
        # We need a <= b <= c.
        # It's faster to sort first.
        sorted_list = sorted(set_list)
        for i in range(n):
            a = sorted_list[i]
            for j in range(i, n):
                b = sorted_list[j]
                for k in range(j, n):
                    c = sorted_list[k]
                    s = a + b + c
                    if s in sums:
                        return False
                    sums.add(s)
        return True

    def run_greedy(order):
        selected = []
        current_sums = set()
        
        for x in order:
            # Generate new sums involving x
            new_sums = set()
            
            # x + x + x
            s = 3 * x
            if s in current_sums:
                continue
            new_sums.add(s)
            
            # x + x + a
            conflict = False
            for a in selected:
                s = 2 * x + a
                if s in current_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.add(s)
            if conflict:
                continue
                
            # x + a + b
            for a, b in combinations_with_replacement(selected, 2):
                s = x + a + b
                if s in current_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.add(s)
            
            if not conflict:
                selected.append(x)
                current_sums.update(new_sums)
                
        return selected

    def optimize_set(initial_set):
        """Hill climbing to improve the set size."""
        current_set = list(initial_set)
        current_size = len(current_set)
        no_improve_count = 0
        max_no_improve = 50
        
        # Pre-compute sums for current set to speed up checks?
        # Since we are swapping, we need to recalculate.
        # For K=40, recalc is ~10k ops. 
        # We can do ~1000 swaps/sec easily.
        
        while time.time() - start_time < time_limit:
            if no_improve_count > max_no_improve:
                break
                
            improved = False
            
            # 1. Try to add a random element
            # Sample candidates from 1..4096
            # To be efficient, we can try a few random ones
            for _ in range(20):
                y = random.randint(1, 4096)
                if y in current_set:
                    continue
                # Check if adding y is valid
                # We can do a fast check: generate new sums with y and check against existing sums
                # But we don't have existing sums stored efficiently in this scope.
                # Let's just use is_valid for simplicity first, or reconstruct sums.
                
                # Optimization: Store sums in a set for current_set
                # This is expensive to maintain if we swap.
                # Let's stick to is_valid for now, but limit attempts.
                
                candidate_set = current_set + [y]
                if is_valid(candidate_set):
                    current_set = candidate_set
                    current_size += 1
                    improved = True
                    no_improve_count = 0
                    break
                    
            if improved:
                continue
                
            # 2. Try to swap an element
            # Pick a random element to remove
            idx = random.randint(0, len(current_set) - 1)
            x = current_set[idx]
            
            # Try to replace x with a random y
            for _ in range(20):
                y = random.randint(1, 4096)
                if y == x or y in current_set:
                    continue
                new_set = current_set.copy()
                new_set[idx] = y
                if is_valid(new_set):
                    current_set = new_set
                    no_improve_count = 0
                    improved = True
                    break
                    
            if not improved:
                no_improve_count += 1
                
        return current_set

    # 1. Deterministic greedy from 1..4096
    baseline = run_greedy(range(1, 4097))
    best_set = baseline
    best_size = len(baseline)
    
    # 2. Randomized restarts with optimization
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Shuffle the candidate range
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        current_set = run_greedy(candidates)
        
        # Optimize the greedy result
        optimized_set = optimize_set(current_set)
        
        if len(optimized_set) > best_size:
            best_size = len(optimized_set)
            best_set = optimized_set
            
        # If we found a very large set, we might stop early?
        # Theoretical max is ~41. If we hit 41, we can stop.
        if best_size >= 41:
            break
            
    return best_set
