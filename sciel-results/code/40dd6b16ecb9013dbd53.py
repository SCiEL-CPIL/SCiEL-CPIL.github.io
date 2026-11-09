"""Rewritten program: structural change to multi-start greedy + local search with deadline guard."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(S, time_budget):
        """
        Solve one instance with multiple greedy starts and local search.
        Uses an array-based membership check for speed.
        """
        best = None
        best_size = 0
        n = len(S)
        
        # Baselines
        odds = [x for x in S if x % 2 != 0]
        best_size = max(best_size, len(odds))
        best = odds
        
        # 2-adic valuation baseline
        groups = {}
        for x in S:
            v = 0
            temp = abs(x)
            while temp % 2 == 0:
                v += 1
                temp //= 2
            if temp % 2 == 1:
                groups.setdefault(v, []).append(x)
        for v, group in groups.items():
            if len(group) > best_size:
                best_size = len(group)
                best = group

        evens = [x for x in S if x % 2 == 0]
        if len(evens) > best_size:
            best = evens
            best_size = len(evens)
            
        positives = [x for x in S if x > 0]
        if len(positives) > best_size:
            best = positives
            best_size = len(positives)
            
        negatives = [x for x in S if x < 0]
        if len(negatives) > best_size:
            best = negatives
            best_size = len(negatives)
            
        # Array for fast membership
        # Range [-10000, 10000] -> size 20001
        OFFSET = 10000
        SIZE = 20001
        
        def can_add(x, subset_set, present):
            # Check if adding x to chosen_set (tracked by present array) creates violation
            # 1. a + b = x for some a, b in chosen_set
            #    Iterate a in chosen_set, check if x - a in present
            for a in subset_set:
                idx = x - a + OFFSET
                if 0 <= idx < SIZE and present[idx]:
                    return False
            # 2. x + a = b for some a, b in chosen_set
            #    Iterate a in chosen_set, check if x + a in present
            for a in subset_set:
                idx = x + a + OFFSET
                if 0 <= idx < SIZE and present[idx]:
                    return False
            return True

        def local_search_with_swaps(initial_subset, S_list, max_iters=5000, rng_seed=42):
            """
            Local search that allows adding and swapping elements.
            """
            rng = random.Random(rng_seed)
            subset = list(initial_subset)
            subset_set = set(subset)
            present = [0] * SIZE
            for x in subset:
                present[x + OFFSET] = 1
            
            # Candidates are those in S but not in subset
            candidates = [x for x in S_list if x not in subset_set]
            
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                
                # 1. Try to add
                rng.shuffle(candidates)
                added = False
                for x in candidates:
                    if can_add(x, subset_set, present):
                        subset.append(x)
                        subset_set.add(x)
                        present[x + OFFSET] = 1
                        # Remove from candidates
                        # Since we are iterating, we can't remove directly easily without breaking iterator
                        # But we can mark it or just break and restart
                        # Given N is small, breaking is fine
                        added = True
                        break 
                
                if added:
                    # Rebuild candidates to reflect the change
                    candidates = [x for x in S_list if x not in subset_set]
                    improved = True
                    continue

                # 2. Try to swap
                # Pick a random element from subset to remove
                if subset:
                    y_idx = rng.randrange(len(subset))
                    y = subset[y_idx]
                    
                    # Remove y
                    subset.pop(y_idx)
                    subset_set.remove(y)
                    present[y + OFFSET] = 0
                    
                    # Try to add a random element from candidates
                    swapped = False
                    if candidates:
                        # Try up to 10 random candidates
                        trial_count = min(10, len(candidates))
                        for _ in range(trial_count):
                            x_idx = rng.randrange(len(candidates))
                            x = candidates[x_idx]
                            if can_add(x, subset_set, present):
                                # Accept swap
                                subset.append(x)
                                subset_set.add(x)
                                present[x + OFFSET] = 1
                                # Remove x from candidates
                                candidates.pop(x_idx)
                                # Add y to candidates
                                candidates.append(y)
                                swapped = True
                                break
                            
                    if swapped:
                        improved = True
                    else:
                        # Revert
                        subset.append(y)
                        subset_set.add(y)
                        present[y + OFFSET] = 1

            return subset

        # Multi-start local search
        num_starts = 50
        for i in range(num_starts):
            if time.time() > start_time + time_budget:
                break
            
            # Random greedy init
            rng = random.Random(i)
            elements = list(S)
            rng.shuffle(elements)
            
            chosen = []
            chosen_set = set()
            present = [0] * SIZE
            
            for x in elements:
                if can_add(x, chosen, present):
                    chosen.append(x)
                    chosen_set.add(x)
                    present[x + OFFSET] = 1
            
            # Local search with swaps
            ls_result = local_search_with_swaps(chosen, S, max_iters=2000, rng_seed=i)
            
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best = ls_result
                
        return best if best else []

    # Generate instances
    instances = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        instances.append(instance(size, seed))

    results = []
    for i, S in enumerate(instances):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            # Not enough time, use quick heuristic
            results.append([x for x in S if x % 2 != 0])
            continue
        
        # Allocate time based on instance size
        if i == 0:
            budget = min(remaining * 0.3, 30)
        elif i == 1:
            budget = min(remaining * 0.3, 40)
        else:
            budget = min(remaining * 0.4, 50)
        
        result = solve_instance(S, budget)
        results.append(result)

    # Final validation
    for i, (S, result) in enumerate(zip(instances, results)):
        if not is_sum_free(result):
            # Fallback to odds
            results[i] = [x for x in S if x % 2 != 0]

    return results
