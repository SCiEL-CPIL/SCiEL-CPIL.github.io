"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin
    
    def make_bitset():
        # bit i set means sum i is reachable, for i in 0..200
        return np.zeros(201, dtype=bool)
    
    def add_to_bitset(bs, x):
        # returns new bitset after adding x, or None if 200 would be reached
        if x >= 201:
            return None
        new_sums = np.zeros(201, dtype=bool)
        new_sums[x:] = bs[:201-x]
        if new_sums[200]:
            return None
        return bs | new_sums
    
    # Greedy: try in given order
    def greedy(order):
        selected = []
        bs = make_bitset()
        bs[0] = True
        for x in order:
            new_bs = add_to_bitset(bs, x)
            if new_bs is not None:
                selected.append(x)
                bs = new_bs
        return selected
    
    # Start-full-then-remove approach: find minimal removals
    def start_full_remove():
        # Start with all 100 elements, find which to remove to avoid sum=200
        # Use greedy removal: repeatedly find a subset summing to 200, remove one element from it
        selected = list(range(1, 101))
        sel_set = set(selected)
        # Check if current set has a subset summing to 200
        def has_sum_200(lst):
            bs = make_bitset()
            bs[0] = True
            for x in lst:
                if x >= 201:
                    continue
                new_sums = np.zeros(201, dtype=bool)
                new_sums[x:] = bs[:201-x]
                bs = bs | new_sums
                if bs[200]:
                    return True
            return bool(bs[200])
        
        def find_subset_summing_to_200(lst):
            # Find one subset of lst that sums to 200
            # Use DP to find which elements
            n = len(lst)
            # dp[s] = set of indices used, but that's expensive
            # Instead: track parent for each sum
            parent = [-1] * 201  # parent[s] = (prev_sum, index)
            reachable = [False] * 201
            reachable[0] = True
            for i, x in enumerate(lst):
                if x >= 201:
                    continue
                for s in range(200 - x, -1, -1):
                    if reachable[s] and not reachable[s + x]:
                        reachable[s + x] = True
                        parent[s + x] = (s, i)
                if reachable[200]:
                    break
            if not reachable[200]:
                return None
            # Reconstruct
            subset = []
            s = 200
            while s > 0:
                if s == 0:
                    break
                p = parent[s]
                if p is None:
                    return None
                prev_s, idx = p
                subset.append(lst[idx])
                s = prev_s
            return subset
        
        # Greedy removal
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            subset = find_subset_summing_to_200(selected)
            if subset is None:
                break
            # Remove the largest element from the subset (heuristic: larger elements are more "valuable" to keep? or remove smallest to preserve flexibility?)
            # Try removing the element that appears in the most other dangerous subsets
            # For now, remove the largest
            to_remove = max(subset)
            selected.remove(to_remove)
        
        return selected
    
    # Try largest-first greedy
    best = greedy(range(100, 0, -1))
    best_size = len(best)
    
    # Try smallest-first
    if time.time() < deadline:
        candidate = greedy(range(1, 101))
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try start-full-remove
    if time.time() < deadline - 2.0:
        candidate = start_full_remove()
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try various structured orderings
    if time.time() < deadline - 2.0:
        # Alternating large/small
        order = []
        for i in range(50):
            order.append(100 - i)
            order.append(i + 1)
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Multiples of 3 last
        order = [x for x in range(1, 101) if x % 3 != 0] + [x for x in range(1, 101) if x % 3 == 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Even first
        order = [x for x in range(1, 101) if x % 2 == 0] + [x for x in range(1, 101) if x % 2 != 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Multiples of 5 last
        order = [x for x in range(1, 101) if x % 5 != 0] + [x for x in range(1, 101) if x % 5 == 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Primes first (approximate)
        primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
        non_primes = [x for x in range(1, 101) if x not in primes]
        order = primes + non_primes
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try random orderings with different seeds
    for seed in [42, 123, 456, 789, 1011, 2023, 3141, 2718, 1618, 9999]:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        for trial in range(30):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, 101))
            rng.shuffle(order)
            candidate = greedy(order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(best)
    
    # Local search: try to add elements not in best
    for pass_num in range(5):
        if time.time() < deadline - 5.0:
            best_set = set(best)
            bs = make_bitset()
            bs[0] = True
            for x in best:
                bs = add_to_bitset(bs, x)
            added_any = False
            for x in range(1, 101):
                if x not in best_set:
                    new_bs = add_to_bitset(bs, x)
                    if new_bs is not None:
                        best.append(x)
                        best_set.add(x)
                        bs = new_bs
                        best_size = len(best)
                        added_any = True
            if not added_any:
                break
    
    if time.time() > deadline:
        return best
    
    # Swap-based local search: remove one element, try to add more
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        best_set = set(best)
        improved = False
        for remove_x in list(best):
            remaining = [x for x in best if x != remove_x]
            bs2 = make_bitset()
            bs2[0] = True
            valid = True
            for x in remaining:
                new_bs2 = add_to_bitset(bs2, x)
                if new_bs2 is None:
                    valid = False
                    break
                bs2 = new_bs2
            if not valid:
                continue
            new_best = list(remaining)
            new_set = set(remaining)
            for x in range(1, 101):
                if x not in new_set:
                    new_bs2 = add_to_bitset(bs2, x)
                    if new_bs2 is not None:
                        new_best.append(x)
                        new_set.add(x)
                        bs2 = new_bs2
            if len(new_best) > best_size:
                best = new_best
                best_size = len(best)
                improved = True
                break
        if not improved:
            break
    
    # Double-swap: remove two elements, try to add more
    if time.time() < deadline - 10.0:
        for i in range(len(best)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, len(best)):
                if time.time() > deadline - 5.0:
                    break
                remove_x = best[i]
                remove_y = best[j]
                remaining = [x for x in best if x != remove_x and x != remove_y]
                bs2 = make_bitset()
                bs2[0] = True
                valid = True
                for x in remaining:
                    new_bs2 = add_to_bitset(bs2, x)
                    if new_bs2 is None:
                        valid = False
                        break
                    bs2 = new_bs2
                if not valid:
                    continue
                new_best = list(remaining)
                new_set = set(remaining)
                for x in range(1, 101):
                    if x not in new_set:
                        new_bs2 = add_to_bitset(bs2, x)
                        if new_bs2 is not None:
                            new_best.append(x)
                            new_set.add(x)
                            bs2 = new_bs2
                if len(new_best) > best_size:
                    best = new_best
                    best_size = len(best)
                    break
            if len(best) != best_size or time.time() > deadline - 5.0:
                break
    
    return best
# EVOLVE-BLOCK-END
