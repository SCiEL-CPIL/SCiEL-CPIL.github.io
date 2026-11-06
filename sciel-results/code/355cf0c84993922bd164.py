import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_one(values, time_budget=15.0):
        n = len(values)
        if n == 0:
            return []
        
        V_set = set(values)
        max_v = max(values)
        min_v = min(values)
        
        best = []
        best_set = set()
        candidates = []

        # Strategy 1: Odd numbers
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)

        # Strategy 2: Large positive numbers
        thresh_pos = max_v / 2.0
        s2 = [x for x in values if x > thresh_pos]
        candidates.append(s2)

        # Strategy 3: Large negative numbers
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)

        # Strategy 4: Union of large pos and neg
        s4 = sorted(set(s2).union(s3))
        candidates.append(s4)
        
        # Strategy 5: Top 1/3 by value
        s5 = sorted(values, reverse=True)[:n//3 + 2]
        candidates.append(s5)
        
        # Strategy 6: Bottom 1/3 by value
        s6 = sorted(values)[:n//3 + 2]
        candidates.append(s6)

        # Strategy 7: Greedy by absolute value
        s7_list, s7_set = greedy_add([], sorted(values, key=abs, reverse=True))
        candidates.append(s7_list)

        # Strategy 8: Greedy by value (smallest first)
        s8_list, s8_set = greedy_add([], sorted(values))
        candidates.append(s8_list)

        # Strategy 9: Greedy by value (largest first)
        s9_list, s9_set = greedy_add([], sorted(values, reverse=True))
        candidates.append(s9_list)

        # Strategy 10: Random shuffles
        rng = random.Random(42)
        for trial in range(10):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, ss = greedy_add([], shuffled)
            candidates.append(sl)

        # Evaluate and keep best
        for c in candidates:
            if not is_sum_free(c):
                c_valid = list(c)
                c_set = set(c_valid)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(c_valid):
                        x = c_valid[i]
                        conflict = False
                        for y in c_set:
                            if y == x:
                                continue
                            if (x + y) in c_set:
                                conflict = True
                                break
                            if (x - y) in c_set:
                                conflict = True
                                break
                        if conflict:
                            c_valid.pop(i)
                            c_set.remove(x)
                            changed = True
                        else:
                            i += 1
                c = c_valid

            if len(c) > len(best):
                best = c
                best_set = set(c)

        # Local search on best
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            try:
                ts = min(time_budget, remaining_time - 3)
                improved = local_search(best, values, n, time_limit=ts)
                if len(improved) > len(best):
                    best = improved
                    best_set = set(best)
            except:
                pass

        # Try additional random restarts if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                trial_list, trial_set = greedy_add([], shuffled)

                if len(trial_list) >= len(best) - 2:
                    ts = min(3.0, deadline - time.time() - 3)
                    if ts > 1:
                        try:
                            improved = local_search(trial_list, values, n, time_limit=ts)
                            if len(improved) > len(best):
                                best = improved
                                best_set = set(best)
                        except:
                            pass

        return best

    def greedy_add(S_list, candidates):
        S_set = set(S_list)
        result = list(S_list)
        for c in candidates:
            if c in S_set:
                continue
            # Check if adding c creates a conflict
            valid = True
            for x in S_set:
                if (c + x) in S_set:
                    valid = False
                    break
                if (c - x) in S_set:
                    valid = False
                    break
            if valid:
                result.append(c)
                S_set.add(c)
        return result, S_set

    def local_search(initial_set, values, n, time_limit=5.0):
        active = set(initial_set)
        values_set = set(values)
        rng = random.Random(98765)
        start = time.time()
        best_size = len(active)
        best_set = set(active)
        no_improve = 0

        # Pre-compute some structures for speed if needed, but for now let's just use sets
        # Actually, for local search, we need to check conflicts quickly.
        
        def can_add(candidate, S):
            if candidate in S:
                return False
            for x in S:
                if (candidate + x) in S:
                    return False
                if (candidate - x) in S:
                    return False
            return True

        def count_conflicts(S):
            count = 0
            for a in S:
                for b in S:
                    if a + b in S:
                        count += 1
            return count

        for _ in range(5000):
            if time.time() - start > time_limit:
                break
            
            # Try to add
            added = False
            candidates_to_add = [v for v in values if v not in active]
            rng.shuffle(candidates_to_add)
            for c in candidates_to_add:
                if can_add(c, active):
                    active.add(c)
                    added = True
                    if len(active) > best_size:
                        best_size = len(active)
                        best_set = set(active)
                        no_improve = 0
                    else:
                        no_improve += 1
                    break

            if not added:
                # Try to remove one from a conflict
                # Find a conflict
                conflict = None
                active_list = list(active)
                for a in active_list:
                    for b in active_list:
                        if a + b in active:
                            conflict = (a, b, a+b)
                            break
                    if conflict:
                        break
                
                if conflict:
                    # Remove one of the elements in conflict
                    # Choose one randomly or the one with fewest connections?
                    a, b, c = conflict
                    # Just remove c for now? Or random from {a,b,c}?
                    # Let's try removing the element that appears in the most conflicts? 
                    # For simplicity, remove a random one from the triple.
                    remove_candidate = rng.choice([a, b, c])
                    active.remove(remove_candidate)
                    no_improve += 1
                else:
                    # No conflict, but couldn't add? That means we are stuck.
                    # Try a swap: remove one, try to add two? Or just break if no improvement for a while?
                    if no_improve > 100:
                        break
                    else:
                        # Try swap: remove one random, see if we can add a different one
                        if active:
                            to_remove = rng.choice(list(active))
                            active.remove(to_remove)
                            # Try to add something back
                            for c in candidates_to_add:
                                if can_add(c, active):
                                    active.add(c)
                                    break
                            no_improve += 1

        return list(best_set)

    results = []
    time_alloc = {60: 12.0, 120: 25.0, 240: 80.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue

        values = instance(size, seed)
        budget = time_alloc.get(size, 15.0)
        remaining = deadline - time.time()
        budget = min(budget, max(2.0, remaining - 3.0))

        subset = solve_one(values, time_budget=budget)

        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]

        results.append(subset)

    return results
