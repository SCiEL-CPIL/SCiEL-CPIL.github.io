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
        # new sums: old sums + x
        # Vectorized: shift bs by x positions
        if x >= 201:
            return None
        new_sums = np.zeros(201, dtype=bool)
        new_sums[x:] = bs[:201-x]
        if new_sums[200]:
            return None
        return bs | new_sums
    
    # Greedy: try largest first
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
    
    # Try largest-first
    best = greedy(range(100, 0, -1))
    best_size = len(best)
    
    # Try smallest-first
    if time.time() < deadline:
        candidate = greedy(range(1, 101))
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try various structured orderings
    if time.time() < deadline - 2.0:
        # Try alternating large/small
        order = []
        for i in range(50):
            order.append(100 - i)
            if i < 50:
                order.append(i + 1)
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Try multiples of 3 first, then others
        order = [x for x in range(1, 101) if x % 3 != 0] + [x for x in range(1, 101) if x % 3 == 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Try even numbers first
        order = [x for x in range(1, 101) if x % 2 == 0] + [x for x in range(1, 101) if x % 2 != 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try random orderings with different seeds
    for seed in [42, 123, 456, 789, 1011, 2023, 3141, 2718]:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        for trial in range(50):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, 101))
            rng.shuffle(order)
            candidate = greedy(order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(best)
    
    # Hypergraph-based greedy: select vertex with lowest degree (fewest forbidden subsets)
    # A "forbidden subset" for element x is any subset S of current selected elements
    # such that sum(S) + x == 200, i.e., sum(S) == 200 - x
    # We count how many reachable sums equal 200 - x for each candidate x
    if time.time() < deadline - 10.0:
        selected = []
        selected_set = set()
        bs = make_bitset()
        bs[0] = True
        
        remaining = list(range(1, 101))
        
        while remaining:
            if time.time() > deadline - 5.0:
                break
            
            # For each remaining element, compute its "degree" = number of ways
            # it could complete a forbidden sum
            # degree(x) = 1 if (200 - x) is reachable, else 0
            # But we want to minimize the number of future conflicts, so we prefer
            # elements with lower degree (fewer immediate conflicts)
            
            best_x = None
            best_degree = float('inf')
            best_tiebreak = -1
            
            for x in remaining:
                target = 200 - x
                if target < 0:
                    degree = 0
                elif target <= 200 and bs[target]:
                    degree = 1
                else:
                    degree = 0
                
                # Tie-break: prefer larger elements (they're harder to combine)
                # Actually, prefer elements that don't create new reachable sums
                # that could enable future conflicts
                if degree < best_degree or (degree == best_degree and x > best_tiebreak):
                    best_degree = degree
                    best_x = x
                    best_tiebreak = x
            
            if best_x is None:
                break
            
            new_bs = add_to_bitset(bs, best_x)
            if new_bs is not None:
                selected.append(best_x)
                selected_set.add(best_x)
                bs = new_bs
                remaining.remove(best_x)
            else:
                # Should not happen if degree check is correct, but skip
                remaining.remove(best_x)
        
        if len(selected) > best_size:
            best = selected
            best_size = len(best)
    
    # Local search: try to add elements not in best
    for pass_num in range(3):
        if time.time() < deadline - 5.0:
            best_set = set(best)
            bs = make_bitset()
            bs[0] = True
            for x in best:
                bs = add_to_bitset(bs, x)  # should not be None
            # Try adding each missing element
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
    
    # Try swap-based local search: remove one element, try to add more
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        best_set = set(best)
        improved = False
        for remove_x in list(best):
            # Try removing remove_x and see if we can add more
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
            # Try adding all missing elements
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
    
    # Try double-swap: remove two elements, try to add three
    if time.time() < deadline - 10.0:
        best_set = set(best)
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
                # Try adding all missing elements
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
