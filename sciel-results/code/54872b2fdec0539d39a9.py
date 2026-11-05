"""Rewritten program for large sum-free subsets of three fixed sets."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def greedy_from_order(S, order):
        """Build a sum-free subset by iterating in 'order', adding if possible."""
        chosen = []
        chosen_set = set()
        for x in order:
            # Check: for all y in chosen_set, x+y not in chosen_set, and 2x not in chosen_set
            # Also need: for all y in chosen_set, y+x not in chosen_set (same as above)
            # And: x + y != z for z in chosen_set => x + y not in chosen_set
            # Actually we need: no a,b,c in chosen such that a+b=c.
            # When adding x, we need to check:
            # 1) For all a in chosen_set: a + x not in chosen_set
            # 2) For all b in chosen_set: b + x not in chosen_set (same as 1)
            # 3) 2*x not in chosen_set (a=b=x case)
            # 4) x + a not in chosen_set for all a (same as 1)
            # Wait, also need: x + x = 2x not in chosen_set, and x + a not in chosen_set for all a in chosen_set.
            # But also: a + b = x for a,b in chosen_set? No, x is being added, so a+b=x would mean x is the sum.
            # The condition is: no a,b,c all in the final set with a+b=c.
            # When adding x, new violations are:
            # - a + b = x for a,b in chosen_set
            # - a + x = c for a in chosen_set, c in chosen_set
            # - x + a = c for a in chosen_set, c in chosen_set (same)
            # - x + x = 2x in chosen_set
            ok = True
            # Check a + x = c for a,c in chosen_set
            for a in chosen_set:
                if a + x in chosen_set:
                    ok = False
                    break
            if ok:
                # Check x + a = c (same as above since addition is commutative, already covered)
                # Check 2x in chosen_set
                if 2 * x in chosen_set:
                    ok = False
            if ok:
                # Check a + b = x for a,b in chosen_set
                for a in chosen_set:
                    if x - a in chosen_set:
                        ok = False
                        break
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def greedy_improve(S, initial_order, max_iters=500):
        """Local search: start from initial_order, try to improve by swapping or reordering."""
        # Start with greedy from initial_order
        current = greedy_from_order(S, initial_order)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        
        # Build conflict structure: for each pair (a,b), c=a+b is forbidden if a,b in set
        # Local search: try to add elements not in current, or remove+add
        # Simpler: try random restarts with different orderings, and hill-climbing on additions
        
        # Hill climbing: repeatedly try to add a random element from S\current
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            remaining = [x for x in S if x not in current_set]
            if not remaining:
                break
            rng_local = Random(iters * 7919 + hash(tuple(S)) % 100000)
            rng_local.shuffle(remaining)
            for x in remaining:
                # Can we add x?
                ok = True
                for a in current_set:
                    if a + x in current_set:
                        ok = False
                        break
                if ok and 2 * x in current_set:
                    ok = False
                if ok:
                    for a in current_set:
                        if x - a in current_set:
                            ok = False
                            break
                if ok:
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    break
        return best

    def solve_instance(S, time_budget):
        """Find a large sum-free subset of S."""
        n = len(S)
        if n == 0:
            return []
        
        # Strategy 1: Odd numbers
        odd = [x for x in S if x % 2 != 0]
        
        # Strategy 2: Numbers in upper half (positive large)
        positive = [x for x in S if x > 0]
        if positive:
            positive.sort(reverse=True)
            # Take the largest n/2 positive numbers? They might still have sums in the set.
            # Better: take numbers > max(S)/2
            max_val = max(S)
            upper = [x for x in S if x > max_val / 2]
        
        # Strategy 3: Greedy from various orderings
        # Order by absolute value descending
        order1 = sorted(S, key=lambda x: -abs(x))
        # Order by value descending
        order2 = sorted(S, reverse=True)
        # Order by value ascending
        order3 = sorted(S)
        # Random orderings
        rng = Random(42)
        orders = [order1, order2, order3]
        for _ in range(10):
            o = list(S)
            rng.shuffle(o)
            orders.append(o)
        
        best_subset = []
        best_size = 0
        
        for o in orders:
            subset = greedy_from_order(S, o)
            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = subset
        
        # Also try odd numbers as a baseline
        if len(odd) > best_size:
            best_size = len(odd)
            best_subset = odd
        
        # Try to improve the best subset with hill climbing
        # Use time budget
        t_start = time.time()
        improved_subset = greedy_improve(S, best_subset, max_iters=200)
        if len(improved_subset) > best_size:
            best_size = len(improved_subset)
            best_subset = improved_subset
        
        # More aggressive local search: try swapping elements
        t_start = time.time()
        while time.time() - t_start < time_budget:
            # Try to find a better subset by random perturbations
            rng2 = Random(int(time.time() * 1000) % 1000000)
            # Start from current best, try to replace one element
            if not best_subset:
                break
            # Try adding a random element
            remaining = [x for x in S if x not in set(best_subset)]
            if remaining:
                x = rng2.choice(remaining)
                # Check if we can add x
                bs = set(best_subset)
                ok = True
                for a in bs:
                    if a + x in bs:
                        ok = False
                        break
                if ok and 2 * x in bs:
                    ok = False
                if ok:
                    for a in bs:
                        if x - a in bs:
                            ok = False
                            break
                if ok:
                    best_subset.append(x)
                    continue
            
            # Try a random restart with a new ordering
            o = list(S)
            rng2.shuffle(o)
            # Bias: put elements that are "harder" later
            subset = greedy_from_order(S, o)
            if len(subset) > len(best_subset):
                best_subset = subset
        
        return best_subset

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        # Allocate time: more time for larger instances
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: just use odd numbers
            results.append([x for x in S if x % 2 != 0])
            continue
        # Allocate proportional time
        time_budget = max(1, remaining * (size / 420))  # 60+120+240=420
        subset = solve_instance(S, time_budget)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
