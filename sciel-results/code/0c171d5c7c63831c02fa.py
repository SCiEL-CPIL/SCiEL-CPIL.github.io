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

    def count_violations(S):
        """Count number of sum-free violations (a+b=c) in list S."""
        S_set = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in S_set:
                    count += 1
        return count

    def greedy_sum_free(S, order_seed=0):
        """
        Greedy: process elements in a given order, add if it doesn't create a violation.
        Returns the sum-free subset.
        """
        rng = random.Random(order_seed)
        elements = list(S)
        rng.shuffle(elements)
        
        chosen = []
        chosen_set = set()
        
        for x in elements:
            # Check if adding x creates any violation
            # Violation: a+b=x for some a,b in chosen_set, OR x+b=a for some b,a in chosen_set
            # i.e., x + b in chosen_set for some b in chosen_set, or a + x in chosen_set for some a in chosen_set
            # Actually: we need to check if there exist a, b in chosen_set such that a + b = x
            # Also check if x + b in chosen_set for any b in chosen_set (i.e., x is the 'a' or 'b')
            # More precisely: adding x violates if:
            #   1) There exist a, b in chosen_set with a + b = x
            #   2) There exists b in chosen_set with x + b in chosen_set  (x acts as a or b)
            #   3) There exists a in chosen_set with a + x in chosen_set  (same as 2 essentially)
            
            # Condition 1: x = a + b for some a, b in chosen_set
            # Condition 2: x + b = c for some b, c in chosen_set, i.e., c - x in chosen_set and c in chosen_set
            #             equivalently, for some b in chosen_set, x + b in chosen_set
            
            ok = True
            for b in chosen_set:
                if x + b in chosen_set:
                    ok = False
                    break
                if b - x in chosen_set and b != x:
                    # b = x + (b-x), so x + (b-x) = b, violation
                    ok = False
                    break
                if x - b in chosen_set:
                    # x = b + (x-b), so b + (x-b) = x, violation
                    ok = False
                    break
            
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        
        return chosen

    def local_search(S, initial_subset, max_iters=2000, rng_seed=42):
        """
        Local search: try to add elements from S not in subset, or swap to improve.
        We try to add elements that don't create violations.
        """
        rng = random.Random(rng_seed)
        subset = list(initial_subset)
        subset_set = set(subset)
        candidates = [x for x in S if x not in subset_set]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            rng.shuffle(candidates)
            
            new_subset = list(subset)
            new_set = set(subset)
            
            for x in candidates:
                # Check if x can be added
                can_add = True
                for b in new_set:
                    if x + b in new_set:
                        can_add = False
                        break
                    if b - x in new_set and b != x:
                        can_add = False
                        break
                    if x - b in new_set:
                        can_add = False
                        break
                
                if can_add:
                    new_subset.append(x)
                    new_set.add(x)
                    improved = True
            
            if improved:
                subset = new_subset
                subset_set = new_set
                candidates = [x for x in S if x not in subset_set]
        
        return subset

    def solve_instance(S, time_budget):
        """
        Solve one instance with multiple greedy starts and local search.
        """
        best = None
        best_size = 0
        
        # Try several different orderings
        num_starts = 20
        for i in range(num_starts):
            if time.time() > start_time + time_budget:
                break
            
            greedy_result = greedy_sum_free(S, order_seed=i)
            ls_result = local_search(S, greedy_result, max_iters=1000, rng_seed=i)
            
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best = ls_result
        
        # Also try the simple odd/even heuristics as baselines
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > best_size:
            best = odds
            best_size = len(odds)
        
        evens = [x for x in S if x % 2 == 0]
        if len(evens) > best_size:
            best = evens
            best_size = len(evens)
        
        # Try positive-only and negative-only
        positives = [x for x in S if x > 0]
        if len(positives) > best_size:
            best = positives
            best_size = len(positives)
        
        negatives = [x for x in S if x < 0]
        if len(negatives) > best_size:
            best = negatives
            best_size = len(negatives)
        
        # Try large-magnitude elements
        by_abs = sorted(S, key=lambda x: -abs(x))
        greedy_abs = greedy_sum_free(S, order_seed=999)
        ls_abs = local_search(S, greedy_abs, max_iters=500, rng_seed=999)
        if len(ls_abs) > best_size:
            best = ls_abs
            best_size = len(ls_abs)
        
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
