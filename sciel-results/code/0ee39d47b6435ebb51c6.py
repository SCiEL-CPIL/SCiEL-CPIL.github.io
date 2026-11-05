import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(S_set, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S_set:
            if (a + x) in S_set:
                return False
            if (x - a) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def is_sum_free(S):
        """Verify S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if (a + b) in S_set:
                    return False
        return True

    def refine(S_init, V, rng, max_iters=100):
        """
        Refine S_init by greedily adding elements from V \ S_init.
        Uses multiple random orderings to escape local optima.
        """
        S_set = set(S_init)
        best_size = len(S_set)
        best_set = set(S_set)
        V_set = set(V)
        
        # Pre-compute elements not in S_init
        candidates = [x for x in V_set if x not in S_set]
        
        if not candidates:
            return list(S_set)

        # Try multiple random orderings
        for _ in range(max_iters):
            current_set = set(S_set)
            current_cands = candidates[:]
            rng.shuffle(current_cands)
            
            for x in current_cands:
                if can_add(current_set, x):
                    current_set.add(x)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
            # Optional: Try to add more in a second pass with different order
            # This handles cases where adding x1 blocks x2, but adding x2 first allows x1
            # But simple greedy is usually sufficient for the first pass.
            
        return list(best_set)

    def local_search_swap(S_list, V, rng, max_iters=50):
        """
        Try to improve S by removing one element and adding two.
        This is a 1-out-2-in move.
        """
        S_set = set(S_list)
        V_set = set(V)
        best_size = len(S_set)
        best_set = set(S_set)
        
        for _ in range(max_iters):
            # Pick a random element to remove
            S_list_temp = list(S_set)
            if not S_list_temp:
                break
            a = rng.choice(S_list_temp)
            S_set.discard(a)
            
            # Try to add 2 elements from V \ S_set
            candidates = [x for x in V_set if x not in S_set]
            rng.shuffle(candidates)
            
            added = 0
            temp_set = set(S_set)
            for x in candidates:
                if can_add(temp_set, x):
                    temp_set.add(x)
                    added += 1
                    if added == 2:
                        break
            
            if len(temp_set) > best_size:
                best_size = len(temp_set)
                best_set = set(temp_set)
            else:
                # Revert
                S_set.add(a)
                
        return list(best_set)

    def search_instance(elements, seed_offset=0, time_budget=30):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        t_start = time.time()

        def update_best(cand):
            nonlocal best, best_size
            if len(cand) > best_size:
                best_size = len(cand)
                best = list(cand)

        # 1. Generate diverse initial seeds
        seeds = []
        
        # Canonical seeds
        odds = [x for x in elements if x % 2 != 0]
        seeds.append(odds)
        
        pos = [x for x in elements if x > 0]
        seeds.append(pos)
        
        neg = [x for x in elements if x < 0]
        seeds.append(neg)
        
        # Intervals
        elements_sorted = sorted(elements)
        n = len(elements_sorted)
        for lo_frac in [0.0, 0.2, 0.4, 0.6]:
            for hi_frac in [0.6, 0.8, 1.0]:
                if hi_frac > lo_frac:
                    lo = int(n * lo_frac)
                    hi = int(n * hi_frac)
                    seeds.append(elements_sorted[lo:hi])
        
        # Top K by magnitude
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.5, 0.7]:
            k = int(n * frac)
            seeds.append(sorted_mag[:k])

        # Random seeds
        for _ in range(5):
            k = int(n * rng.uniform(0.3, 0.8))
            seeds.append(rng.sample(elements, k))

        # 2. Refine each seed
        for seed in seeds:
            if time.time() - t_start > time_budget * 0.8:
                break
            
            # Refine with greedy expansion
            refined = refine(seed, elements, rng, max_iters=20)
            update_best(refined)
            
            # Further refine with swap local search
            if time.time() - t_start < time_budget * 0.9:
                swapped = local_search_swap(refined, elements, rng, max_iters=10)
                update_best(swapped)

        # 3. Random restarts with remaining time
        restart_start = time.time()
        while time.time() - restart_start < time_budget * 0.2:
            k = int(n * rng.uniform(0.3, 0.8))
            seed = rng.sample(elements, k)
            refined = refine(seed, elements, rng, max_iters=5)
            update_best(refined)

        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    # Allocate time budget
    time_budgets = [10, 25, 50]

    results = []
    for i, elems in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5
        if remaining < 3:
            # Fallback: odds
            results.append([x for x in elems if x % 2 != 0])
            continue
        budget = min(time_budgets[i], max(3, remaining))
        result = search_instance(elems, seed_offset=801 + i, time_budget=budget)
        results.append(result)

    # Final verification
    for i, (result, elems) in enumerate(zip(results, instances)):
        if not is_sum_free(result):
            # Fallback to odds if invalid
            results[i] = [x for x in elems if x % 2 != 0]

    return results
