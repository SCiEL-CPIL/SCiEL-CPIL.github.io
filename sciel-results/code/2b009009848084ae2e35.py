"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 130.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def greedy_sum_free(elements, max_time=None):
        """Greedy: try to build largest sum-free subset.
        Strategy: sort by some heuristic, add if doesn't violate sum-free."""
        n = len(elements)
        if n <= 1:
            return list(elements)
        
        # Quick check: all odd
        odd = [x for x in elements if x % 2 != 0]
        if len(odd) >= n * 0.45:
            return odd
        
        # Try multiple orderings and pick best
        best = odd  # fallback
        best_size = len(odd)
        
        # Heuristic 1: sort by absolute value, try adding in that order
        # Heuristic 2: sort by value
        # Heuristic 3: random orders
        
        def try_order(order):
            if max_time and time.time() > max_time:
                return None
            s = set()
            result = []
            for x in order:
                # Check if adding x keeps sum-free
                ok = True
                for a in result:
                    if a + x in s or x + a in s:
                        ok = False
                        break
                    if 2 * x in s or x in s:
                        ok = False
                        break
                if ok:
                    s.add(x)
                    result.append(x)
            return result

        # Try sorted by abs value
        order1 = sorted(elements, key=lambda x: abs(x))
        r1 = try_order(order1)
        if r1 and len(r1) > best_size:
            best, best_size = r1, len(r1)

        # Try sorted by value
        order2 = sorted(elements)
        r2 = try_order(order2)
        if r2 and len(r2) > best_size:
            best, best_size = r2, len(r2)

        # Try a few random orders
        rng = random.Random(42)
        for _ in range(5):
            if max_time and time.time() > max_time:
                break
            order = list(elements)
            rng.shuffle(order)
            r = try_order(order)
            if r and len(r) > best_size:
                best, best_size = r, len(r)

        # Local search: try to improve by swapping
        if max_time and time.time() < max_time - 5:
            best = local_search(elements, best, max_time)
        
        return best

    def local_search(elements, current, max_time=None):
        """Try to improve current sum-free set by removing one and adding another."""
        current_set = set(current)
        current_list = list(current)
        elements_set = set(elements)
        
        def can_add(x, s):
            for a in s:
                if a + x in s:
                    return False
            if 2 * x in s:
                return False
            return True

        improved = True
        while improved:
            if max_time and time.time() > max_time:
                break
            improved = False
            # Try removing each element and adding a better one
            for i in range(len(current_list)):
                if max_time and time.time() > max_time:
                    break
                removed = current_list[i]
                s = current_set - {removed}
                # Find best element to add
                best_add = None
                best_score = 0
                for x in elements:
                    if x in s:
                        continue
                    if can_add(x, s):
                        # Score: prefer elements that allow more future additions
                        # Simple: just take it
                        best_add = x
                        break
                if best_add is not None and best_add != removed:
                    # Check if this actually improves (it's a swap, size stays same)
                    # Only do swap if it helps future additions - skip for simplicity
                    pass
            
            # Try adding elements not in set
            for x in elements:
                if max_time and time.time() > max_time:
                    break
                if x not in current_set:
                    if can_add(x, current_set):
                        current_set.add(x)
                        current_list.append(x)
                        improved = True
        
        return list(current_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        remaining = _DEADLINE - time.time() - 5.0
        if remaining < 1.0:
            # Not enough time, just return odds
            results.append([x for x in elems if x % 2 != 0])
        else:
            result = greedy_sum_free(elems, max_time=_DEADLINE - 5.0)
            # Verify sum-free
            if not is_sum_free(result):
                result = [x for x in elems if x % 2 != 0]
            results.append(result)

    return results
# EVOLVE-BLOCK-END
