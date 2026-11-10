"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    # Calculate LCM
    L = reduce(math.lcm, elems)
    
    # Convert to scaled integers
    weights = [L // e for e in elems]
    
    # Meet-in-the-middle for larger sets
    if n > 12:
        mid = n // 2
        left_weights = weights[:mid]
        right_weights = weights[mid:]
        
        # Generate all subset sums for left half
        left_sums = {0}
        for w in left_weights:
            left_sums = left_sums | {s + w for s in left_sums}
        
        # Generate all subset sums for right half
        right_sums = {0}
        for w in right_weights:
            right_sums = right_sums | {s + w for s in right_sums}
        
        # Check if any sum can be formed in two ways
        # A collision exists if there are s1 != s1' in left_sums and s2 != s2' in right_sums
        # such that s1 + s2 = s1' + s2'
        # Equivalently: (s1 - s1') = (s2' - s2)
        # We need to check if the difference sets have non-trivial overlap
        
        # Better approach: check if total number of distinct sums equals 2^n
        # But that's expensive. Instead, check for collisions directly.
        
        # For each pair of left sums, check if the difference appears in right differences
        # This is still O(2^(n/2)^2) which might be too slow for n=22
        
        # Alternative: use the fact that if we have 2^n distinct sums, no collision exists
        # Let's just build all sums and check count
        all_sums = set()
        for ls in left_sums:
            for rs in right_sums:
                all_sums.add(ls + rs)
        
        return len(all_sums) == (1 << n)
    
    # For smaller sets, use incremental check
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        # Check for collision between new_sums and sums
        if sums & new_sums:
            return False
        sums = sums | new_sums
        
    return len(sums) == (1 << n)

def _build_sums(elems):
    """Build the set of all subset sums for the given elements (scaled integers).
    Returns (L, weights, sums_set) or None if not dissociated."""
    if not elems:
        return (1, [], {0})
    L = reduce(math.lcm, elems)
    weights = [L // e for e in elems]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if sums & new_sums:
            return None
        sums |= new_sums
    return (L, weights, sums)

def _can_add_incremental(L, weights, sums, new_elem):
    """Check if we can add new_elem to a set with LCM L, weights, and sums.
    Returns (new_L, new_weights, new_sums) if dissociated, else None."""
    new_L = math.lcm(L, new_elem)
    scale = new_L // L
    new_w = new_L // new_elem
    if scale == 1:
        # No scaling needed - just check if any existing sum + new_w is already in sums
        # For large sums sets, use a more efficient collision check
        if len(sums) > 4096:
            # Use sorted list for binary search
            sorted_sums = sorted(sums)
            import bisect
            for s in sorted_sums:
                target = s + new_w
                idx = bisect.bisect_left(sorted_sums, target)
                if idx < len(sorted_sums) and sorted_sums[idx] == target:
                    return None
        else:
            for s in sums:
                if (s + new_w) in sums:
                    return None
        new_weights = weights + [new_w]
        new_sums = sums | {s + new_w for s in sums}
        return (new_L, new_weights, new_sums)
    else:
        # Need to scale
        scaled_sums = {s * scale for s in sums}
        if len(scaled_sums) > 4096:
            sorted_sums = sorted(scaled_sums)
            import bisect
            for s in sorted_sums:
                target = s + new_w
                idx = bisect.bisect_left(sorted_sums, target)
                if idx < len(sorted_sums) and sorted_sums[idx] == target:
                    return None
        else:
            for s in scaled_sums:
                if (s + new_w) in scaled_sums:
                    return None
        new_weights = [w * scale for w in weights] + [new_w]
        new_sums = scaled_sums | {s + new_w for s in scaled_sums}
        return (new_L, new_weights, new_sums)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, keep buffer for safety/overhead

    universe = list(range(1, 23))

    def time_left():
        return deadline - time.time()

    def greedy(order, max_size=22):
        """Greedy construction using incremental checks."""
        L = 1
        weights = []
        sums = {0}
        result = []
        for e in order:
            if len(result) >= max_size:
                break
            res = _can_add_incremental(L, weights, sums, e)
            if res is not None:
                L, weights, sums = res
                result.append(e)
        return result

    def local_search_add(s, L, weights, sums):
        """Try to add any missing element."""
        improved = True
        while improved and time_left() > 5:
            improved = False
            s_set = set(s)
            remaining = [e for e in universe if e not in s_set]
            random.shuffle(remaining)
            for e in remaining:
                res = _can_add_incremental(L, weights, sums, e)
                if res is not None:
                    L, weights, sums = res
                    s = s + [e]
                    improved = True
                    break
        return s, L, weights, sums

    def local_search_swap(s, L, weights, sums):
        """Try swaps to escape local maxima."""
        improved = True
        while improved and time_left() > 5:
            improved = False
            s_set = set(s)
            in_list = list(s)
            out_list = [e for e in universe if e not in s_set]
            random.shuffle(out_list)
            for u in in_list:
                if time_left() < 5:
                    break
                for v in out_list:
                    if time_left() < 5:
                        break
                    # Rebuild without u, then try adding v
                    new_elems = [x for x in s if x != u] + [v]
                    res = _build_sums(new_elems)
                    if res is not None:
                        L, weights, sums = res
                        s = new_elems
                        # Try to extend
                        s, L, weights, sums = local_search_add(s, L, weights, sums)
                        improved = True
                        break
                if improved:
                    break
        return s, L, weights, sums

    best_set = []
    best_L = 1
    best_weights = []
    best_sums = {0}

    # 1. Deterministic greedy orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe),
        # Try orders that favor larger elements first (they create more separation)
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]

    for order in orders:
        if time_left() < 10:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = result
            res = _build_sums(result)
            if res:
                best_L, best_weights, best_sums = res

    # 2. Local search add on best
    best_set, best_L, best_weights, best_sums = local_search_add(
        best_set, best_L, best_weights, best_sums)

    # 3. Random restarts with local search (biased toward larger elements)
    num_restarts = 0
    while time_left() > 15:
        # Weighted random: larger elements get higher priority
        weights_rand = [e * e for e in universe]  # quadratic bias toward large
        total_w = sum(weights_rand)
        order = []
        remaining = list(universe)
        rem_weights = list(weights_rand)
        for _ in range(len(universe)):
            r = random.random() * sum(rem_weights)
            cum = 0
            for i, w in enumerate(rem_weights):
                cum += w
                if r <= cum:
                    order.append(remaining[i])
                    remaining.pop(i)
                    rem_weights.pop(i)
                    break
        result = greedy(order)
        res = _build_sums(result)
        if res:
            L, weights, sums = res
        else:
            L, weights, sums = 1, [], {0}
            result = []
        result, L, weights, sums = local_search_add(result, L, weights, sums)
        
        if len(result) > len(best_set):
            best_set = result
            best_L, best_weights, best_sums = L, weights, sums
            if len(best_set) >= 16:
                break
        num_restarts += 1

    # 4. Swap-based local search with more aggressive exploration
    best_set, best_L, best_weights, best_sums = local_search_swap(
        best_set, best_L, best_weights, best_sums)
    
    # 4.5. Try to add more elements with different orderings
    if time_left() > 10:
        # Try adding elements in decreasing order of "difficulty" (larger elements first)
        s_set = set(best_set)
        remaining = sorted([e for e in universe if e not in s_set], reverse=True)
        for e in remaining:
            if time_left() < 5:
                break
            res = _can_add_incremental(best_L, best_weights, best_sums, e)
            if res is not None:
                best_L, best_weights, best_sums = res
                best_set.append(e)

    # 5. 2-for-1 swap: remove one, try adding two different ones
    improved = True
    while improved and time_left() > 5:
        improved = False
        s_set = set(best_set)
        in_list = list(best_set)
        out_list = [e for e in universe if e not in s_set]
        random.shuffle(out_list)
        for u in in_list:
            if time_left() < 5:
                break
            # Build without u
            elems_without = [x for x in best_set if x != u]
            res = _build_sums(elems_without)
            if res is None:
                continue
            L_tmp, w_tmp, s_tmp = res
            added_count = 0
            for v in out_list:
                if time_left() < 5:
                    break
                res2 = _can_add_incremental(L_tmp, w_tmp, s_tmp, v)
                if res2 is not None:
                    L_tmp, w_tmp, s_tmp = res2
                    elems_without.append(v)
                    added_count += 1
                    if added_count >= 2:
                        break
            if added_count >= 2:
                best_set = elems_without
                best_L, best_weights, best_sums = L_tmp, w_tmp, s_tmp
                improved = True
                # Try to extend further
                best_set, best_L, best_weights, best_sums = local_search_add(
                    best_set, best_L, best_weights, best_sums)
                break

    # 6. Final add pass
    best_set, best_L, best_weights, best_sums = local_search_add(
        best_set, best_L, best_weights, best_sums)

    # Final verification
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback - a known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
