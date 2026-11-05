import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1  # bits 0..200
    TARGET_BIT = 1 << TARGET

    def recompute_reach(selected):
        """Recompute reach bitset from a list/set of selected elements."""
        reachable = 1
        for x in selected:
            reachable |= (reachable << x) & MASK
        return reachable

    def is_safe_add(x, reachable):
        """Check if adding x to current reachable set is safe."""
        return not ((reachable << x) & MASK & TARGET_BIT)

    def greedy_add(selected_set, reachable, candidates, num_passes=1):
        """Add as many candidates as possible to selected_set.
        Returns (selected_set, reachable)."""
        for _ in range(num_passes):
            added_any = False
            for x in candidates:
                if x in selected_set:
                    continue
                new_sums = (reachable << x) & MASK
                if not (new_sums & TARGET_BIT):
                    selected_set.add(x)
                    reachable |= new_sums
                    added_any = True
            if not added_any:
                break
        return selected_set, reachable

    best_S = set()
    best_size = 0
    best_reach = 1

    # Strategy 1: Deterministic orderings (fast, no randomness)
    deterministic_orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
        list(range(100, 50, -1)) + list(range(50, 0, -1)),  # large then medium
        list(range(1, 51)) + list(range(100, 50, -1)),      # small then large
        list(range(50, 101)) + list(range(49, 0, -1)),      # middle-out
    ]

    # Alternating from ends
    alt_order = []
    lo, hi = 1, 100
    while lo <= hi:
        alt_order.append(hi)
        if lo != hi:
            alt_order.append(lo)
        lo += 1
        hi -= 1
    deterministic_orders.append(alt_order)

    for order in deterministic_orders:
        if time.time() > deadline - 5:
            break
        reachable = 1
        selected = []
        for x in order:
            new_sums = (reachable << x) & MASK
            if not (new_sums & TARGET_BIT):
                selected.append(x)
                reachable |= new_sums
        if len(selected) > best_size:
            best_size = len(selected)
            best_S = set(selected)
            best_reach = reachable

    # Strategy 2: Start with safe large numbers, then greedily add small
    # The set {67,...,100} is safe: max pair sum is 100+99=199 < 200
    safe_large = set(range(67, 101))
    safe_large_reach = recompute_reach(safe_large)
    if not (safe_large_reach & TARGET_BIT):
        small_numbers = list(range(1, 67))
        random.shuffle(small_numbers)
        S_temp = set(safe_large)
        reach_temp = safe_large_reach
        S_temp, reach_temp = greedy_add(S_temp, reach_temp, small_numbers, num_passes=3)
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)
            best_reach = reach_temp

    # Strategy 3: Try different safe large set bases
    for base_start in range(60, 70):
        if time.time() > deadline - 10:
            break
        base_set = set(range(base_start, 101))
        base_reach = recompute_reach(base_set)
        if base_reach & TARGET_BIT:
            continue

        S_temp = set(base_set)
        reach_temp = base_reach
        remaining = [x for x in range(1, base_start)]
        random.shuffle(remaining)
        S_temp, reach_temp = greedy_add(S_temp, reach_temp, remaining, num_passes=5)
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)
            best_reach = reach_temp

    # Strategy 4: Randomized greedy with "fewest conflicts" ordering
    for iteration in range(200):
        if time.time() > deadline - 10:
            break

        candidates = list(range(1, 101))
        random.shuffle(candidates)

        S_temp = set()
        reach_temp = 1

        # Greedy with dynamic conflict-aware ordering
        remaining = list(candidates)
        while remaining:
            # Pick the element with fewest conflicts (i.e., (200-x) not in reach)
            # Among those, pick randomly for diversity
            safe_candidates = [x for x in remaining if not (reach_temp & (1 << (TARGET - x)))]
            if not safe_candidates:
                break
            x = random.choice(safe_candidates)
            remaining.remove(x)
            S_temp.add(x)
            reach_temp |= (reach_temp << x) & MASK

        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)
            best_reach = reach_temp

    # Strategy 5: Local search - 1-swap moves
    if time.time() < deadline - 10 and best_S:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_list = list(best_S)
            random.shuffle(S_list)

            for y in S_list:
                if time.time() > deadline - 5:
                    break

                S_temp = set(best_S)
                S_temp.discard(y)
                reach_temp = recompute_reach(S_temp)

                # Re-greedy: add all valid remaining elements
                candidates = [x for x in range(1, 101) if x not in S_temp]
                random.shuffle(candidates)
                S_temp, reach_temp = greedy_add(S_temp, reach_temp, candidates, num_passes=3)

                if len(S_temp) > best_size:
                    best_S = S_temp
                    best_reach = reach_temp
                    best_size = len(best_S)
                    improved = True
                    break

    # Final polish: try to add any remaining valid elements
    if best_S:
        best_reach = recompute_reach(best_S)
        remaining = [x for x in range(1, 101) if x not in best_S]
        random.shuffle(remaining)
        best_S, best_reach = greedy_add(best_S, best_reach, remaining, num_passes=5)
        best_size = len(best_S)

    return list(best_S) if best_S else [1]
