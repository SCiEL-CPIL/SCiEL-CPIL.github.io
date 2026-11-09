"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_construction(order):
    """Fast greedy: pick elements in given order, tracking reachable subset sums via bitset."""
    reachable = 1  # bit i set means sum i is reachable; start with sum 0
    selected = []
    for x in order:
        # Adding x creates sum 200 iff sum (200-x) is currently reachable
        if (200 - x) >= 0 and (reachable >> (200 - x)) & 1:
            continue
        selected.append(x)
        # Update reachable sums: new_sums = old_sums | (old_sums << x), masked to [0,200]
        mask = (1 << 201) - 1  # bits 0..200
        reachable = (reachable | (reachable << x)) & mask
    return selected

def _has_sum_200(elements):
    """Check if any subset of elements sums to exactly 200."""
    reachable = 1
    for x in elements:
        mask = (1 << 201) - 1
        reachable = (reachable | (reachable << x)) & mask
    return (reachable >> 200) & 1

def _reachable_bitset(elements):
    """Compute the reachable sums bitset for a list of elements."""
    reachable = 1
    mask = (1 << 201) - 1
    for x in elements:
        reachable = (reachable | (reachable << x)) & mask
    return reachable

def _can_add(element, reachable):
    """Check if we can add 'element' given current reachable sums."""
    needed = 200 - element
    if needed < 0:
        return True
    return not ((reachable >> needed) & 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    # Start with known-good baseline: {66, ..., 100} has no subset summing to 200
    # (min 3-element sum = 66+67+68=201>200, max 2-element sum=99+100=199<200)
    best = list(range(66, 101))
    best_size = len(best)
    best_reachable = _reachable_bitset(best)

    # Phase 1: Try greedy constructions with various orderings
    deterministic_orders = [
        list(range(100, 0, -1)),          # descending
        list(range(1, 101)),              # ascending
        list(range(51, 101)) + list(range(1, 51)),  # large then small
        list(range(1, 51)) + list(range(100, 50, -1)),  # small then large
        list(range(100, 50, -1)) + list(range(1, 51)),  # large desc then small asc
        list(range(1, 51)) + list(range(51, 101)),     # small asc then large asc
        list(range(100, 65, -1)) + list(range(1, 66)),  # >65 then <=65
        list(range(1, 66)) + list(range(100, 65, -1)),  # <=65 then >65
    ]

    for order in deterministic_orders:
        if time.time() > deadline - 15.0:
            break
        result = _greedy_construction(order)
        if len(result) > best_size:
            best_size = len(result)
            best = result[:]
            best_reachable = _reachable_bitset(best)

    # Random restarts
    rng = random.Random(42)
    trial = 0
    while time.time() < deadline - 15.0:
        order = list(range(1, 101))
        rng.shuffle(order)
        result = _greedy_construction(order)
        if len(result) > best_size:
            best_size = len(result)
            best = result[:]
            best_reachable = _reachable_bitset(best)
        trial += 1
        if trial % 500 == 0 and time.time() > deadline - 15.0:
            break

    # Phase 2: Local search - try to add elements not in best
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            best_set = set(best)
            available = [x for x in range(1, 101) if x not in best_set]
            
            # Try adding one element at a time using cached reachable bitset
            for x in available:
                if time.time() > deadline - 3.0:
                    break
                if _can_add(x, best_reachable):
                    best.append(x)
                    best_size += 1
                    best_reachable = (best_reachable | (best_reachable << x)) & ((1 << 201) - 1)
                    best_set.add(x)
                    available.remove(x)
                    improved = True

            # Try swapping: remove one, add two
            if not improved and time.time() < deadline - 3.0:
                best_list = best[:]
                for idx_remove in range(len(best_list)):
                    if time.time() > deadline - 3.0:
                        break
                    remove = best_list[idx_remove]
                    remaining = [x for i, x in enumerate(best_list) if i != idx_remove]
                    rem_reachable = _reachable_bitset(remaining)
                    avail = [x for x in range(1, 101) if x not in set(remaining) and x != remove]
                    
                    found = False
                    for i in range(len(avail)):
                        if time.time() > deadline - 3.0 or found:
                            break
                        for j in range(i + 1, len(avail)):
                            if time.time() > deadline - 3.0 or found:
                                break
                            # Check if adding both avail[i] and avail[j] to remaining creates sum 200
                            # First add avail[i]
                            ri = (rem_reachable | (rem_reachable << avail[i])) & ((1 << 201) - 1)
                            if _can_add(avail[j], ri):
                                # Valid swap
                                best = remaining + [avail[i], avail[j]]
                                best_size = len(best)
                                best_reachable = _reachable_bitset(best)
                                improved = True
                                found = True
                                break

    # Final verification
    if _has_sum_200(best):
        best = list(range(66, 101))

    return best

# EVOLVE-BLOCK-END
