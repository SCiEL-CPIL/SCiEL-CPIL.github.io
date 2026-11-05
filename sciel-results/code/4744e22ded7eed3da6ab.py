"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before the 150s kill

    def remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to sum-free set S (given as a set)."""
        # x + y in S for some y in S  =>  for all y in S: x+y not in S
        # x = a + b for some a,b in S => for all a in S: x-a not in S
        # Combine: for all z in S: (x+z) not in S AND (x-z) not in S
        for z in S_set:
            if x + z in S_set:
                return False
            if x - z in S_set:
                return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset. Returns list of chosen elements."""
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def greedy_then_addback(arr, key, time_budget):
        """Greedy in given order, then iteratively try to add back excluded elements."""
        chosen = []
        chosen_set = set()
        excluded = []
        for x in sorted(arr, key=key):
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
            else:
                excluded.append(x)
        # Add-back pass: repeatedly scan excluded and add any that now fit
        improved = True
        while improved and time_budget() > 0.3:
            improved = False
            new_excluded = []
            for x in excluded:
                if can_add(x, chosen_set):
                    chosen.append(x)
                    chosen_set.add(x)
                    improved = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        return chosen

    def local_search(arr, initial, time_budget):
        """Hill-climbing: try adding elements, and swap-remove when blocked."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = arr_set - current_set
        best = list(current)
        best_len = len(current)

        rng = random.Random(999)
        iterations = 0
        while time_budget() > 0.5 and iterations < 20000:
            iterations += 1
            if not not_in:
                break
            x = rng.choice(tuple(not_in))
            if can_add(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.discard(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                # Try removing one element (that conflicts with x) then adding x
                if current:
                    # Find a conflicting element to remove
                    removed = None
                    for y in current:
                        if x + y in current_set or x - y in current_set:
                            removed = y
                            break
                    if removed is None:
                        removed = current[rng.randrange(len(current))]
                    current.remove(removed)
                    current_set.discard(removed)
                    if can_add(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.discard(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                    else:
                        current.append(removed)
                        current_set.add(removed)
        return best

    def solve_instance(arr):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        # Order 1: by absolute value ascending (small magnitudes first)
        cand = greedy_then_addback(arr, lambda x: (abs(x), x), remaining)
        if len(cand) > len(best):
            best = cand
        # Order 2: by value ascending
        cand = greedy_then_addback(arr, lambda x: x, remaining)
        if len(cand) > len(best):
            best = cand
        # Order 3: by value descending
        cand = greedy_then_addback(arr, lambda x: -x, remaining)
        if len(cand) > len(best):
            best = cand
        # Order 4: by absolute value descending (large magnitudes first)
        cand = greedy_then_addback(arr, lambda x: (-abs(x), x), remaining)
        if len(cand) > len(best):
            best = cand
        # Order 5: by absolute value descending, tie-break by value
        cand = greedy_then_addback(arr, lambda x: (-abs(x), -x), remaining)
        if len(cand) > len(best):
            best = cand
        # Random restarts while time remains
        rng = random.Random(12345)
        guard = 0
        while remaining() > 4.0 and guard < 200:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = greedy_then_addback(order, lambda x: x, remaining)
            if len(cand) > len(best):
                best = cand
        # Local search: try to improve by adding/swapping elements
        if remaining() > 1.0:
            best = local_search(arr, best, remaining)
        # Final safety: ensure validity
        if not is_sum_free(best):
            # rebuild deterministically
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
