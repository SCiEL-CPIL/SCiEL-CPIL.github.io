"""Combined: systematic backtracking (exact integer via LCM) + random-restart greedy + swap local search."""
import time
import random
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))

    # Precompute reciprocal as integer: 1/i * L
    recip = [L // i for i in range(1, 23)]

    # Best known fallback: 9 elements
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    def time_left():
        return deadline - time.time()

    def is_dissociated(selected):
        """Check if all subset sums are distinct using integer arithmetic."""
        if not selected:
            return True
        sums = [0]
        for idx in selected:
            r = recip[idx - 1]
            new_sums = [s + r for s in sums]
            sums.extend(new_sums)
        return len(sums) == len(set(sums))

    def compute_sums(selected):
        """Compute all subset sums as a set and list."""
        sums = [0]
        for idx in selected:
            r = recip[idx - 1]
            new_sums = [s + r for s in sums]
            sums.extend(new_sums)
        return set(sums), sums

    def try_add_element(sums_set, sums_list, new_idx):
        """Try to add new_idx. Returns (success, new_sums_set, new_sums_list)."""
        r = recip[new_idx - 1]
        new_sums = [s + r for s in sums_list]
        for s in new_sums:
            if s in sums_set:
                return False, None, None
        new_set = sums_set.union(new_sums)
        new_list = sums_list + new_sums
        return True, new_set, new_list

    def backtrack(idx, current, sums_set, sums_list):
        """Backtracking search for maximum dissociated subset."""
        nonlocal best_result
        if time_left() < 1.0:
            return

        # Prune: if we can't beat current best
        remaining = 22 - idx + 1
        if len(current) + remaining <= len(best_result):
            return

        # Try including idx first (to find large sets faster)
        if idx <= 22:
            valid, new_set, new_list = try_add_element(sums_set, sums_list, idx)
            if valid:
                new_current = current + [idx]
                if len(new_current) > len(best_result):
                    best_result = new_current[:]
                backtrack(idx + 1, new_current, new_set, new_list)

        # Try not including idx
        backtrack(idx + 1, current, sums_set, sums_list)

    def local_search_swap(selected, best_k):
        """
        Swap-based local search: try removing one element and adding another
        to find a larger valid set. Returns the best set found.
        """
        nonlocal best_result
        if not selected:
            return selected

        current = list(selected)
        current_set = set(current)
        best_local = current[:]
        best_local_k = len(current)

        max_iters = 50
        for iteration in range(max_iters):
            if time_left() < 2.0:
                break

            found_improvement = False

            # Try each possible swap: remove one, add another
            for remove_a in list(current):
                if len(current) - 1 + 1 <= best_local_k:
                    continue

                # Compute sums without remove_a
                reduced = [x for x in current if x != remove_a]
                r_sums_set, r_sums_list = compute_sums(reduced)

                # Try to add each element not in current
                for add_a in range(1, 23):
                    if add_a in current_set:
                        continue
                    if time_left() < 2.0:
                        break
                    valid, new_set, new_list = try_add_element(r_sums_set, r_sums_list, add_a)
                    if valid:
                        # Found a valid swap
                        new_current = reduced + [add_a]
                        new_current_set = current_set - {remove_a} | {add_a}
                        if len(new_current) > best_local_k:
                            best_local_k = len(new_current)
                            best_local = new_current[:]
                        # Continue searching from this new state
                        current = new_current
                        current_set = new_current_set
                        found_improvement = True
                        break
                if found_improvement:
                    break

            if not found_improvement:
                break

        if len(best_local) > len(best_result):
            best_result = best_local[:]
        return best_local

    def greedy_random(order):
        """Greedily add elements in given order, keeping only if no collision."""
        sums = {0}
        sums_list = [0]
        chosen = []
        for a in order:
            valid, new_set, new_list = try_add_element(sums, sums_list, a)
            if valid:
                sums = new_set
                sums_list = new_list
                chosen.append(a)
        return chosen

    # Phase 1: Systematic backtracking search
    initial_sums_set = {0}
    initial_sums_list = [0]
    backtrack(1, [], initial_sums_set, initial_sums_list)

    # Phase 2: Local search on the best result found
    if time_left() > 5.0:
        local_search_swap(best_result, len(best_result))

    # Phase 3: Random restarts with greedy + local search, using remaining time
    random.seed()
    full = list(range(1, 23))
    best_k = len(best_result)

    while time_left() > 3.0:
        order = full[:]
        random.shuffle(order)
        cand = greedy_random(order)

        # If greedy found something larger, try local search on it
        if len(cand) >= best_k:
            improved = local_search_swap(cand, best_k)
            if len(improved) > best_k:
                best_k = len(improved)
                if best_k >= 15:
                    break
        elif len(cand) > best_k - 2:
            # Close to best, try local search to see if it can improve
            improved = local_search_swap(cand, best_k)
            if len(improved) > best_k:
                best_k = len(improved)
                if best_k >= 15:
                    break

    return best_result
