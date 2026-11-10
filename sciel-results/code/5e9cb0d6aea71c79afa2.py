"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    def check_valid(elements):
        """Check if a set of elements satisfies the constraint using sum_to_card."""
        if not elements:
            return True
        if len(set(elements)) != len(elements):
            return False
        for x in elements:
            if not isinstance(x, int) or x < 1 or x > 256:
                return False
        sum_to_card = {0: 0}
        for val in elements:
            new_entries = {}
            for s, k in sum_to_card.items():
                ns = s + val
                nk = k + 1
                if ns in new_entries:
                    if new_entries[ns] != nk:
                        return False
                else:
                    new_entries[ns] = nk
            for ns, nk in new_entries.items():
                if ns in sum_to_card and sum_to_card[ns] != nk:
                    return False
            for ns, nk in new_entries.items():
                sum_to_card[ns] = nk
        return True

    def greedy_construct(order, deadline):
        """Greedy: add each element if it doesn't violate the constraint."""
        elements = []
        sum_to_card = {0: 0}

        for val in order:
            if time.time() > deadline:
                break
            new_entries = {}
            valid = True
            for s, k in sum_to_card.items():
                ns = s + val
                nk = k + 1
                if ns in new_entries:
                    if new_entries[ns] != nk:
                        valid = False
                        break
                else:
                    new_entries[ns] = nk
            if not valid:
                continue
            for ns, nk in new_entries.items():
                if ns in sum_to_card and sum_to_card[ns] != nk:
                    valid = False
                    break
            if not valid:
                continue
            for ns, nk in new_entries.items():
                sum_to_card[ns] = nk
            elements.append(val)

        return elements

    def backtrack_search(elements, remaining, depth, deadline, max_depth):
        """Backtracking search: try to add elements from remaining in descending order."""
        if time.time() > deadline:
            return elements
        if depth >= max_depth:
            return elements
        if not remaining:
            return elements

        # Compute current sum_to_card
        sum_to_card = {0: 0}
        for val in elements:
            new_entries = {}
            for s, k in sum_to_card.items():
                ns = s + val
                nk = k + 1
                if ns not in new_entries:
                    new_entries[ns] = nk
            for ns, nk in new_entries.items():
                if ns in sum_to_card and sum_to_card[ns] != nk:
                    return elements
                sum_to_card[ns] = nk

        # Try each candidate from remaining (descending for better separation)
        for i in range(len(remaining) - 1, -1, -1):
            if time.time() > deadline:
                break
            val = remaining[i]
            new_entries = {}
            valid = True
            for s, k in sum_to_card.items():
                ns = s + val
                nk = k + 1
                if ns in new_entries:
                    if new_entries[ns] != nk:
                        valid = False
                        break
                else:
                    new_entries[ns] = nk
            if not valid:
                continue
            for ns, nk in new_entries.items():
                if ns in sum_to_card and sum_to_card[ns] != nk:
                    valid = False
                    break
            if not valid:
                continue

            # Try adding this element and recurse
            new_elements = elements + [val]
            new_remaining = remaining[:i] + remaining[i+1:]
            result = backtrack_search(new_elements, new_remaining, depth + 1, deadline, max_depth)
            if len(result) > len(elements):
                return result

        return elements

    def local_search_improve(elements, deadline):
        """Try to improve by swapping elements for alternatives."""
        elements = list(elements)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(elements)):
                if time.time() > deadline:
                    break
                old_val = elements[idx]
                for new_val in range(256, 0, -1):
                    if new_val == old_val or new_val in elements:
                        continue
                    candidate = elements[:idx] + [new_val] + elements[idx+1:]
                    if check_valid(candidate):
                        elements = candidate
                        improved = True
                        break
                if improved:
                    break
        return elements

    # Baseline: superincreasing sequence (powers of 2) - guaranteed valid
    baseline = [1, 2, 4, 8, 16, 32, 64, 128]
    best = baseline
    best_size = len(best)

    # Greedy from smallest to largest
    result = greedy_construct(range(1, 257), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)

    # Greedy from largest to smallest
    result2 = greedy_construct(range(256, 0, -1), deadline)
    if len(result2) > best_size:
        best = result2
        best_size = len(best)

    # Greedy from middle outward
    mid = list(range(129, 257)) + list(range(1, 129))
    result3 = greedy_construct(mid, deadline)
    if len(result3) > best_size:
        best = result3
        best_size = len(best)

    # Backtracking search with descending heuristic
    if time.time() < deadline - 10:
        remaining = [v for v in range(256, 0, -1) if v not in best]
        bt_result = backtrack_search(list(best), remaining, 0, deadline - 5, 15)
        if len(bt_result) > best_size:
            best = bt_result
            best_size = len(best)

    # Local search improvement
    if time.time() < deadline - 5:
        ls_result = local_search_improve(list(best), deadline)
        if len(ls_result) > best_size:
            best = ls_result
            best_size = len(best)

    # Try random orders
    random.seed(42)
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        result_r = greedy_construct(order, deadline - 5)
        if len(result_r) > best_size:
            best = result_r
            best_size = len(best)

    # Final validation
    if not check_valid(best):
        best = [1, 2, 4, 8, 16, 32, 64, 128]

    # Ensure all elements are distinct ints in 1..256
    best = [int(x) for x in best]
    return best
# EVOLVE-BLOCK-END
