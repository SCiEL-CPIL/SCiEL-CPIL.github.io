import time
import random
import sys

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0

    def _validate_full(elements):
        """Full validation: distinct subset sums and no divisibility."""
        n = len(elements)
        sums = [0]
        for e in elements:
            sums += [s + e for s in sums]
        nonempty = [s for s in sums if s != 0]
        if len(nonempty) != len(set(nonempty)):
            return False
        sset = set(nonempty)
        slist = sorted(sset)
        max_s = slist[-1]
        for a in slist:
            m = a * 2
            while m <= max_s:
                if m in sset:
                    return False
                m += a
        return True

    def _divisible_conflict(existing_sums, new_sums):
        """Check if any new_sum is divisible by any existing_sum (or vice versa).
        Since new_sums > all existing_sums, we only need: new_sum % old_sum == 0.
        """
        for ns in new_sums:
            for os in existing_sums:
                if ns % os == 0:
                    return True
        return False

    def backtrack(current_set, current_sums, current_total, depth, order):
        """
        Backtracking search.
        current_set: list of chosen elements
        current_sums: set of all nonempty subset sums
        current_total: sum of all elements
        depth: current depth in search
        order: list of candidate values to try (pre-sorted or shuffled)
        """
        nonlocal best_set, best_size

        if time.time() > deadline:
            return

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

        # Try to add more elements
        for idx, x in enumerate(order):
            if time.time() > deadline:
                return

            if x <= current_total:
                continue
            if x > 256:
                break

            # Generate new subset sums
            new_sums = set()
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(x + s)

            # Check: no new_sum divides another new_sum
            ns_list = sorted(new_sums)
            valid_internal = True
            for i, a in enumerate(ns_list):
                m = a * 2
                while m <= ns_list[-1]:
                    if m in new_sums:
                        valid_internal = False
                        break
                    m += a
                if not valid_internal:
                    break

            if not valid_internal:
                continue

            # Check: no new_sum divisible by existing_sum
            if _divisible_conflict(current_sums, new_sums):
                continue

            # Recurse
            new_order = [v for v in order if v > x and v > current_total + x]
            backtrack(current_set + [x], current_sums | new_sums, current_total + x, depth + 1, new_order)

    # Strategy 1: Backtracking from various starting values
    # Use different orderings to explore different branches
    for start in range(1, 30):
        if time.time() > deadline:
            break

        # Try with natural order (smallest first)
        order = list(range(start + 1, 257))
        backtrack([start], {start}, start, 0, order)

        # Try with slightly perturbed orders
        for _ in range(3):
            if time.time() > deadline:
                break
            order = list(range(start + 1, 257))
            random.shuffle(order)
            # But ensure start is first conceptually; we pass start as initial
            # Actually reorder the candidates after start
            order = [start] + order
            # Remove start from consideration in backtrack since it's already placed
            # Let's restructure: pass the full candidate list and let backtrack pick
            order_no_start = [v for v in order if v != start]
            backtrack([start], {start}, start, 0, order_no_start)

    # Strategy 2: Try starting with larger values
    for start in range(30, 80):
        if time.time() > deadline:
            break
        order = list(range(start + 1, 257))
        backtrack([start], {start}, start, 0, order)

    # Strategy 3: Randomized full set search for validation
    for _ in range(100):
        if time.time() > deadline:
            break
        target_size = random.randint(3, 7)
        candidates = random.sample(range(1, 257), target_size)
        if _validate_full(candidates):
            if len(candidates) > best_size:
                best_size = len(candidates)
                best_set = list(candidates)

    if best_set:
        return list(best_set)
    else:
        return [1]
