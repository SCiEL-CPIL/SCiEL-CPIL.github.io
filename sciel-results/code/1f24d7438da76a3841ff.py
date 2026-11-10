import time
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}

    def is_valid_set(S):
        """Check if a set S is reciprocal-subsum-free."""
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True

        for x in S_list:
            target = recip[x]
            others = [recip[y] for y in S_list if y != x]
            others.sort(reverse=True)
            total_sum = sum(others)
            if total_sum < target:
                continue

            # Meet-in-the-middle for subset sum
            # Check if target can be formed from others
            if _can_make_sum(target, others):
                return False
        return True

    def _can_make_sum(target, others):
        """Check if target can be formed as sum of a subset of others."""
        n = len(others)
        if n == 0:
            return target == 0

        # If target is 0, yes
        if target == 0:
            return True

        # Quick prune: if sum of all < target, no
        if sum(others) < target:
            return False

        # Meet-in-the-middle
        mid = n // 2
        left = others[:mid]
        right = others[mid:]

        # Generate all subset sums for left
        left_sums = set()
        left_sums.add(Fraction(0))
        for val in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + val)
            left_sums.update(new_sums)

        # Generate all subset sums for right and check
        right_sums = set()
        right_sums.add(Fraction(0))
        for val in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + val)
            right_sums.update(new_sums)

        # Check if any pair sums to target
        for s in left_sums:
            if (target - s) in right_sums:
                return True
        return False

    # Start with a known good seed
    best_set = list(range(13, 25))  # [13, 14, ..., 24], size 12
    best_size = len(best_set)

    # Try to add elements one by one
    elements_to_try = list(range(1, 13))
    elements_to_try.sort(key=lambda x: -recip[x])  # Try larger elements first (smaller reciprocals)

    # Local search: try to add elements
    current_set = set(best_set)

    # Greedy add
    for x in elements_to_try:
        if time.time() > deadline:
            break
        new_set = current_set | {x}
        if is_valid_set(new_set):
            current_set = new_set

    if len(current_set) > best_size:
        best_set = list(current_set)
        best_size = len(best_set)

    # Try removing elements to free up space for others
    # This is a simple local search
    found_improvement = True
    while found_improvement and time.time() < deadline:
        found_improvement = False
        # Try to remove one element and add another
        for x in list(current_set):
            if time.time() > deadline:
                break
            for y in range(1, 25):
                if y in current_set:
                    continue
                new_set = (current_set - {x}) | {y}
                if is_valid_set(new_set):
                    current_set = new_set
                    found_improvement = True
                    break
            if found_improvement:
                break
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)

    # Try to add more elements greedily
    remaining = [x for x in range(1, 25) if x not in current_set]
    remaining.sort(key=lambda x: -recip[x])

    for x in remaining:
        if time.time() > deadline:
            break
        new_set = current_set | {x}
        if is_valid_set(new_set):
            current_set = new_set

    if len(current_set) > best_size:
        best_set = list(current_set)
        best_size = len(current_set)

    return best_set
