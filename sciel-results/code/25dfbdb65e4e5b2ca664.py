"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin
    best_set = set()
    best_size = 0

    # Initial guess: The largest known safe tail
    # Tail {k, ..., n} is safe if 3*k > 2*n + 2
    # 256 * 2 / 3 = 170.66, so start with 171.
    start_threshold = 171
    current_set = set(range(start_threshold, n + 1))
    best_set = set(current_set)
    best_size = len(current_set)

    # Candidate pool: numbers not in current set, sorted descending (larger candidates first,
    # since they are less constrained as 'a' and more likely to be valid)
    candidates = list(range(start_threshold - 1, 0, -1))

    # Helper to check if adding x to S is valid
    def is_valid_addition(S, x):
        S_list = sorted(S)
        # 1. x is the smallest element (a): x < b <= c, x must NOT divide b+c
        for i, b in enumerate(S_list):
            if x < b:
                for c in S_list[i:]:
                    if b <= c and (b + c) % x == 0:
                        return False

        # 2. x is the middle element (b): a < x <= c, a must NOT divide x+c
        for a in S_list:
            if a < x:
                for c in S_list:
                    if x <= c and (x + c) % a == 0:
                        return False

        # 3. x is the largest element (c): a < b <= x, a must NOT divide b+x
        for a in S_list:
            for b in S_list:
                if a < b <= x and (b + x) % a == 0:
                    return False

        return True

    # Multi-pass greedy: try all candidates, retry failed ones in subsequent passes
    # since adding new elements can change validity (though for this constraint,
    # adding elements only makes it harder, so once invalid, always invalid).
    # Actually, adding elements makes the set larger, so previously valid additions
    # remain valid, but previously invalid ones stay invalid. So single pass suffices
    # for pure additions. But let's do multiple passes with different orderings to
    # be safe and catch any edge cases.

    additions_made = 0
    no_progress_passes = 0
    max_no_progress = 5

    for pass_num in range(20):  # Up to 20 passes
        if time.time() > deadline - 5:
            break

        progress_this_pass = 0
        # Try candidates in order; for even passes, reverse order for diversity
        order = candidates[:] if pass_num % 2 == 0 else candidates[::-1]

        for x in order:
            if time.time() > deadline - 5:
                break
            if x in current_set:
                continue

            if is_valid_addition(current_set, x):
                current_set.add(x)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                additions_made += 1
                progress_this_pass += 1

        if progress_this_pass == 0:
            no_progress_passes += 1
            if no_progress_passes >= max_no_progress:
                break
        else:
            no_progress_passes = 0

    # Return the best set found
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
