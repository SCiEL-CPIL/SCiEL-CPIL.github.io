"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    n_max = 128

    # Precompute conflict pairs: two elements conflict if their sum is also in [1, 128]
    # Actually, we need a different approach. Let me think about this more carefully.
    #
    # The constraint: no three distinct a, b, c in S such that a+b, a+c, b+c are all in S.
    # This is equivalent to: for any pair (a,b) in S with a+b in S, there should be no
    # third element c in S such that both a+c and b+c are in S.
    #
    # Strategy: Use a greedy approach with bit-parallel validity checking.
    # Represent S as a bitset over 1..128. For a candidate x to add, check if adding x
    # creates a forbidden triple.
    #
    # A forbidden triple is created when we add x if there exist a, b in S (a != b, a != x, b != x)
    # such that:
    #   - a+b in S (or a+b == x)
    #   - a+x in S (or a+x == x, impossible since a>0)
    #   - b+x in S (or b+x == x, impossible since b>0)
    #
    # Wait, let me re-read: "a+b, a+c and b+c are also selected"
    # So if we add c=x, we need to check: for any a,b in S\{x}, if a+b in S, a+x in S, and b+x in S,
    # then we have a violation. But a+x and b+x must be in S (the current set before adding x, or after?).
    # Actually, the set S includes x after we add it. So a+x could be x itself? No, a>0 so a+x > x.
    # Similarly b+x > x. So a+x and b+x must be in S (which now includes x, but a+x > x so they can't be x).
    # So a+x and b+x must be in the original S (before adding x).
    # And a+b must be in S (after adding x), so a+b could be x or in original S.
    #
    # So the check for adding x to set S:
    #   For all a, b in S with a < b, a != x, b != x:
    #     if (a+b in S or a+b == x) and (a+x in S) and (b+x in S):
    #       violation
    #
    # This is still O(n^2) per check. Let me think of a faster way.
    #
    # Alternative: Precompute for each pair (a,b) whether they form a "conflict" with some c.
    # Actually, let's just use a smarter greedy with random restarts and a fast validity check.

    def _is_valid_bitmask(mask):
        """Check if bitmask (representing set of elements 1..128) is valid."""
        # Extract elements
        elems = []
        for i in range(1, 129):
            if mask & (1 << (i - 1)):
                elems.append(i)
        n = len(elems)
        elem_set = set(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                ab = a + b
                if ab not in elem_set:
                    continue
                for k in range(j + 1, n):
                    c = elems[k]
                    if a + c in elem_set and b + c in elem_set:
                        return False
        return True

    def _count_violations_bitmask(mask):
        """Count violations in bitmask."""
        elems = []
        for i in range(1, 129):
            if mask & (1 << (i - 1)):
                elems.append(i)
        n = len(elems)
        elem_set = set(elems)
        count = 0
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                ab = a + b
                if ab not in elem_set:
                    continue
                for k in range(j + 1, n):
                    c = elems[k]
                    if a + c in elem_set and b + c in elem_set:
                        count += 1
        return count

    def _can_add(mask, x):
        """Check if adding x to mask keeps it valid. x is 1..128."""
        if mask & (1 << (x - 1)):
            return True  # already in set
        new_mask = mask | (1 << (x - 1))
        # Extract elements
        elems = []
        for i in range(1, 129):
            if new_mask & (1 << (i - 1)):
                elems.append(i)
        elem_set = set(elems)
        n = len(elems)
        # Check if adding x creates a violation
        # A violation is created if there exist a, b in elems\{x} such that:
        # (a+b in elem_set) and (a+x in elem_set) and (b+x in elem_set)
        # Note: a+b could be x, and a+x, b+x > x so they must be in original mask
        for i in range(n):
            a = elems[i]
            if a == x:
                continue
            for j in range(i + 1, n):
                b = elems[j]
                if b == x:
                    continue
                ab = a + b
                ax = a + x
                bx = b + x
                if ab in elem_set and ax in elem_set and bx in elem_set:
                    return False
        return True

    # Start with a known valid set: {63, ..., 128}, size 66
    best_mask = 0
    for i in range(63, 129):
        best_mask |= (1 << (i - 1))
    best_size = 66

    current_mask = best_mask

    # Greedy addition from below
    candidates = list(range(1, 63))
    random.shuffle(candidates)

    for cand in candidates:
        if time.time() > deadline - 15:
            break
        if _can_add(current_mask, cand):
            current_mask |= (1 << (cand - 1))
            if bin(current_mask).count('1') > best_size:
                best_mask = current_mask
                best_size = bin(current_mask).count('1')

    # Local search with swaps
    if time.time() < deadline - 10:
        all_nums = list(range(1, 129))
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            # Try random swaps
            elems_in = [i for i in range(1, 129) if current_mask & (1 << (i - 1))]
            elems_out = [i for i in range(1, 129) if not (current_mask & (1 << (i - 1)))]
            if not elems_in or not elems_out:
                break
            to_remove = random.choice(elems_in)
            to_add = random.choice(elems_out)
            trial_mask = (current_mask & ~(1 << (to_remove - 1))) | (1 << (to_add - 1))
            if _is_valid_bitmask(trial_mask):
                current_mask = trial_mask
                sz = bin(current_mask).count('1')
                if sz > best_size:
                    best_mask = current_mask
                    best_size = sz

    # Final greedy pass
    if time.time() < deadline - 5:
        current_mask = best_mask
        for cand in range(1, 129):
            if time.time() > deadline - 5:
                break
            if current_mask & (1 << (cand - 1)):
                continue
            if _can_add(current_mask, cand):
                current_mask |= (1 << (cand - 1))
                if bin(current_mask).count('1') > best_size:
                    best_mask = current_mask
                    best_size = bin(current_mask).count('1')

    result = [i for i in range(1, 129) if best_mask & (1 << (i - 1))]
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
