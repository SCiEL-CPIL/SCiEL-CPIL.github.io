"""Optimized 3-AP-free set maximizer using bitset acceleration."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    N = 256

    # Precompute: for each candidate x, the set of y values that would conflict
    # If we add x, we need: for all y in S, 2x-y not in S, 2y-x not in S
    # Bitset approach: represent S as a bitset (Python int)
    # For candidate x, conflict means:
    #   exists y in S such that (2*x - y) in S  [x is middle or first/last]
    #   Actually: 3-AP (a,b,c) with a+b=2c or a+c=2b or b+c=2a
    #   If x is added, conflicts if:
    #     x is middle: a + c = 2x, a,c in S
    #     x is first: x + c = 2b, b,c in S  
    #     x is last: a + x = 2b, a,b in S

    # Precompute for each x the "conflict pairs"
    # For x to be addable to S: no (a,c) in S x S with a+c=2x, 
    #   no (b,c) in S x S with x+c=2b,
    #   no (a,b) in S x S with a+x=2b

    # Bitset: S as int, bit i set means (i+1) in S
    # For checking if x is valid to add to S:
    #   Let S_bits be the bitset
    #   Middle: need no a in S such that (2x-a) in S, i.e., S & reverse_shift(S, 2x) == 0
    #   This is complex with bitsets. Let's use a different approach.

    # Precompute for each x: list of pairs (a, c) such that a + c = 2x (x is middle)
    # and (x, b, c) with x + c = 2b (x is first), and (a, b, x) with a + x = 2b (x is last)
    
    # For efficiency, precompute:
    # middle_conflicts[x] = list of (a, c) where a + c = 2x, a < c, both in [1,256]
    # first_conflicts[x] = list of (b, c) where x + c = 2b, x < b < c, all in [1,256]
    # last_conflicts[x] = list of (a, b) where a + x = 2b, a < b < x, all in [1,256]

    # Actually, simpler: for each x, precompute the set of "forbidden pairs"
    # If both elements of any forbidden pair are in S, then x cannot be added.

    # Let me use a cleaner bitset approach:
    # S as a Python int (bitset), bit i represents element (i+1)
    # To check if x (1-indexed) can be added:
    #   For each y in S: check if (2*x - y) in S, (2*y - x) in S
    #   This is O(|S|) per check, but with bitset we can do it faster.

    # Better: precompute for each x a bitmask of "partners"
    # When adding x, it conflicts with existing S if:
    #   exists y in S such that (2x-y) in S  => this means S intersects (2x - S)
    #   exists y in S such that (2y-x) in S  => this means S intersects (2S - x)
    #   (these two conditions cover all 3-AP cases)

    # Let me just use the O(|S|) check but optimize with numpy or precomputed tables.

    # Precompute: for each x in 1..256, and for each possible S (too large)
    # Instead: for each x, precompute the list of y values such that adding x 
    # would create a 3-AP with y and some z.
    
    # For a given x, the forbidden y values (if y is in S, then we need to check z):
    # Case 1: x is middle, y and z are ends: y + z = 2x, z = 2x - y
    #   So for each y in S, if (2x-y) in S, conflict.
    # Case 2: x is first, y is middle, z is last: x + z = 2y, z = 2y - x
    #   So for each y in S, if (2y-x) in S, conflict.
    # Case 3: x is last, y is first, z is middle: y + x = 2z, y = 2z - x
    #   So for each z in S, if (2z-x) in S, conflict. (same as case 2 with different naming)

    # So: x is valid to add iff:
    #   For all y in S: (2*x - y) not in S  AND  (2*y - x) not in S
    #   (with the constraint that the computed values are in [1,256])

    # Precompute for each x:
    #   partner1[x][y] = 2*x - y (if in range)
    #   partner2[x][y] = 2*y - x (if in range)

    # For speed, I'll precompute these as arrays.

    # partner1[x][y] = 2x - y, valid if 1 <= 2x-y <= 256
    # partner2[x][y] = 2y - x, valid if 1 <= 2y-x <= 256

    # Use 0-indexed: element i represents value (i+1)
    # x_idx = x-1, y_idx = y-1
    # partner1_val = 2*(x_idx+1) - (y_idx+1) = 2*x_idx - y_idx + 1
    # partner1_idx = partner1_val - 1 = 2*x_idx - y_idx
    # Valid if 0 <= partner1_idx <= 255

    # partner2_val = 2*(y_idx+1) - (x_idx+1) = 2*y_idx - x_idx + 1
    # partner2_idx = partner2_val - 1 = 2*y_idx - x_idx
    # Valid if 0 <= partner2_idx <= 255

    # Precompute partner1 and partner2 as 2D arrays (256x256), with -1 for invalid
    partner1 = [[-1]*256 for _ in range(256)]
    partner2 = [[-1]*256 for _ in range(256)]
    for xi in range(256):
        for yi in range(256):
            p1 = 2*xi - yi
            if 0 <= p1 <= 255:
                partner1[xi][yi] = p1
            p2 = 2*yi - xi
            if 0 <= p2 <= 255:
                partner2[xi][yi] = p2

    def is_valid_addition(S_bits, x_idx):
        """Check if x_idx can be added to set represented by S_bits."""
        # For each y in S, check if partner1[x][y] in S or partner2[x][y] in S
        # Iterate over set bits in S_bits
        temp = S_bits
        while temp:
            # Get lowest set bit
            y_idx = (temp & -temp).bit_length() - 1
            temp &= temp - 1  # clear lowest set bit
            
            p1 = partner1[x_idx][y_idx]
            if p1 >= 0:
                if S_bits & (1 << p1):
                    return False
            p2 = partner2[x_idx][y_idx]
            if p2 >= 0:
                if S_bits & (1 << p2):
                    return False
        return True

    def reciprocal_sum_from_bits(S_bits):
        """Compute exact reciprocal sum from bitset."""
        total = Fraction(0, 1)
        temp = S_bits
        while temp:
            idx = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            total += Fraction(1, idx + 1)
        return total

    def bits_to_list(S_bits):
        """Convert bitset to list of 1-indexed values."""
        result = []
        temp = S_bits
        while temp:
            idx = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            result.append(idx + 1)
        return result

    def greedy_init(order):
        """Greedy: add elements in given order if valid."""
        S_bits = 0
        for x in order:
            x_idx = x - 1
            if not (S_bits & (1 << x_idx)):  # not already in set
                if is_valid_addition(S_bits, x_idx):
                    S_bits |= (1 << x_idx)
        return S_bits

    def local_search(S_bits, max_iter=50):
        """Local search: try additions and swaps."""
        S_bits = S_bits
        best_sum = reciprocal_sum_from_bits(S_bits)
        best_S = S_bits

        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break

            improved = False

            # Try single additions (small numbers first for max reciprocal gain)
            candidates = [x for x in range(1, 257) if not (S_bits & (1 << (x-1)))]
            for cand in sorted(candidates, key=lambda x: x):  # small first
                cand_idx = cand - 1
                if is_valid_addition(S_bits, cand_idx):
                    new_S = S_bits | (1 << cand_idx)
                    new_sum = best_sum + Fraction(1, cand)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_S = new_S
                        S_bits = new_S
                        improved = True
                        break

            # Try swaps: remove large element, add small element
            if not improved:
                S_list = bits_to_list(S_bits)
                # Sort by value descending (remove largest first)
                for elem in sorted(S_list, reverse=True):
                    elem_idx = elem - 1
                    S_without = S_bits & ~(1 << elem_idx)
                    # Find smallest cand not in S_without that is valid
                    for cand in range(1, elem):  # only smaller candidates improve sum
                        if S_without & (1 << (cand-1)):
                            continue
                        if is_valid_addition(S_without, cand-1):
                            new_S = S_without | (1 << (cand-1))
                            new_sum = best_sum - Fraction(1, elem) + Fraction(1, cand)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_S = new_S
                                S_bits = new_S
                                improved = True
                            break  # Only try smallest valid cand for this elem

                    if improved:
                        break

            if not improved:
                break

        return best_S

    # Phase 1: Multiple greedy initializations
    S1 = greedy_init(list(range(1, 257)))  # ascending
    S2 = greedy_init(list(range(256, 0, -1)))  # descending
    S3 = greedy_init(sorted(range(1, 257), key=lambda x: random.random()))  # random

    best_S = max([S1, S2, S3], key=lambda s: reciprocal_sum_from_bits(s))
    best_sum = reciprocal_sum_from_bits(best_S)

    # Phase 2: Local search on best
    best_S = local_search(best_S, max_iter=30)
    best_sum = reciprocal_sum_from_bits(best_S)

    # Phase 3: Random restarts
    while (time.time() - start_time) < DEADLINE - 3:
        # Perturb and re-search
        order = list(range(1, 257))
        random.shuffle(order)
        S_new = greedy_init(order)
        S_new = local_search(S_new, max_iter=15)
        new_sum = reciprocal_sum_from_bits(S_new)
        if new_sum > best_sum:
            best_sum = new_sum
            best_S = S_new

    # Final polish
    best_S = local_search(best_S, max_iter=40)

    return bits_to_list(best_S)
# EVOLVE-BLOCK-END
