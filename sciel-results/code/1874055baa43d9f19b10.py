"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 3-AP-free subset of [1, 256] maximizing the sum of reciprocals.
    Strategy:
    1. Precompute conflict table for fast validity checks.
    2. Greedy construction (smallest-first) for initial set.
    3. Aggressive local search with 1-swap moves.
    4. Multi-start with different greedy orderings.
    5. Deadline guard for 60s limit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin out of 60s

    N = 256
    MAX_VAL = N

    # Precompute conflict table:
    # conflict[x][y] = z if x, y, z form a 3-AP, else -1
    # A 3-AP is (a, b, c) with a + c = 2b, a < b < c.
    # For each pair (x, y), check if they can be part of a 3-AP.
    # If x < y:
    #   - x is first: z = 2y - x (if z > y and z <= MAX_VAL)
    #   - x is middle: z = 2x - y (if z < x and z >= 1)
    #   - x is last: z = 2y - x (if z < y and z >= 1)
    # If x > y:
    #   - y is first: z = 2x - y (if z > x and z <= MAX_VAL)
    #   - y is middle: z = 2y - x (if z < y and z >= 1)
    #   - y is last: z = 2x - y (if z < x and z >= 1)

    # Actually, let's precompute for each x the list of y values such that
    # there exists z with {x, y, z} forming a 3-AP.
    # Then, when checking if x can be added, we check if any y in S has a conflict
    # with x AND the third element z is also in S.

    # For efficiency, precompute:
    # conflicts[x] = list of (y, z) pairs such that {x, y, z} contains a 3-AP
    # Then, is_valid_addition(x) checks: for each (y, z) in conflicts[x], if y in S and z in S, invalid.

    conflicts = [[] for _ in range(MAX_VAL + 1)]
    for a in range(1, MAX_VAL + 1):
        for b in range(a + 1, MAX_VAL + 1):
            c = 2 * b - a
            if c <= MAX_VAL and c > b:
                # (a, b, c) is a 3-AP
                conflicts[a].append((b, c))
                conflicts[b].append((a, c))
                conflicts[c].append((a, b))

    def is_valid_addition(in_set, x):
        """Check if x can be added to the set represented by in_set without creating a 3-AP."""
        for y, z in conflicts[x]:
            if in_set[y] and in_set[z]:
                return False
        return True

    # Greedy construction
    def greedy_construct(order=None):
        """Greedy construction with a given order (default: 1, 2, ..., 256)."""
        if order is None:
            order = list(range(1, MAX_VAL + 1))
        in_set = [False] * (MAX_VAL + 1)
        S = set()
        for x in order:
            if is_valid_addition(in_set, x):
                in_set[x] = True
                S.add(x)
        return S, in_set

    # Local search: try to improve by removing one element and adding better ones
    def local_search(S, in_set, current_score):
        """
        Local search: repeatedly try to remove one element x from S and add one
        element y not in S such that the total reciprocal sum increases.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = sorted(S, reverse=True)  # Largest first (candidates for removal)
            not_in_S = sorted([i for i in range(1, MAX_VAL + 1) if i not in S])  # Smallest first

            for x in S_list:
                if time.time() > deadline:
                    break
                if x not in S:
                    continue

                # Try removing x
                in_set_temp = in_set[:]
                in_set_temp[x] = False
                S_temp = S - {x}

                # Try adding the best y (smallest first) to S_temp
                best_y = None
                best_gain = 0.0
                for y in not_in_S:
                    if y >= x:
                        break
                    if y in S:
                        continue
                    if is_valid_addition(in_set_temp, y):
                        gain = 1.0 / y - 1.0 / x
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y

                if best_y is not None:
                    # Apply the swap
                    S = S_temp | {best_y}
                    in_set = in_set_temp
                    in_set[best_y] = True
                    current_score += best_gain
                    improved = True
                    break  # Restart the outer loop

        return S, in_set, current_score

    # Multi-start local search
    best_S = None
    best_score = 0.0
    best_in_set = None

    # Start 1: Greedy from smallest
    S, in_set = greedy_construct()
    current_score = sum(1.0 / x for x in S)
    S, in_set, current_score = local_search(S, in_set, current_score)
    if current_score > best_score:
        best_score = current_score
        best_S = S.copy()
        best_in_set = in_set[:]

    # Start 2: Greedy from largest
    if time.time() < deadline:
        S, in_set = greedy_construct(order=list(range(MAX_VAL, 0, -1)))
        current_score = sum(1.0 / x for x in S)
        S, in_set, current_score = local_search(S, in_set, current_score)
        if current_score > best_score:
            best_score = current_score
            best_S = S.copy()
            best_in_set = in_set[:]

    # Start 3: Greedy with random perturbations
    import random
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, MAX_VAL + 1))
        # Keep first 16 in order, shuffle the rest
        for i in range(16, MAX_VAL):
            j = random.randint(i, MAX_VAL - 1)
            order[i], order[j] = order[j], order[i]
        S, in_set = greedy_construct(order=order)
        current_score = sum(1.0 / x for x in S)
        S, in_set, current_score = local_search(S, in_set, current_score)
        if current_score > best_score:
            best_score = current_score
            best_S = S.copy()
            best_in_set = in_set[:]

    # Final polish: try adding any valid element
    if best_S is not None and time.time() < deadline:
        in_set = best_in_set[:]
        for y in range(1, MAX_VAL + 1):
            if time.time() > deadline:
                break
            if y not in best_S:
                if is_valid_addition(in_set, y):
                    best_S.add(y)
                    in_set[y] = True
                    best_score += 1.0 / y

    # Final polish: try more swaps
    if best_S is not None and time.time() < deadline:
        S = best_S.copy()
        in_set = best_in_set[:]
        current_score = best_score
        S, in_set, current_score = local_search(S, in_set, current_score)
        if current_score > best_score:
            best_score = current_score
            best_S = S.copy()
            best_in_set = in_set[:]

    return list(best_S)

# EVOLVE-BLOCK-END
