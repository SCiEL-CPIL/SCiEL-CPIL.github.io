import time
import sys
from math import gcd
from fractions import Fraction

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)

    # Precompute compatibility using bitmasks
    compat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            if a * b // g > N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)

    weights = [Fraction(1, v) for v in vertices]

    def get_value(mask):
        s = Fraction(0, 1)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def greedy_fill(mask, order=None):
        """Greedy add vertices in given order (or by weight descending)."""
        if order is None:
            order = sorted(range(n), key=lambda i: -weights[i])
        for v in order:
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return mask

    def local_search(mask, val, time_limit):
        """Local search with remove-refill and swap moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each vertex and refilling
            m = mask
            while m and time.time() < time_limit:
                lsb = m & -m
                u_idx = lsb.bit_length() - 1
                m ^= lsb

                candidate = mask & ~lsb
                cand_val = val - weights[u_idx]

                # Greedy refill
                new_mask = greedy_fill(candidate)
                new_val = get_value(new_mask)

                if new_val > val:
                    val = new_val
                    mask = new_mask
                    improved = True
                    break

            if time.time() >= time_limit:
                break

            # Try swap moves: remove one, try adding a different one
            m = mask
            while m and time.time() < time_limit:
                lsb = m & -m
                u_idx = lsb.bit_length() - 1
                m ^= lsb

                candidate = mask & ~lsb
                cand_val = val - weights[u_idx]

                # Try adding each non-present vertex
                for v in range(n):
                    if candidate & (1 << v):
                        continue
                    if (candidate & ~compat[v]) == 0:
                        new_mask = candidate | (1 << v)
                        new_val = cand_val + weights[v]
                        if new_val > val:
                            val = new_val
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break

        return mask, val

    best_mask = 0
    best_val = Fraction(0, 1)

    # Try different starting points
    # Strategy: pick a small number as anchor, then greedily fill
    small_anchors = [2, 3, 4, 5, 6, 7, 8, 11, 13]

    for anchor in small_anchors:
        if time.time() > deadline:
            break
        idx = anchor - 2
        if idx < 0 or idx >= n:
            continue
        S = (1 << idx)
        # Greedy fill from this starting point
        S = greedy_fill(S)
        val = get_value(S)

        if val > best_val:
            best_val = val
            best_mask = S

    # Also try the original heuristic: start with 128
    idx_128 = 128 - 2
    S = (1 << idx_128)
    for v in range(130, N + 1, 2):
        idx = v - 2
        if (S & ~compat[idx]) == 0:
            S |= (1 << idx)
    for v in range(3, N + 1, 2):
        idx = v - 2
        if (S & ~compat[idx]) == 0:
            S |= (1 << idx)
    val = get_value(S)
    if val > best_val:
        best_val = val
        best_mask = S

    # Local search on the best found so far
    best_mask, best_val = local_search(best_mask, best_val, deadline)

    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(vertices[idx])
        m ^= lsb

    if not result:
        return [2]

    return result
