"""Maximize reciprocal sum of subset of [2,256] with pairwise lcm > 256."""
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    val = [0] * n
    for i in range(n):
        val[i] = numbers[i]

    # Precompute conflict masks (bit i conflicts with bit j)
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = val[i], val[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    weights = [1.0 / val[i] for i in range(n)]

    def sum_of(mask):
        s = 0.0
        for i in range(n):
            if mask & (1 << i):
                s += weights[i]
        return s

    def can_add(i, mask):
        return (conflict[i] & mask) == 0

    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        if conflict[i] & (1 << j):
                            return False
        return True

    # --- Build initial solution: all of [129,256] (mutually compatible) ---
    base_mask = 0
    for i in range(n):
        if val[i] >= 129:
            base_mask |= (1 << i)

    # Try adding small numbers greedily to base
    def greedy_add(mask):
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)) and can_add(i, new_mask):
                new_mask |= (1 << i)
        return new_mask

    candidates = []
    candidates.append(base_mask)
    candidates.append(greedy_add(base_mask))

    # Alternative: start empty, greedy by weight
    def greedy_from_empty(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask

    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    candidates.append(greedy_from_empty(order_w))

    # Greedy by weight but prefer large numbers (to keep [129,256] core)
    order_large = list(range(n))
    order_large.sort(key=lambda i: -val[i])
    candidates.append(greedy_from_empty(order_large))

    # Mixed: large first then small
    order_mixed = order_large + order_w
    candidates.append(greedy_from_empty(order_mixed))

    best_mask = 0
    best_sum = -1.0
    for c in candidates:
        s = sum_of(c)
        if s > best_sum:
            best_sum = s
            best_mask = c

    # --- Local search ---
    def local_search(mask, max_iters=200):
        cur = mask
        cur_sum = sum_of(cur)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE:
            improved = False
            iters += 1

            # Phase 1: add all possible
            cur = greedy_add(cur)
            cur_sum = sum_of(cur)

            # Phase 2: 1-removal + refill
            for i in range(n):
                if elapsed() > DEADLINE:
                    break
                if not (cur & (1 << i)):
                    continue
                m2 = cur & ~(1 << i)
                m3 = greedy_add(m2)
                s3 = sum_of(m3)
                if s3 > cur_sum + 1e-12:
                    cur = m3
                    cur_sum = s3
                    improved = True
                    break

            # Phase 3: 1-swap (remove i, add best j)
            if not improved and elapsed() < DEADLINE:
                for i in range(n):
                    if elapsed() > DEADLINE:
                        break
                    if not (cur & (1 << i)):
                        continue
                    m2 = cur & ~(1 << i)
                    best_j = -1
                    best_gain = 0.0
                    for j in range(n):
                        if (cur & (1 << j)):
                            continue
                        if can_add(j, m2):
                            gain = weights[j] - weights[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                    if best_j >= 0 and best_gain > 1e-12:
                        cur = m2 | (1 << best_j)
                        cur_sum += best_gain
                        improved = True
                        break

            # Phase 4: 2-swap (remove two, add two)
            if not improved and iters < max_iters and elapsed() < DEADLINE:
                found = False
                for i in range(n):
                    if elapsed() > DEADLINE:
                        break
                    if not (cur & (1 << i)):
                        continue
                    for k in range(n):
                        if elapsed() > DEADLINE:
                            break
                        if k <= i:
                            continue
                        if not (cur & (1 << k)):
                            continue
                        m2 = cur & ~(1 << i) & ~(1 << k)
                        # try adding two
                        best_pair = None
                        best_gain = 0.0
                        for j in range(n):
                            if (cur & (1 << j)):
                                continue
                            if not can_add(j, m2):
                                continue
                            for jj in range(n):
                                if (cur & (1 << jj)):
                                    continue
                                if jj == j:
                                    continue
                                if not can_add(jj, m2 | (1 << j)):
                                    continue
                                gain = weights[j] + weights[jj] - weights[i] - weights[k]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (j, jj)
                        if best_pair and best_gain > 1e-12:
                            j, jj = best_pair
                            cur = m2 | (1 << j) | (1 << jj)
                            cur_sum += best_gain
                            improved = True
                            found = True
                            break
                    if found:
                        break

        return cur, cur_sum

    # Run local search from best
    best_mask, best_sum = local_search(best_mask, max_iters=300)

    # Run local search from each candidate
    for c in candidates:
        if elapsed() > DEADLINE:
            break
        if c == best_mask:
            continue
        m, s = local_search(c, max_iters=150)
        if s > best_sum:
            best_sum = s
            best_mask = m

    # Final verification
    assert is_valid(best_mask), "Result is invalid!"

    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(val[i])
    result.sort()

    if not result:
        result = [2]

    return result
