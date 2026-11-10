import time
from math import gcd

def construct():
    start = time.time()
    DEADLINE = start + 85.0

    LIMIT = 512
    N = 512

    # Precompute compatibility bitsets
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            l = a * b // gcd(a, b)
            if l <= LIMIT:
                compat[a] |= (1 << b)
                if a != b:
                    compat[b] |= (1 << a)

    def can_add(x, current_bits):
        return (current_bits & compat[x]) == current_bits

    def bits_to_set(bits):
        s = set()
        for a in range(1, N + 1):
            if (bits >> a) & 1:
                s.add(a)
        return s

    def set_to_bits(s):
        b = 0
        for a in s:
            b |= (1 << a)
        return b

    # Compute degree (number of compatible elements) for ordering
    degree = [0] * (N + 1)
    for a in range(1, N + 1):
        degree[a] = bin(compat[a]).count('1')

    # Multiple orderings for greedy construction
    order_asc = sorted(range(1, N + 1), key=lambda x: degree[x])
    order_desc = sorted(range(1, N + 1), key=lambda x: -degree[x])
    order_num = list(range(1, N + 1))
    order_rev = list(range(N, 0, -1))
    
    def lpf(n):
        if n <= 1:
            return 1
        f = 2
        m = n
        while f * f <= m:
            while m % f == 0:
                m //= f
            f += 1
        return m if m > 1 else f
    order_lpf = sorted(range(1, N + 1), key=lambda x: (lpf(x), x))
    # By number of divisors (fewer divisors = more constrained)
    def num_divisors(n):
        cnt = 0
        for d in range(1, int(n**0.5) + 1):
            if n % d == 0:
                cnt += 1
                if d != n // d:
                    cnt += 1
        return cnt
    order_nd = sorted(range(1, N + 1), key=lambda x: num_divisors(x))

    orderings = [order_asc, order_desc, order_num, order_rev, order_lpf, order_nd]

    best_count = 0
    best_set_bits = 0

    for order in orderings:
        if time.time() > DEADLINE:
            break
        bits = 0
        count = 0
        for x in order:
            if (bits >> x) & 1:
                continue
            if can_add(x, bits):
                bits |= (1 << x)
                count += 1
        if count > best_count:
            best_count = count
            best_set_bits = bits

    # Local search: iterative improvement with multiple swap strategies
    current_set = bits_to_set(best_set_bits)
    improved = True
    max_rounds = 100
    round_num = 0

    while improved and time.time() < DEADLINE and round_num < max_rounds:
        improved = False
        round_num += 1
        current_list = sorted(current_set)
        n_cur = len(current_list)

        # Strategy A: Try removing each element and greedily adding back all possible
        for i, remove in enumerate(current_list):
            if time.time() > DEADLINE:
                break
            remaining = set(current_list)
            remaining.discard(remove)
            remaining_bits = set_to_bits(remaining)

            # Greedily add all compatible elements not in remaining
            new_bits = remaining_bits
            new_count = len(remaining)
            for x in order_asc:  # Most constrained first
                if (new_bits >> x) & 1:
                    continue
                if can_add(x, new_bits):
                    new_bits |= (1 << x)
                    new_count += 1

            if new_count > len(current_set):
                current_set = bits_to_set(new_bits)
                best_set_bits = new_bits
                best_count = new_count
                improved = True
                break  # Restart with new set

        # Strategy B: 1-for-2 swaps
        if not improved and time.time() < DEADLINE:
            for i, remove in enumerate(current_list):
                if time.time() > DEADLINE:
                    break
                remaining = set(current_list)
                remaining.discard(remove)
                remaining_bits = set_to_bits(remaining)

                candidates = []
                for x in range(1, N + 1):
                    if (best_set_bits >> x) & 1:
                        continue
                    if x == remove:
                        continue
                    if can_add(x, remaining_bits):
                        candidates.append(x)

                if len(candidates) >= 2:
                    found = False
                    for ci in range(len(candidates)):
                        if found or time.time() > DEADLINE:
                            break
                        c1 = candidates[ci]
                        bits_with_c1 = remaining_bits | (1 << c1)
                        for cj in range(ci + 1, len(candidates)):
                            if time.time() > DEADLINE:
                                break
                            c2 = candidates[cj]
                            if can_add(c2, bits_with_c1):
                                new_set = remaining | {c1, c2}
                                if len(new_set) > len(current_set):
                                    current_set = new_set
                                    best_set_bits = set_to_bits(current_set)
                                    best_count = len(current_set)
                                    improved = True
                                    found = True
                                    break
                    if improved:
                        break

        # Strategy C: 2-for-3 swaps
        if not improved and time.time() < DEADLINE and n_cur >= 2:
            for i in range(n_cur):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, n_cur):
                    if time.time() > DEADLINE:
                        break
                    r1, r2 = current_list[i], current_list[j]
                    remaining = set(current_list)
                    remaining.discard(r1)
                    remaining.discard(r2)
                    remaining_bits = set_to_bits(remaining)

                    candidates = []
                    for x in range(1, N + 1):
                        if (best_set_bits >> x) & 1:
                            continue
                        if x == r1 or x == r2:
                            continue
                        if can_add(x, remaining_bits):
                            candidates.append(x)

                    if len(candidates) >= 3:
                        found = False
                        for ci in range(len(candidates)):
                            if found or time.time() > DEADLINE:
                                break
                            c1 = candidates[ci]
                            b1 = remaining_bits | (1 << c1)
                            for cj in range(ci + 1, len(candidates)):
                                if found or time.time() > DEADLINE:
                                    break
                                c2 = candidates[cj]
                                if not can_add(c2, b1):
                                    continue
                                b2 = b1 | (1 << c2)
                                for ck in range(cj + 1, len(candidates)):
                                    if time.time() > DEADLINE:
                                        break
                                    c3 = candidates[ck]
                                    if can_add(c3, b2):
                                        new_set = remaining | {c1, c2, c3}
                                        if len(new_set) > len(current_set):
                                            current_set = new_set
                                            best_set_bits = set_to_bits(current_set)
                                            best_count = len(current_set)
                                            improved = True
                                            found = True
                                            break
                        if improved:
                            break
                if improved:
                    break

    # Final: try adding any remaining elements (multiple passes)
    if time.time() < DEADLINE:
        for _ in range(3):
            current_bits = set_to_bits(current_set)
            added = False
            for x in range(1, N + 1):
                if time.time() > DEADLINE:
                    break
                if (current_bits >> x) & 1:
                    continue
                if can_add(x, current_bits):
                    current_bits |= (1 << x)
                    current_set.add(x)
                    added = True
            if not added:
                break
        best_set_bits = set_to_bits(current_set)
        best_count = len(current_set)

    # Convert to sorted list
    result = []
    for a in range(1, N + 1):
        if (best_set_bits >> a) & 1:
            result.append(a)

    if not result:
        result = [1]

    return result
