import time
from math import gcd

def construct():
    start = time.time()
    DEADLINE = start + 85.0

    N = 512
    LIMIT = 512

    # Precompute compatibility bitsets (integer bit operations)
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if a * b // gcd(a, b) <= LIMIT:
                compat[a] |= (1 << b)
                if a != b:
                    compat[b] |= (1 << a)

    def can_add(x, bits):
        return (bits & compat[x]) == bits

    def popcount(x):
        return bin(x).count('1')

    def bits_to_list(bits):
        return [a for a in range(1, N + 1) if (bits >> a) & 1]

    def set_to_bits(s):
        b = 0
        for a in s:
            b |= (1 << a)
        return b

    def greedy_from(bits, order):
        c = bits
        for x in order:
            if time.time() > DEADLINE:
                break
            if not (c >> x) & 1 and can_add(x, c):
                c |= (1 << x)
        return c

    # Multi-start greedy with many orderings
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

    degree = [0] * (N + 1)
    for a in range(1, N + 1):
        degree[a] = popcount(compat[a])

    orderings = [
        sorted(range(1, N + 1), key=lambda x: degree[x]),
        sorted(range(1, N + 1), key=lambda x: -degree[x]),
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (lpf(x), x)),
        sorted(range(1, N + 1), key=lambda x: (-degree[x], x)),
        sorted(range(1, N + 1), key=lambda x: (degree[x], -x)),
        sorted(range(1, N + 1), key=lambda x: (x.bit_length(), -degree[x])),
        sorted(range(1, N + 1), key=lambda x: (-x.bit_length(), x)),
        sorted(range(1, N + 1), key=lambda x: (degree[x], x.bit_length())),
        sorted(range(1, N + 1), key=lambda x: (-degree[x], -x.bit_length())),
    ]

    best_count = 0
    best_bits = 0
    for order in orderings:
        if time.time() > DEADLINE:
            break
        bits = greedy_from(0, order)
        cnt = popcount(bits)
        if cnt > best_count:
            best_count = cnt
            best_bits = bits

    # Local search with fast bitset moves
    current_bits = best_bits
    current_list = bits_to_list(current_bits)
    n_cur = len(current_list)

    improved = True
    max_rounds = 300
    round_num = 0

    while improved and time.time() < DEADLINE and round_num < max_rounds:
        improved = False
        round_num += 1
        current_list = bits_to_list(current_bits)
        n_cur = len(current_list)

        # Move type 1: 1-for-2 (remove one, add two)
        for i, remove in enumerate(current_list):
            if time.time() > DEADLINE:
                break
            remaining = current_bits & ~(1 << remove)
            cand_bits = 0
            for x in range(1, N + 1):
                if (current_bits >> x) & 1 or x == remove:
                    continue
                if can_add(x, remaining):
                    cand_bits |= (1 << x)
            if popcount(cand_bits) >= 2:
                found = False
                for c1 in range(1, N + 1):
                    if found or time.time() > DEADLINE:
                        break
                    if not (cand_bits >> c1) & 1:
                        continue
                    b1 = remaining | (1 << c1)
                    for c2 in range(c1 + 1, N + 1):
                        if found or time.time() > DEADLINE:
                            break
                        if not (cand_bits >> c2) & 1:
                            continue
                        if can_add(c2, b1):
                            current_bits = b1 | (1 << c2)
                            best_count = popcount(current_bits)
                            if current_bits != best_bits:
                                best_bits = current_bits
                            improved = True
                            found = True
                            break

        # Move type 2: 2-for-3 (remove two, add three)
        if time.time() < DEADLINE and n_cur >= 2:
            for i in range(n_cur):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, n_cur):
                    if time.time() > DEADLINE:
                        break
                    r1, r2 = current_list[i], current_list[j]
                    remaining = current_bits & ~(1 << r1) & ~(1 << r2)
                    cand_bits = 0
                    for x in range(1, N + 1):
                        if (current_bits >> x) & 1 or x == r1 or x == r2:
                            continue
                        if can_add(x, remaining):
                            cand_bits |= (1 << x)
                    if popcount(cand_bits) >= 3:
                        found = False
                        for c1 in range(1, N + 1):
                            if found or time.time() > DEADLINE:
                                break
                            if not (cand_bits >> c1) & 1:
                                continue
                            b1 = remaining | (1 << c1)
                            for c2 in range(c1 + 1, N + 1):
                                if found or time.time() > DEADLINE:
                                    break
                                if not (cand_bits >> c2) & 1:
                                    continue
                                if not can_add(c2, b1):
                                    continue
                                b2 = b1 | (1 << c2)
                                for c3 in range(c2 + 1, N + 1):
                                    if time.time() > DEADLINE:
                                        break
                                    if not (cand_bits >> c3) & 1:
                                        continue
                                    if can_add(c3, b2):
                                        current_bits = b2 | (1 << c3)
                                        best_count = popcount(current_bits)
                                        if current_bits != best_bits:
                                            best_bits = current_bits
                                        improved = True
                                        found = True
                                        break

        # Move type 3: 1-for-1 swap + greedy add (unblock)
        if time.time() < DEADLINE:
            for i, elem in enumerate(current_list):
                if time.time() > DEADLINE:
                    break
                remaining = current_bits & ~(1 << elem)
                for x in range(1, N + 1):
                    if time.time() > DEADLINE:
                        break
                    if (current_bits >> x) & 1:
                        continue
                    if not can_add(x, remaining):
                        continue
                    new_bits = remaining | (1 << x)
                    # Greedily add more
                    for y in range(1, N + 1):
                        if time.time() > DEADLINE:
                            break
                        if (new_bits >> y) & 1:
                            continue
                        if can_add(y, new_bits):
                            new_bits |= (1 << y)
                    cnt = popcount(new_bits)
                    if cnt > popcount(current_bits):
                        current_bits = new_bits
                        best_count = cnt
                        if current_bits != best_bits:
                            best_bits = current_bits
                        improved = True
                    break

        # Move type 4: 1-for-2 with exhaustive pair search (more thorough)
        if time.time() < DEADLINE and round_num % 3 == 0:
            for i, remove in enumerate(current_list):
                if time.time() > DEADLINE:
                    break
                remaining = current_bits & ~(1 << remove)
                cand_bits = 0
                for x in range(1, N + 1):
                    if (current_bits >> x) & 1 or x == remove:
                        continue
                    if can_add(x, remaining):
                        cand_bits |= (1 << x)
                if popcount(cand_bits) >= 2:
                    found = False
                    cands = [x for x in range(1, N + 1) if (cand_bits >> x) & 1]
                    for ci in range(len(cands)):
                        if found or time.time() > DEADLINE:
                            break
                        c1 = cands[ci]
                        b1 = remaining | (1 << c1)
                        for cj in range(ci + 1, len(cands)):
                            if time.time() > DEADLINE:
                                break
                            c2 = cands[cj]
                            if can_add(c2, b1):
                                current_bits = b1 | (1 << c2)
                                best_count = popcount(current_bits)
                                if current_bits != best_bits:
                                    best_bits = current_bits
                                improved = True
                                found = True
                                break

        # Move type 5: 2-for-3 with exhaustive triple search (more thorough)
        if time.time() < DEADLINE and round_num % 5 == 0 and n_cur >= 2:
            for i in range(n_cur):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, n_cur):
                    if time.time() > DEADLINE:
                        break
                    r1, r2 = current_list[i], current_list[j]
                    remaining = current_bits & ~(1 << r1) & ~(1 << r2)
                    cand_bits = 0
                    for x in range(1, N + 1):
                        if (current_bits >> x) & 1 or x == r1 or x == r2:
                            continue
                        if can_add(x, remaining):
                            cand_bits |= (1 << x)
                    if popcount(cand_bits) >= 3:
                        found = False
                        cands = [x for x in range(1, N + 1) if (cand_bits >> x) & 1]
                        for ci in range(len(cands)):
                            if found or time.time() > DEADLINE:
                                break
                            c1 = cands[ci]
                            b1 = remaining | (1 << c1)
                            for cj in range(ci + 1, len(cands)):
                                if found or time.time() > DEADLINE:
                                    break
                                c2 = cands[cj]
                                if not can_add(c2, b1):
                                    continue
                                b2 = b1 | (1 << c2)
                                for ck in range(cj + 1, len(cands)):
                                    if time.time() > DEADLINE:
                                        break
                                    c3 = cands[ck]
                                    if can_add(c3, b2):
                                        current_bits = b2 | (1 << c3)
                                        best_count = popcount(current_bits)
                                        if current_bits != best_bits:
                                            best_bits = current_bits
                                        improved = True
                                        found = True
                                        break

    # Final: try adding any remaining elements
    if time.time() < DEADLINE:
        current_bits = best_bits
        for x in range(1, N + 1):
            if time.time() > DEADLINE:
                break
            if (current_bits >> x) & 1:
                continue
            if can_add(x, current_bits):
                current_bits |= (1 << x)
        best_count = popcount(current_bits)
        best_bits = current_bits

    # Convert to sorted list
    result = []
    for a in range(1, N + 1):
        if (best_bits >> a) & 1:
            result.append(a)

    if not result:
        result = [1]

    return result
