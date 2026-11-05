import time
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin under 90s wall-clock limit

    n = 30
    # Common denominator
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # Integer weights: sum c_i/i = (sum c_i * (L/i)) / L
    weights = [L // i for i in range(1, n + 1)]

    half_size = n // 2  # 15
    w1 = weights[:half_size]
    w2 = weights[half_size:]

    # Enumerate all 3^15 assignments for a half.
    # For each, store (integer_sum, bitpacked_coefficients).
    # Bit i (0-based) holds coefficient of weight[i]: 0->0, 1->+1, 2->-1.
    def enumerate_half(w_list):
        m = len(w_list)
        size = 3 ** m
        sums = [0] * size
        bits = [0] * size
        # Build iteratively
        sums[0] = 0
        bits[0] = 0
        idx = 1
        # We'll generate in base-3 order: position k varies fastest.
        # For k in 0..m-1, digit d_k in {0,1,2} -> coeff (0,+1,-1)
        for k in range(m):
            w = w_list[k]
            block = 3 ** k
            for base in range(0, size, block * 3):
                for d in range(3):
                    coeff = 0 if d == 0 else (1 if d == 1 else -1)
                    delta = coeff * w
                    bitval = d
                    for j in range(block):
                        idx = base + d * block + j
                        sums[idx] = sums[base + j] + delta
                        bits[idx] = (bits[base + j] & ~(3 << (2 * k))) | (bitval << (2 * k))
        return sums, bits

    sums1, bits1 = enumerate_half(w1)
    sums2, bits2 = enumerate_half(w2)

    # Sort each half by sum, keep (sum, bit) pairs
    order1 = sorted(range(len(sums1)), key=lambda i: sums1[i])
    order2 = sorted(range(len(sums2)), key=lambda i: sums2[i])

    s1_sorted = [sums1[i] for i in order1]
    b1_sorted = [bits1[i] for i in order1]
    s2_sorted = [sums2[i] for i in order2]
    b2_sorted = [bits2[i] for i in order2]

    n1 = len(s1_sorted)
    n2 = len(s2_sorted)

    # Two-pointer: find minimal |s1 + s2| with s1+s2 != 0.
    best_abs = None
    best_i = 0
    best_j = n2 - 1
    i = 0
    j = n2 - 1
    while i < n1 and j >= 0:
        if time.time() - start_time > deadline - 5:
            break
        s_val = s1_sorted[i] + s2_sorted[j]
        if s_val != 0:
            a = abs(s_val)
            if best_abs is None or a < best_abs:
                best_abs = a
                best_i = i
                best_j = j
                if best_abs == 1:
                    break
        # Move pointer to reduce |s1+s2|
        if s_val > 0:
            j -= 1
        else:
            i += 1

    if best_abs is None:
        # Fallback: any nonzero
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients from bit-packed values
    def decode(bits, m):
        coeffs = [0] * m
        for k in range(m):
            d = (bits >> (2 * k)) & 3
            coeffs[k] = 0 if d == 0 else (1 if d == 1 else -1)
        return coeffs

    c1 = decode(b1_sorted[best_i], half_size)
    c2 = decode(b2_sorted[best_j], n - half_size)
    result = c1 + c2

    # Verify nonzero
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]

    # Local-search polish: try to reduce |total| further (single + double moves).
    cur = total
    improved = True
    while improved and time.time() - start_time < deadline - 2:
        improved = False
        for idx in range(n):
            if time.time() - start_time > deadline - 2:
                break
            old = result[idx]
            for nv in (-1, 0, 1):
                if nv == old:
                    continue
                new_sum = cur + (nv - old) * weights[idx]
                if new_sum == 0:
                    continue
                if abs(new_sum) < abs(cur):
                    result[idx] = nv
                    cur = new_sum
                    improved = True
                    break
            if improved:
                break
        # Double-move pass
        if not improved:
            for a in range(n):
                if time.time() - start_time > deadline - 2:
                    break
                for b in range(a + 1, n):
                    if time.time() - start_time > deadline - 2:
                        break
                    oa, ob = result[a], result[b]
                    for na in (-1, 0, 1):
                        for nb in (-1, 0, 1):
                            if na == oa and nb == ob:
                                continue
                            ns = cur + (na - oa) * weights[a] + (nb - ob) * weights[b]
                            if ns == 0:
                                continue
                            if abs(ns) < abs(cur):
                                result[a] = na
                                result[b] = nb
                                cur = ns
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]

    return result

# EVOLVE-BLOCK-END
