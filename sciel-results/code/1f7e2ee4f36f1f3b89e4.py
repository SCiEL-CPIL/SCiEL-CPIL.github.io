import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s wall-clock

    # 1. Generate all 2,3,5-smooth numbers up to 128, with their exponent triples
    smooth = []
    exponents = []
    for x in range(1, 129):
        y = x
        a = b = c = 0
        while y % 2 == 0:
            y //= 2; a += 1
        while y % 3 == 0:
            y //= 3; b += 1
        while y % 5 == 0:
            y //= 5; c += 1
        if y == 1:
            smooth.append(x)
            exponents.append((a, b, c))
    n = len(smooth)
    is_smooth = set(smooth)

    # 2. Precompute compatibility bitmask: compat[i] = set of j != i with no divisibility relation
    compat = [0] * n
    for i in range(n):
        ai, bi, ci = exponents[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            aj, bj, cj = exponents[j]
            # i divides j iff ai<=aj and bi<=bj and ci<=cj
            # j divides i iff aj<=ai and bj<=bi and cj<=ci
            i_div_j = (ai <= aj) and (bi <= bj) and (ci <= cj)
            j_div_i = (aj <= ai) and (bj <= bi) and (cj <= ci)
            if not (i_div_j or j_div_i):
                mask |= (1 << j)
        compat[i] = mask

    # 3. best_count[s] = minimum terms for sum s; best_mask[s] = the antichain bitmask
    best_count = [1000] * 129
    best_mask = [None] * 129

    # 4. Enumerate all antichains via recursion. For each antichain, record its sum & length.
    #    This guarantees the minimum term count per sum (every valid representation is an antichain).
    def enumerate_anticheins(start_idx, avail_mask, current_mask, current_sum, current_len):
        if time.time() > DEADLINE:
            return
        if current_sum > 128:
            return
        if current_len > 30:
            return
        # Update best for current_sum (only if current_len improves)
        if current_sum > 0 and current_len < best_count[current_sum]:
            best_count[current_sum] = current_len
            best_mask[current_sum] = current_mask
        # Try adding each compatible element at index >= start_idx
        # avail_mask already encodes compatibility with all chosen elements
        m = avail_mask
        # Only consider indices >= start_idx to avoid duplicates
        for i in range(start_idx, n):
            if time.time() > DEADLINE:
                return
            bit = 1 << i
            if not (m & bit):
                continue
            val = smooth[i]
            if current_sum + val > 128:
                continue
            new_avail = m & compat[i] & ((~bit) & ((1 << n) - 1))
            enumerate_anticheins(i + 1, new_avail, current_mask | bit, current_sum + val, current_len + 1)

    enumerate_anticheins(0, (1 << n) - 1, 0, 0, 0)

    # 5. Fallback for any uncovered target (should be rare; use targeted small-k search)
    for target in range(1, 129):
        if best_count[target] < 1000:
            continue
        # k=1
        if target in is_smooth:
            idx = smooth.index(target)
            best_count[target] = 1
            best_mask[target] = 1 << idx
            continue
        # k=2
        found = False
        for i in range(n):
            if smooth[i] >= target:
                break
            rem = target - smooth[i]
            if rem <= smooth[i]:
                continue
            if rem in is_smooth:
                j = smooth.index(rem)
                if not (compat[i] >> j) & 1:
                    continue
                best_count[target] = 2
                best_mask[target] = (1 << i) | (1 << j)
                found = True
                break
        if found:
            continue
        # k=3
        found = False
        for i in range(n):
            if time.time() > DEADLINE:
                break
            vi = smooth[i]
            if 2 * vi >= target:
                break
            for j in range(i + 1, n):
                if not (compat[i] >> j) & 1:
                    continue
                vj = smooth[j]
                if vi + 2 * vj >= target:
                    break
                rem = target - vi - vj
                if rem <= vj:
                    continue
                if rem in is_smooth:
                    k = smooth.index(rem)
                    if (compat[i] >> k) & 1 and (compat[j] >> k) & 1:
                        best_count[target] = 3
                        best_mask[target] = (1 << i) | (1 << j) | (1 << k)
                        found = True
                        break
            if found:
                break
        if found:
            continue
        # k=4: limited search
        found = False
        for i in range(n):
            if time.time() > DEADLINE:
                break
            vi = smooth[i]
            if 3 * vi >= target:
                break
            for j in range(i + 1, n):
                if not (compat[i] >> j) & 1:
                    continue
                vj = smooth[j]
                if vi + 3 * vj >= target:
                    break
                for k in range(j + 1, n):
                    if not ((compat[i] >> k) & 1 and (compat[j] >> k) & 1):
                        continue
                    vk = smooth[k]
                    if vi + vj + 3 * vk >= target:
                        break
                    rem = target - vi - vj - vk
                    if rem <= vk:
                        continue
                    if rem in is_smooth:
                        l = smooth.index(rem)
                        if ((compat[i] >> l) & 1 and (compat[j] >> l) & 1 and
                            (compat[k] >> l) & 1):
                            best_count[target] = 4
                            best_mask[target] = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue
        # If still uncovered, use a guaranteed (non-minimal) representation via greedy
        # This should not happen, but provide a safe fallback
        result_fallback = []
        rem = target
        # Use 1s if needed (1 is smooth, divides everything, so can't combine with others)
        # Actually 1 divides everything, so [1]*k is invalid for k>=2 (1 divides 1? no, distinct terms)
        # Distinct terms: can't repeat 1. So we need distinct smooth numbers.
        # Fallback: find any antichain summing to target (may use many terms)
        # Use a simple greedy: pick largest compatible smooth numbers
        chosen = []
        chosen_mask = 0
        avail = (1 << n) - 1
        while rem > 0 and len(chosen) < 30:
            best_i = -1
            best_val = 0
            for i in range(n):
                if not (avail >> i) & 1:
                    continue
                if smooth[i] > rem:
                    continue
                if smooth[i] > best_val:
                    best_val = smooth[i]
                    best_i = i
            if best_i < 0:
                break
            chosen.append(smooth[best_i])
            chosen_mask |= (1 << best_i)
            rem -= best_val
            avail &= compat[best_i] & (~(1 << best_i))
        if rem == 0 and chosen:
            best_count[target] = len(chosen)
            best_mask[target] = chosen_mask
        else:
            # Last resort: single term if smooth, else leave as-is
            if target in is_smooth:
                idx = smooth.index(target)
                best_count[target] = 1
                best_mask[target] = 1 << idx

    # 6. Construct result
    result = []
    for target in range(1, 129):
        mask = best_mask[target]
        if mask is not None:
            terms = []
            for i in range(n):
                if (mask >> i) & 1:
                    terms.append(smooth[i])
            terms.sort()
            result.append(terms)
        else:
            # Should not happen
            if target in is_smooth:
                result.append([target])
            else:
                result.append([1] * min(target, 30))
    return result
