import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    # Generate all 2,3,5-smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)

    n = len(smooth)
    smooth_set = set(smooth)

    # Precompute conflict bitsets: conflict[i] has bit j set if smooth[i] and smooth[j] conflict
    # Two numbers conflict if one divides the other
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # Precompute prefix sums for pruning in DFS fallback
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + smooth[i]

    # Precompute all valid antichains of sizes 1 through 5
    # antichain_sums[k] = dict mapping sum -> list of terms (first found, i.e., one valid rep)
    max_precompute_size = 5
    antichain_sums = {k: {} for k in range(1, max_precompute_size + 1)}

    # Size 1: each smooth number is a valid antichain of size 1
    for i in range(n):
        s = smooth[i]
        if s <= 128:
            antichain_sums[1][s] = [s]

    # Size 2: all compatible pairs
    for i in range(n):
        vi = smooth[i]
        for j in range(i + 1, n):
            if conflict[i] & (1 << j):
                continue
            s = vi + smooth[j]
            if s <= 128:
                if s not in antichain_sums[2]:
                    antichain_sums[2][s] = [vi, smooth[j]]

    # Size 3: all compatible triples
    for i in range(n):
        vi = smooth[i]
        for j in range(i + 1, n):
            if conflict[i] & (1 << j):
                continue
            for k in range(j + 1, n):
                if (conflict[i] & (1 << k)) or (conflict[j] & (1 << k)):
                    continue
                s = vi + smooth[j] + smooth[k]
                if s <= 128:
                    if s not in antichain_sums[3]:
                        antichain_sums[3][s] = [vi, smooth[j], smooth[k]]

    # Size 4: all compatible quadruples
    for i in range(n):
        vi = smooth[i]
        for j in range(i + 1, n):
            if conflict[i] & (1 << j):
                continue
            for k in range(j + 1, n):
                if (conflict[i] & (1 << k)) or (conflict[j] & (1 << k)):
                    continue
                for m in range(k + 1, n):
                    if (conflict[i] & (1 << m)) or (conflict[j] & (1 << m)) or (conflict[k] & (1 << m)):
                        continue
                    s = vi + smooth[j] + smooth[k] + smooth[m]
                    if s <= 128:
                        if s not in antichain_sums[4]:
                            antichain_sums[4][s] = [vi, smooth[j], smooth[k], smooth[m]]

    # Size 5: all compatible quintuples (only for targets not yet covered)
    # First, identify which targets still need coverage
    targets_needing_5 = set()
    for target in range(1, 129):
        if target not in antichain_sums[1] and target not in antichain_sums[2] \
           and target not in antichain_sums[3] and target not in antichain_sums[4]:
            targets_needing_5.add(target)

    if targets_needing_5:
        for i in range(n):
            vi = smooth[i]
            for j in range(i + 1, n):
                if conflict[i] & (1 << j):
                    continue
                for k in range(j + 1, n):
                    if (conflict[i] & (1 << k)) or (conflict[j] & (1 << k)):
                        continue
                    for m in range(k + 1, n):
                        if (conflict[i] & (1 << m)) or (conflict[j] & (1 << m)) or (conflict[k] & (1 << m)):
                            continue
                        for p in range(m + 1, n):
                            if (conflict[i] & (1 << p)) or (conflict[j] & (1 << p)) or \
                               (conflict[k] & (1 << p)) or (conflict[m] & (1 << p)):
                                continue
                            s = vi + smooth[j] + smooth[k] + smooth[m] + smooth[p]
                            if s <= 128 and s in targets_needing_5:
                                if s not in antichain_sums[5]:
                                    antichain_sums[5][s] = [vi, smooth[j], smooth[k], smooth[m], smooth[p]]

    # DFS fallback for targets not covered by precomputation
    def find_min_antichain_dfs(target):
        """Find minimum-cardinality antichain summing to target using DFS."""
        for k in range(1, 31):
            if time.time() > deadline:
                return None

            def dfs(start_idx, remaining_sum, remaining_k, current_mask, path):
                if time.time() > deadline:
                    return None

                if remaining_k == 0:
                    return path[:] if remaining_sum == 0 else None

                if start_idx >= n:
                    return None

                if n - start_idx < remaining_k:
                    return None

                # Pruning: minimum possible sum
                min_sum = prefix_sum[start_idx + remaining_k] - prefix_sum[start_idx]
                if min_sum > remaining_sum:
                    return None

                # Pruning: maximum possible sum
                avail_count = n - start_idx
                if avail_count >= remaining_k:
                    max_start = n - remaining_k
                    max_sum = prefix_sum[n] - prefix_sum[max_start]
                else:
                    return None

                if max_sum < remaining_sum:
                    return None

                for i in range(start_idx, n):
                    val = smooth[i]
                    if val > remaining_sum:
                        break

                    if conflict[i] & current_mask:
                        continue

                    new_remaining = remaining_sum - val
                    new_k = remaining_k - 1
                    new_start = i + 1

                    if new_k > 0:
                        if n - new_start < new_k:
                            continue
                        min_possible = prefix_sum[new_start + new_k] - prefix_sum[new_start]
                        if min_possible > new_remaining:
                            continue
                        max_possible = prefix_sum[n] - prefix_sum[n - new_k]
                        if max_possible < new_remaining:
                            continue

                    new_mask = current_mask | (1 << i)
                    path.append(i)

                    res = dfs(new_start, new_remaining, new_k, new_mask, path)
                    if res is not None:
                        return res

                    path.pop()

                return None

            result = dfs(0, target, k, 0, [])
            if result is not None:
                return [smooth[i] for i in result]

        return None

    # Build solutions for all targets 1..128
    solutions = [None] * 129

    for t in range(1, 129):
        if time.time() > deadline:
            # Fallback: use [t] if t is smooth, else None
            if t in smooth_set:
                solutions[t] = [t]
            else:
                solutions[t] = None
            continue

        # Try precomputed antichains from size 1 upward
        found = None
        for k in range(1, max_precompute_size + 1):
            if t in antichain_sums[k]:
                found = antichain_sums[k][t]
                break

        if found is None:
            # Fall back to DFS search
            found = find_min_antichain_dfs(t)

        if found is not None:
            solutions[t] = found
        else:
            # Last resort fallback
            if t in smooth_set:
                solutions[t] = [t]
            else:
                solutions[t] = None

    return solutions[1:129]
