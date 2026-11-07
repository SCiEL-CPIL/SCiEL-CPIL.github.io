def construct():
    import time
    start = time.time()
    DEADLINE = 85.0

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    idx = {v: i for i, v in enumerate(numbers)}

    # Precompute conflict masks: for each pair (i,j), which k would create a violation
    # k conflicts with pair (i,j) if:
    #   numbers[k] divides numbers[i]*numbers[j]
    #   numbers[i] divides numbers[k]*numbers[j]
    #   numbers[j] divides numbers[i]*numbers[k]
    conflict_mask = [[0] * n for _ in range(n)]

    for i in range(n):
        vi = numbers[i]
        for j in range(i + 1, n):
            vj = numbers[j]
            prod_ij = vi * vj
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                vk = numbers[k]
                if prod_ij % vk == 0 or (vk * vj) % vi == 0 or (vi * vk) % vj == 0:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask

    def is_valid_set(s):
        """Check if set s (of indices) is valid using conflict masks - O(|s|^2)"""
        s_list = list(s)
        m = len(s_list)
        for ai in range(m):
            a = s_list[ai]
            for bi in range(ai + 1, m):
                b = s_list[bi]
                # Check if any c in s conflicts with pair (a,b)
                cm = conflict_mask[a][b]
                # Remove a and b from consideration
                for ci in range(m):
                    c = s_list[ci]
                    if c != a and c != b and (cm >> c) & 1:
                        return False
        return True

    def greedy(order):
        """Greedy add in given order, checking validity via conflict masks"""
        included = set()
        for ia in order:
            # Check if adding ia conflicts with any pair in included
            conflict = False
            inc_list = list(included)
            for bi in range(len(inc_list)):
                b = inc_list[bi]
                for ci in range(bi + 1, len(inc_list)):
                    c = inc_list[ci]
                    if (conflict_mask[b][c] >> ia) & 1:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                included.add(ia)
        return included

    def local_search(s, deadline):
        """Improve set s via 1-for-1 swaps and 2-for-1 moves using fast validation"""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            s_list = list(s)
            excluded = [i for i in range(n) if i not in s]

            # Try 1-for-1 swaps
            for x in excluded:
                if time.time() > deadline:
                    return s
                for y in s_list:
                    new_s = (s - {y}) | {x}
                    if is_valid_set(new_s):
                        s = new_s
                        s_list = list(s)
                        improved = True
                        break
                if improved:
                    break

            # Try 2-for-1 moves (add x, remove two y,z)
            if not improved:
                for x in excluded:
                    if time.time() > deadline:
                        return s
                    # Find pairs (y,z) in s such that removing both and adding x is valid
                    for yi in range(len(s_list)):
                        y = s_list[yi]
                        for zi in range(yi + 1, len(s_list)):
                            z = s_list[zi]
                            new_s = (s - {y, z}) | {x}
                            if len(new_s) == len(s) - 1:  # Net loss, skip
                                continue
                            if is_valid_set(new_s):
                                s = new_s
                                s_list = list(s)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            # Try 1-for-2 moves (remove x, add two y,z) - increases size
            if not improved:
                for x in list(s):
                    if time.time() > deadline:
                        return s
                    base_s = s - {x}
                    base_list = list(base_s)
                    for yi in range(len(excluded)):
                        y = excluded[yi]
                        for zi in range(yi + 1, len(excluded)):
                            z = excluded[zi]
                            new_s = base_s | {y, z}
                            if is_valid_set(new_s):
                                s = new_s
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        return s

    # Multi-start with different orderings
    best_set = set()
    best_size = 0

    orderings = [
        list(range(n - 1, -1, -1)),  # Largest to smallest
        list(range(0, n)),           # Smallest to largest
        [i for i in range(n) if numbers[i] > N // 2] + [i for i in range(n) if numbers[i] <= N // 2],
        [i for i in range(n) if numbers[i] % 2 == 0] + [i for i in range(n) if numbers[i] % 2 == 1],
    ]

    for order in orderings:
        if time.time() > DEADLINE - 30:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    # Local search on best found
    if time.time() < DEADLINE - 5:
        best_set = local_search(best_set, DEADLINE - 5)
        best_size = len(best_set)

    result = [numbers[i] for i in best_set]
    return sorted(result)
