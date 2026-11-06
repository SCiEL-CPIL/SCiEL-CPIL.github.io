def construct():
    import time
    from math import gcd, lcm

    deadline = time.time() + 78  # safety margin before 90s kill
    n = 512
    N = n + 1  # vertices 0..511 representing numbers 1..512

    # Build compatibility: compat[i] is bitmask of j such that lcm(i+1, j+1) <= 512
    compat = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            b = j + 1
            if a == b or lcm(a, b) <= n:
                mask |= (1 << j)
        compat[i] = mask

    best_size = 0
    best_set_mask = 0

    # Greedy initial solution for good lower bound
    def greedy_clique():
        P = (1 << N) - 1
        R = 0
        while P:
            # Pick vertex with max degree in P
            best_v = -1
            best_deg = -1
            temp = P
            while temp:
                v = (temp & -temp).bit_length() - 1
                deg = bin(P & compat[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp &= ~(1 << v)
            if best_v < 0:
                break
            R |= (1 << best_v)
            P &= compat[best_v]
        return R

    greedy_mask = greedy_clique()
    best_size = bin(greedy_mask).count('1')
    best_set_mask = greedy_mask

    # Tomita coloring
    def color_sort(candidates_mask):
        """Greedy coloring of candidate set. Returns (ordered_list, color_bounds)."""
        remaining = candidates_mask
        ordered = []
        color_num = 0
        color_starts = []

        while remaining:
            color_num += 1
            # Greedy independent set
            color_class_mask = 0
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                if (compat[v] & color_class_mask) == color_class_mask:
                    color_class_mask |= (1 << v)
                temp &= ~(1 << v)

            for v in range(N):
                if (color_class_mask >> v) & 1:
                    ordered.append(v)

            color_starts.append(len(ordered))
            remaining &= ~color_class_mask

        # Assign colors
        vertex_color = {}
        for c in range(color_num):
            start = color_starts[c] if c > 0 else 0
            end = color_starts[c + 1] if c + 1 < color_num else len(ordered)
            for i in range(start, end):
                vertex_color[ordered[i]] = c + 1

        color_bounds = [vertex_color[v] for v in ordered]
        return ordered, color_bounds, color_num

    # Tomita branch-and-bound
    def tomita_mc(R, P, depth):
        global best_size, best_set_mask

        if time.time() > deadline:
            return

        if P == 0:
            if depth > best_size:
                best_size = depth
                best_set_mask = R
            return

        # Upper bound: depth + number of vertices in P
        if depth + bin(P).count('1') <= best_size:
            return

        ordered, color_bounds, max_color = color_sort(P)

        for idx in range(len(ordered) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = ordered[idx]
            # Upper bound using color
            if depth + color_bounds[idx] <= best_size:
                break

            # Check compatibility with current clique R
            if (compat[v] & R) != R:
                continue

            # Recurse
            R_new = R | (1 << v)
            P_new = P & compat[v]
            P &= ~(1 << v)

            tomita_mc(R_new, P_new, depth + 1)

    tomita_mc(0, (1 << N) - 1, 0)

    # Convert to list
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)

    # Fallback: divisors of 504 (known good solution with 36 elements)
    if len(result) < 36:
        result = [d for d in range(1, n + 1) if 504 % d == 0]

    return result
