def construct():
    import time
    import random
    import sys
    from collections import defaultdict

    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit

    N = 512

    # Precompute all 4-APs and for each element, the list of (other1, other2, other3)
    # such that {x, other1, other2, other3} forms a 4-AP.
    # We'll store this as a list of lists for each element x (1-indexed).
    ap_list = [[] for _ in range(N + 1)]

    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # AP: a, b, c, e
            ap_list[a].append((b, c, e))
            ap_list[b].append((a, c, e))
            ap_list[c].append((a, b, e))
            ap_list[e].append((a, b, c))

    # Precompute a mask for each element: a tuple of (bit1, bit2, bit3) masks?
    # Or just use the list and check bits.
    # Let's optimize can_add by precomputing the bit positions.
    # ap_list_bits[x] = list of (pos1, pos2, pos3) where pos = element - 1
    ap_list_bits = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for (o1, o2, o3) in ap_list[x]:
            ap_list_bits[x].append((o1 - 1, o2 - 1, o3 - 1))

    def can_add(S_bits, x):
        """Check if x can be added to S (represented as bits) without forming a 4-AP."""
        idx = x - 1
        if (S_bits >> idx) & 1:
            return False
        for (p1, p2, p3) in ap_list_bits[x]:
            if ((S_bits >> p1) & 1) and ((S_bits >> p2) & 1) and ((S_bits >> p3) & 1):
                return False
        return True

    # Greedy construction with different orderings
    def greedy_construct(order):
        S_bits = 0
        count = 0
        for x in order:
            if can_add(S_bits, x):
                S_bits |= (1 << (x - 1))
                count += 1
        return S_bits, count

    # Generate multiple orderings
    best_S_bits = 0
    best_count = 0

    # 1. Natural order
    S_bits, count = greedy_construct(list(range(1, N + 1)))
    if count > best_count:
        best_count = count
        best_S_bits = S_bits

    # 2. Reverse order
    S_bits, count = greedy_construct(list(range(N, 0, -1)))
    if count > best_count:
        best_count = count
        best_S_bits = S_bits

    # 3. Sorted by AP degree (ascending)
    degrees = [(len(ap_list[x]), x) for x in range(1, N + 1)]
    degrees.sort()
    order = [x for (_, x) in degrees]
    S_bits, count = greedy_construct(order)
    if count > best_count:
        best_count = count
        best_S_bits = S_bits

    # 4. Sorted by AP degree (descending)
    order = [x for (_, x) in reversed(degrees)]
    S_bits, count = greedy_construct(order)
    if count > best_count:
        best_count = count
        best_S_bits = S_bits

    # 5. Randomized orderings
    random.seed(42)
    for i in range(200):
        if time.time() > deadline - 10.0:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_bits, count = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # Local search: 1-2 exchange
    # State: S_bits, count
    # Move: Remove x from S, add y (if valid). If count increases or stays same, accept.
    # Actually, we want to maximize count. So if we remove x and add y, count stays same.
    # But if removing x allows adding z, then count increases.
    # Let's do: Try to add any valid element. If none, try swapping.

    S_bits = best_S_bits
    count = best_count
    S_list = [i + 1 for i in range(N) if (S_bits >> i) & 1]
    all_elements = list(range(1, N + 1))

    # First, try to add any valid element
    for x in all_elements:
        if can_add(S_bits, x):
            S_bits |= (1 << (x - 1))
            S_list.append(x)
            count += 1

    # Now try swaps: remove x, add y
    # We'll iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try to add any valid element first
        for x in all_elements:
            if can_add(S_bits, x):
                S_bits |= (1 << (x - 1))
                S_list.append(x)
                count += 1
                improved = True
                break  # Restart search for more additions

        if not improved:
            # Try swaps
            for i in range(len(S_list)):
                if time.time() > deadline - 5.0:
                    break
                x = S_list[i]
                # Remove x
                S_bits ^= (1 << (x - 1))
                # Try to add any valid element
                for y in all_elements:
                    if y == x:
                        continue
                    if can_add(S_bits, y):
                        S_bits |= (1 << (y - 1))
                        S_list[i] = y
                        improved = True
                        break
                # Revert if no improvement
                if not improved:
                    S_bits |= (1 << (x - 1))
                else:
                    break  # Restart

    # Update best
    if count > best_count:
        best_count = count
        best_S_bits = S_bits

    # Final polish: try to add any valid element
    for x in all_elements:
        if time.time() > deadline - 1.0:
            break
        if can_add(S_bits, x):
            S_bits |= (1 << (x - 1))
            count += 1

    # Convert to list
    result = [i + 1 for i in range(N) if (S_bits >> i) & 1]
    return sorted(result)
