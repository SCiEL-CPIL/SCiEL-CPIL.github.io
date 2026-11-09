import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256

    # Precompute 3-AP and 4-AP structures using bitmasks for speed
    # A 3-AP is (a, b, c) where a < b < c and b-a = c-b, i.e., a+c = 2b
    # A 4-AP is (a, b, c, d) where a < b < c < d and b-a = c-b = d-c

    # For each element x, precompute which pairs of other elements form a 3-AP with x
    # x as first: x, x+d, x+2d -> need both x+d and x+2d in set
    # x as second: x-d, x, x+d -> need both x-d and x+d in set
    # x as third: x-2d, x-d, x -> need both x-2d and x-d in set

    # Precompute for each x, list of (other1, other2) pairs that form 3-AP with x
    # where adding x completes the AP
    three_ap_pairs = [[] for _ in range(N + 1)]  # three_ap_pairs[x] = list of (y, z) where {x,y,z} is a 3-AP and x is the "new" element
    for x in range(1, N + 1):
        # x as first: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if z <= N:
                three_ap_pairs[x].append((y, z))
        # x as second: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))
        # x as third: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))

    # Precompute for each x, which triples of other elements form a 4-AP with x
    # x as 1st: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d all in set
    # x as 2nd: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d all in set
    # x as 3rd: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d all in set
    # x as 4th: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d all in set

    four_ap_triples = [[] for _ in range(N + 1)]  # four_ap_triples[x] = list of (y, z, w) where {x,y,z,w} is a 4-AP
    for x in range(1, N + 1):
        # x as 1st
        for d in range(1, (N - x) // 3 + 1):
            y = x + d
            z = x + 2 * d
            w = x + 3 * d
            if w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            y = x - d
            z = x + d
            w = x + 2 * d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            y = x - 2 * d
            z = x - d
            w = x + d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 4th
        for d in range(1, (x - 1) // 3 + 1):
            y = x - 3 * d
            z = x - 2 * d
            w = x - d
            if y >= 1:
                four_ap_triples[x].append((y, z, w))

    def count_3aps_bitmask(mask):
        """Count 3-APs using bitmask representation."""
        count = 0
        for x in range(1, N + 1):
            if not (mask >> x) & 1:
                continue
            for (y, z) in three_ap_pairs[x]:
                if (mask >> y) & 1 and (mask >> z) & 1:
                    count += 1
        return count

    def has_4ap_bitmask(mask):
        """Check for 4-AP using bitmask."""
        for x in range(1, N + 1):
            if not (mask >> x) & 1:
                continue
            for (y, z, w) in four_ap_triples[x]:
                if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                    return True
        return False

    def creates_4ap_bitmask(x, mask):
        """Check if adding x creates a 4-AP."""
        for (y, z, w) in four_ap_triples[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return True
        return False

    def gain_3ap_bitmask(x, mask):
        """Count 3-APs completed by adding x."""
        gain = 0
        for (y, z) in three_ap_pairs[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                gain += 1
        return gain

    def greedy_conflict_graph(temperature=0.0, seed=None):
        """
        Weighted greedy on conflict graph:
        - Vertices: elements 1..N
        - Edge between x,y if they participate in a common 4-AP (conflict)
        - At each step, pick vertex with max (3-AP gain) - penalty * (degree in remaining conflict graph)
        """
        if seed is not None:
            random.seed(seed)

        mask = 0
        remaining = set(range(1, N + 1))

        while remaining and time.time() < deadline - 5:
            best_x = None
            best_score = -1e18

            for x in remaining:
                if creates_4ap_bitmask(x, mask):
                    continue

                g = gain_3ap_bitmask(x, mask)

                # Conflict penalty: count how many remaining elements would conflict with x
                # (i.e., adding x would create a 4-AP involving that element)
                # This is expensive, so use a cheaper proxy:
                # Count remaining elements y such that y and x share a 4-AP structure
                conflict_count = 0
                for (y, z, w) in four_ap_triples[x]:
                    # If y, z, w are all in remaining (not yet added), they could form 4-AP with x
                    # But we only care if they're already in mask (which we check via creates_4ap_bitmask)
                    # For penalty: count how many of {y,z,w} are in remaining
                    if y in remaining:
                        conflict_count += 1
                    if z in remaining:
                        conflict_count += 1
                    if w in remaining:
                        conflict_count += 1

                # Score: maximize 3-AP gain, penalize high conflict degree
                score = g * 10.0 - conflict_count * 0.1
                if temperature > 0:
                    score += random.gauss(0, temperature)

                if score > best_score:
                    best_score = score
                    best_x = x

            if best_x is None:
                break

            mask |= (1 << best_x)
            remaining.remove(best_x)

        return mask

    def local_search_bitmask(mask, max_iters=20):
        """Local search: try single swaps to improve 3-AP count."""
        current_count = count_3aps_bitmask(mask)
        remaining = set(range(1, N + 1)) - set(x for x in range(1, N+1) if (mask >> x) & 1)

        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break

            improved = False
            in_set = [x for x in range(1, N+1) if (mask >> x) & 1]

            # Compute loss for each element
            losses = {}
            for a in in_set:
                # Temporarily remove a
                mask_no_a = mask & ~(1 << a)
                loss = current_count - count_3aps_bitmask(mask_no_a)
                losses[a] = loss

            # Sort by loss ascending
            sorted_in = sorted(in_set, key=lambda x: losses[x])

            for a in sorted_in:
                if time.time() > deadline - 10:
                    break
                if improved:
                    break

                mask_no_a = mask & ~(1 << a)
                loss = losses[a]

                # Find best replacement
                best_b = None
                best_gain = -1
                for b in remaining:
                    if creates_4ap_bitmask(b, mask_no_a):
                        continue
                    gain = gain_3ap_bitmask(b, mask_no_a)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b

                if best_b is not None and best_gain > loss:
                    mask = mask_no_a | (1 << best_b)
                    remaining.remove(best_b)
                    remaining.add(a)
                    current_count += best_gain - loss
                    improved = True

            if not improved:
                break

        return mask, current_count

    # Main: run multiple greedy passes with different random seeds and temperatures
    best_mask = 0
    best_count = 0

    # Deterministic pass
    mask = greedy_conflict_graph(temperature=0.0, seed=42)
    if not has_4ap_bitmask(mask):
        cnt = count_3aps_bitmask(mask)
        if cnt > best_count:
            best_count = cnt
            best_mask = mask

    # Randomized passes
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break

        temp = 0.05 + 0.15 * (i / num_restarts)
        seed = 1000 + i * 7919
        mask = greedy_conflict_graph(temperature=temp, seed=seed)

        if has_4ap_bitmask(mask):
            # Remove problematic elements
            for x in range(1, N + 1):
                if (mask >> x) & 1:
                    if has_4ap_bitmask(mask):
                        mask &= ~(1 << x)

        if mask == 0:
            continue

        # Local search refinement
        mask, cnt = local_search_bitmask(mask, max_iters=15)

        if not has_4ap_bitmask(mask):
            if cnt > best_count:
                best_count = cnt
                best_mask = mask

    # Final verification
    if has_4ap_bitmask(best_mask):
        for x in range(1, N + 1):
            if (best_mask >> x) & 1:
                if has_4ap_bitmask(best_mask):
                    best_mask &= ~(1 << x)

    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    if not result:
        result = [1]

    return result
