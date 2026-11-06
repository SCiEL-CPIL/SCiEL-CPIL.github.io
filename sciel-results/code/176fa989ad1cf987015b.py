import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256
    # Precompute squares up to 512
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1

    # Build adjacency list using bitmasks
    # adj[i] is a bitmask of nodes j where i+j is a perfect square
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        adj[a] = mask

    # Filter out nodes where 2a is a square (diagonal constraint)
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)

    # Restrict adjacency to valid nodes only
    for a in range(1, N + 1):
        adj[a] &= valid_mask_all

    # Greedy upper bound: repeatedly pick the node with minimum degree in the
    # remaining candidate set, add it, remove it and its neighbors.
    # This gives an upper bound on the size of any independent set in candidates.
    def greedy_bound(candidates_mask):
        """Upper bound on max independent set size in candidates_mask."""
        count = 0
        remaining = candidates_mask
        while remaining:
            # Find node with minimum degree in remaining
            min_deg = 1000000
            min_node = 0
            # Iterate through set bits
            r = remaining
            while r:
                lsb = r & (-r)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
                r ^= lsb
            if min_node == 0:
                break
            count += 1
            remaining &= ~(1 << min_node)
            remaining &= ~adj[min_node]
        return count

    # Branch and bound for maximum independent set
    # We work on the valid_nodes set
    best_size = 0
    best_mask = 0

    # Initial lower bound from greedy
    def greedy_independent_set(candidates_mask):
        """Greedy: repeatedly pick node with minimum degree, add to set."""
        result_mask = 0
        remaining = candidates_mask
        while remaining:
            min_deg = 1000000
            min_node = 0
            r = remaining
            while r:
                lsb = r & (-r)
                node = lsb.bit_length() - 1
                deg = (adj[node] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
                r ^= lsb
            if min_node == 0:
                break
            result_mask |= (1 << min_node)
            remaining &= ~(1 << min_node)
            remaining &= ~adj[min_node]
        return result_mask

    # Get initial solution
    init_mask = greedy_independent_set(valid_mask_all)
    best_size = init_mask.bit_count()
    best_mask = init_mask

    # Branch and bound
    def bnb(candidates_mask, current_mask, current_size, depth):
        nonlocal best_size, best_mask
        if time.time() - start_time > deadline - 5:
            return
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Upper bound
        ub = current_size + greedy_bound(candidates_mask)
        if ub <= best_size:
            return

        # Pick a vertex to branch on
        # Strategy: pick vertex with maximum degree in candidates
        max_deg = -1
        pivot = 0
        r = candidates_mask
        while r:
            lsb = r & (-r)
            node = lsb.bit_length() - 1
            deg = (adj[node] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = node
            r ^= lsb

        if pivot == 0:
            return

        # Branch 1: Include pivot
        new_candidates = candidates_mask & ~(1 << pivot) & ~adj[pivot]
        new_current = current_mask | (1 << pivot)
        bnb(new_candidates, new_current, current_size + 1, depth + 1)

        # Branch 2: Exclude pivot
        bnb(candidates_mask & ~(1 << pivot), current_mask, current_size, depth + 1)

    # Run B&B
    bnb(valid_mask_all, 0, 0, 0)

    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        node = lsb.bit_length() - 1
        result.append(node)
        m ^= lsb

    # Verify
    res_set = set(result)
    for a in res_set:
        if 2 * a in squares:
            # Should not happen, but safety
            return list(range(1, 2))
        for b in res_set:
            if a <= b and (a + b) in squares:
                return list(range(1, 2))

    if not result:
        result = [1]

    return result
