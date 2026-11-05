import time
import sys
import math
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 90s limit

    # 1. Filter valid nodes (2*v not square)
    valid_nodes = []
    for v in range(1, 257):
        s = 2 * v
        r = int(s**0.5)
        if r*r != s and (r+1)*(r+1) != s:
            valid_nodes.append(v)

    n = len(valid_nodes)
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}

    # 2. Build conflict graph
    conflict = [0] * n

    squares = set()
    k = 2
    while k*k <= 512:
        squares.add(k*k)
        k += 1

    for i in range(n):
        vi = valid_nodes[i]
        for j in range(i+1, n):
            vj = valid_nodes[j]
            if vi + vj in squares:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # 3. Maximum matching on induced subgraph (for Lovasz-type bound)
    # Returns size of maximum matching in the subgraph induced by 'cands'
    def max_matching_size(cands):
        # Simple augmenting path matching for the subgraph
        # match[v] = u means v is matched to u
        match = {}
        remaining = cands

        # Get list of vertices
        verts = []
        m = cands
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb

        def augment(v, visited):
            # Try to find augmenting path from v
            neighbors = conflict[v] & cands
            m2 = neighbors
            while m2:
                lsb = m2 & -m2
                u = lsb.bit_length() - 1
                m2 ^= lsb
                if u in visited:
                    continue
                visited.add(u)
                if u not in match or augment(match[u], visited):
                    match[v] = u
                    match[u] = v
                    return True
            return False

        count = 0
        for v in verts:
            if v not in match:
                if augment(v, {v}):
                    count += 1

        return count

    # 4. Upper bound using matching: alpha(G) <= n_cands - matching_size
    def get_bound(cands):
        n_cands = cands.bit_count()
        if n_cands == 0:
            return 0
        ms = max_matching_size(cands)
        return n_cands - ms

    # 5. Greedy lower bound with modular seeding
    def greedy_lower_bound(nodes_mask):
        size = 0
        mask = nodes_mask
        while mask:
            # Pick node with min degree in mask
            min_deg = n + 1
            min_node = -1
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = (conflict[idx] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_node = idx
                m ^= lsb
            size += 1
            mask &= ~(conflict[min_node] | (1 << min_node))
        return size

    initial_cands = (1 << n) - 1
    best_size = greedy_lower_bound(initial_cands)
    best_mask = 0

    # Build initial solution mask via greedy
    def build_greedy_mask(nodes_mask):
        mask = nodes_mask
        result_mask = 0
        while mask:
            min_deg = n + 1
            min_node = -1
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = (conflict[idx] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_node = idx
                m ^= lsb
            result_mask |= (1 << min_node)
            mask &= ~(conflict[min_node] | (1 << min_node))
        return result_mask

    best_mask = build_greedy_mask(initial_cands)

    # 6. Branch and Bound
    def solve(candidates, current_size, current_mask):
        nonlocal best_size, best_mask

        if time.time() - start_time > deadline:
            return

        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        bound = get_bound(candidates)
        if current_size + bound <= best_size:
            return

        # Pick a vertex to branch on: max degree in candidates
        max_deg = -1
        v = -1
        m = candidates
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (conflict[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            m ^= lsb

        if v == -1:
            return

        # Branch 1: Include v
        new_cands_include = candidates & ~(conflict[v] | (1 << v))
        solve(new_cands_include, current_size + 1, current_mask | (1 << v))

        # Branch 2: Exclude v
        new_cands_exclude = candidates & ~(1 << v)
        solve(new_cands_exclude, current_size, current_mask)

    solve(initial_cands, 0, 0)

    # Reconstruct set from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])

    if not result:
        return list(range(1, 257, 3))

    return result
