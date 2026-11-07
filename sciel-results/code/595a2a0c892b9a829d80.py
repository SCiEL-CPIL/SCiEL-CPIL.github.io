"""Optimized exhaustive search with multi-start local search to maximize distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency as integer bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    def compute_distinct_degrees(combo):
        """Compute number of distinct degrees in induced subgraph on combo."""
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = set()
        for v in combo:
            degs.add(popcount(adj[v] & mask))
        return len(degs)

    vertices = list(range(n))
    best_count = -1
    best_vertices = list(range(minv))
    all_results = []  # (count, vertices) for multi-start

    # Phase 1: Complete exhaustive search over all subsets of size >= minv
    # For n=17, this is only ~63k combinations, very fast
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            c = compute_distinct_degrees(combo)
            if c > best_count:
                best_count = c
                best_vertices = list(combo)
            all_results.append((c, list(combo)))

    # Phase 2: Multi-start local search from top candidates
    if time.time() < deadline - 5.0:
        # Sort by count descending, take top candidates for local search
        all_results.sort(key=lambda x: -x[0])
        
        # Take unique top candidates (avoid duplicates)
        seen = set()
        candidates = []
        for c, verts in all_results:
            key = tuple(sorted(verts))
            if key not in seen:
                seen.add(key)
                candidates.append((c, verts))
                if len(candidates) >= 20:
                    break
        
        for start_count, start_verts in candidates:
            if time.time() > deadline - 5.0:
                break
            
            current = set(start_verts)
            current_count = start_count
            
            # Local search: try all single swaps, adds, removes
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                
                # Try 1-for-1 swaps
                for v_in in range(n):
                    if v_in in current:
                        continue
                    for v_out in current:
                        new_set = current - {v_out} | {v_in}
                        if len(new_set) < minv:
                            continue
                        c = compute_distinct_degrees(tuple(new_set))
                        if c > current_count:
                            current_count = c
                            current = new_set
                            improved = True
                            break
                    if improved:
                        break
                
                # Try adding one vertex
                if not improved and len(current) < n:
                    for v_in in range(n):
                        if v_in in current:
                            continue
                        new_set = current | {v_in}
                        c = compute_distinct_degrees(tuple(new_set))
                        if c > current_count:
                            current_count = c
                            current = new_set
                            improved = True
                            break
                
                # Try removing one vertex (if size > minv)
                if not improved and len(current) > minv:
                    for v_out in current:
                        new_set = current - {v_out}
                        c = compute_distinct_degrees(tuple(new_set))
                        if c > current_count:
                            current_count = c
                            current = new_set
                            improved = True
                            break
                
                # Try 2-for-2 swaps
                if not improved and len(current) >= minv + 1:
                    in_candidates = [v for v in range(n) if v not in current]
                    out_candidates = list(current)
                    if len(in_candidates) >= 2 and len(out_candidates) >= 2:
                        for i1, i2 in itertools.combinations(in_candidates, 2):
                            if time.time() > deadline - 5.0:
                                break
                            for o1, o2 in itertools.combinations(out_candidates, 2):
                                new_set = current - {o1, o2} | {i1, i2}
                                if len(new_set) < minv:
                                    continue
                                c = compute_distinct_degrees(tuple(new_set))
                                if c > current_count:
                                    current_count = c
                                    current = new_set
                                    improved = True
                                    break
                            if improved:
                                break
                
                if current_count > best_count:
                    best_count = current_count
                    best_vertices = list(current)

    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
