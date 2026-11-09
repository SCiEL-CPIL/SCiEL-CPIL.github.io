"""Decompose fixed Erdős-Gallai instance graphs into few cycles and edges."""

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # Build adjacency and remaining edge set
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        remaining = set(edge_set)
        pieces = []
        rng = Random(42) # Fixed seed for reproducibility of the heuristic

        def has_edge(a, b):
            e = (a, b) if a < b else (b, a)
            return e in remaining

        def remove_edge(a, b):
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)

        def rem_degree(v):
            return sum(1 for w in adj[v] if has_edge(v, w))

        def find_cycle_randomized(start, num_walks=100):
            """
            Try to find a long cycle starting and ending at 'start' 
            using randomized walks.
            Returns the best cycle found as a list of vertices, or None.
            """
            best_cycle = None
            best_len = 0
            
            for _ in range(num_walks):
                path = [start]
                onpath = {start}
                current = start
                success = False
                
                # Max length of walk is n (Hamiltonian)
                for _ in range(n):
                    # Get neighbors with remaining edges
                    nbrs = [w for w in adj[current] if has_edge(current, w)]
                    
                    if not nbrs:
                        break
                    
                    # Check if we can close the cycle
                    if start in nbrs and len(path) >= 3:
                        # We have a cycle!
                        cyc = path[:]
                        if len(cyc) > best_len:
                            best_len = len(cyc)
                            best_cycle = cyc
                        success = True
                        break
                    
                    # Choose next node
                    # Prefer nodes not on path to extend cycle
                    # If all are on path, we are stuck or can only close
                    ext = [w for w in nbrs if w not in onpath]
                    
                    if ext:
                        # Bias towards higher degree or random?
                        # Random selection from ext to explore different paths
                        current = rng.choice(ext)
                        path.append(current)
                        onpath.add(current)
                    else:
                        # All neighbors are on path.
                        # If start is in nbrs, we could have closed it in previous step 
                        # or current step if we check again? 
                        # We checked start in nbrs at the beginning of the loop.
                        # If we are here, start is NOT in nbrs (or len < 3).
                        # So we are stuck.
                        break
                        
                # If we didn't break by closing, we might have exhausted options.
                # The loop above handles the closing check at the start of each step.
                # If we exit the for loop without success, this walk failed.
                
                # Optimization: If we found a Hamiltonian cycle, stop
                if best_len == n:
                    break

            return best_cycle

        # Main loop: extract cycles greedily
        while remaining:
            # Find a vertex with degree >= 2
            candidates = [v for v in range(n) if rem_degree(v) >= 2]
            if not candidates:
                break

            # Try to find the best cycle among top candidates
            # Sort by degree descending to prioritize dense areas
            candidates.sort(key=rem_degree, reverse=True)
            
            found = False
            best_cycle = None
            best_len = 0

            # Try a subset of candidates to save time
            # In these graphs, cycles are likely anywhere, but high degree is a good start
            for start in candidates[:5]:
                cyc = find_cycle_randomized(start, num_walks=200)
                if cyc is not None and len(cyc) > best_len:
                    best_len = len(cyc)
                    best_cycle = cyc
                    if len(cyc) == n: # Hamiltonian
                        break

            if best_cycle is None:
                # No cycles found; remaining edges are single-edge pieces
                for (u, v) in sorted(remaining):
                    pieces.append([u, v])
                remaining.clear()
                break

            # Add the cycle and remove its edges
            pieces.append(best_cycle)
            for i in range(len(best_cycle)):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % len(best_cycle)]
                remove_edge(a, b)

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
    return results
