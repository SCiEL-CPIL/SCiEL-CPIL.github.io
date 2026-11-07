"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result (edges as cliques) for remaining
            # But we must return a valid list of 3.
            # If we run out of time, we return what we have?
            # The prompt says "return the best result found".
            # If we don't finish, we should probably return the seed result to be safe?
            # Or just return empty? No, must be valid.
            # Let's assume it finishes. The algorithm is very fast.
            pass

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency for V2 -> V1
        # V1 = 0..k-1, V2 = k..n-1
        adj_v2_to_v1 = {v: set() for v in range(k, n)}

        for u, v in edges:
            if u < k and v >= k:
                adj_v2_to_v1[v].add(u)
            elif u >= k and v < k:
                adj_v2_to_v1[u].add(v)
            # V1-V1 edges are implied by the complete graph structure, 
            # but we need to know which ones are "covered" by V2 cliques.

        cliques = []
        covered_v1_edges = set()

        # Step 1: Form cliques for each v in V2 with degree >= 1
        for v in range(k, n):
            neighbors = adj_v2_to_v1[v]
            if len(neighbors) == 0:
                continue
            
            clique = [v] + list(neighbors)
            cliques.append(clique)
            
            # Mark V1 edges covered by this clique
            neigh_list = list(neighbors)
            for i in range(len(neigh_list)):
                for j in range(i + 1, len(neigh_list)):
                    covered_v1_edges.add((neigh_list[i], neigh_list[j]))

        # Step 2: Identify remaining edges in V1
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_v1_edges:
                    remaining_edges.add((u, w))

        # Step 3: Cover remaining edges with cliques
        while remaining_edges:
            if time.time() > deadline:
                # Emergency: add remaining edges as individual cliques
                for u, w in remaining_edges:
                    cliques.append([u, w])
                break

            # Pick an edge to start a clique
            u, w = next(iter(remaining_edges))
            
            # Build a maximal clique containing (u, w) using only remaining_edges
            C = {u, w}
            
            # Find candidates that connect to both u and w in remaining_edges
            candidates = set()
            for x in range(k):
                if x == u or x == w:
                    continue
                e_xu = (min(x, u), max(x, u))
                e_xw = (min(x, w), max(x, w))
                if e_xu in remaining_edges and e_xw in remaining_edges:
                    candidates.add(x)
            
            # Greedily expand C
            while candidates:
                # Pick a candidate
                x = next(iter(candidates))
                C.add(x)
                
                # Update candidates: must connect to x
                new_candidates = set()
                for y in candidates:
                    if y == x:
                        continue
                    e_xy = (min(x, y), max(x, y))
                    if e_xy in remaining_edges:
                        new_candidates.add(y)
                candidates = new_candidates

            cliques.append(list(C))
            
            # Remove edges in C from remaining_edges
            c_list = list(C)
            for i in range(len(c_list)):
                for j in range(i + 1, len(c_list)):
                    e = (min(c_list[i], c_list[j]), max(c_list[i], c_list[j]))
                    if e in remaining_edges:
                        remaining_edges.remove(e)

        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
