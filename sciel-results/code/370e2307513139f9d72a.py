"""Program: minimize total number of cliques in an edge-exact clique partition of two dense random graphs."""
# EVOLVE-BLOCK-START
def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(edges):
        # edges: set of (u,v) with u<v
        uncovered = set(edges)
        cliques = []
        while uncovered:
            # Build adjacency among uncovered edges for clique finding
            # Map vertex -> set of neighbors connected by an uncovered edge
            adj = {}
            for (u, v) in uncovered:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)

            # Greedy maximal clique in the uncovered-edge graph
            # Start from vertex with highest degree, tie-break by "clique density"
            # density = (number of common neighbors among its neighbors) / degree
            best_start = None
            best_score = -1
            for vtx in adj:
                nbrs = adj[vtx]
                deg = len(nbrs)
                if deg == 0:
                    continue
                # count edges among nbrs (common neighbors pairs)
                common = 0
                nbr_list = list(nbrs)
                for i in range(len(nbr_list)):
                    a = nbr_list[i]
                    for b in nbr_list[i+1:]:
                        if b in adj.get(a, set()):
                            common += 1
                score = common / deg  # clique density ratio
                if score > best_score:
                    best_score = score
                    best_start = vtx

            if best_start is None:
                # No vertex with neighbors; pick any edge
                e = next(iter(uncovered))
                cliques.append([min(e), max(e)])
                uncovered.discard(e)
                continue

            # Greedily build clique starting from best_start
            clique = [best_start]
            # Candidate set: neighbors of all clique members connected via uncovered edges
            candidates = set(adj[best_start])
            # Also ensure candidates are mutually connected to all clique members
            # We'll iteratively add the best candidate
            while candidates:
                # Pick candidate that is connected (via uncovered edge) to all current clique members
                # and has highest "future potential" (degree within candidates that are connected to all)
                valid = []
                for c in candidates:
                    ok = True
                    for m in clique:
                        if c not in adj.get(m, set()):
                            ok = False
                            break
                    if ok:
                        valid.append(c)
                if not valid:
                    break
                # Choose the valid candidate with max number of connections to other valid candidates
                best_c = None
                best_conn = -1
                for c in valid:
                    conn = 0
                    for d in valid:
                        if d != c and d in adj.get(c, set()):
                            conn += 1
                    if conn > best_conn:
                        best_conn = conn
                        best_c = c
                clique.append(best_c)
                # Update candidates: must be connected to best_c and all clique members
                new_candidates = set()
                for c in valid:
                    if c != best_c and best_c in adj.get(c, set()):
                        # also connected to all clique (already ensured by valid, and to best_c)
                        new_candidates.add(c)
                candidates = new_candidates

            # Emit clique if size >= 2
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove all edges within clique from uncovered
                for i in range(len(clique)):
                    a = clique[i]
                    for j in range(i + 1, len(clique)):
                        b = clique[j]
                        e = (min(a, b), max(a, b))
                        uncovered.discard(e)
            else:
                # Fallback: emit a single edge
                e = next(iter(uncovered))
                cliques.append([min(e), max(e)])
                uncovered.discard(e)

        return cliques

    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = random_graph(n, p, seed)
        cliques = partition_cliques(edges)
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
