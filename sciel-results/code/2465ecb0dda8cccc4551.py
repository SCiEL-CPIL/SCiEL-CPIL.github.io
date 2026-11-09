"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, k, edges):
        if not edges:
            return []
        edge_set = set(edges)
        cliques = []
        while edge_set:
            # Pick the edge with the smallest min-vertex (deterministic)
            first_edge = min(edge_set)
            u, v = first_edge
            # Start clique with this edge
            clique = {u, v}
            # Greedily add vertices that connect to all in clique
            # Try vertices in order, prefer those that connect to most clique members
            candidates = set(range(n)) - clique
            # Sort candidates by how many clique members they connect to
            def conn_count(w):
                cnt = 0
                for x in clique:
                    if (min(x, w), max(x, w)) in edge_set:
                        cnt += 1
                return cnt
            # Only add if connects to ALL in clique
            added = True
            while added:
                added = False
                # Find best candidate (most connections, then smallest vertex)
                best_w = None
                best_score = -1
                for w in candidates:
                    if conn_count(w) == len(clique):
                        # w can join
                        score = (len(clique), -w)  # prefer larger clique growth, smaller vertex
                        if best_w is None or (-w) > (-best_w):
                            best_w = w
                            best_score = -w
                if best_w is not None:
                    clique.add(best_w)
                    candidates.discard(best_w)
                    added = True
            # Remove all edges within clique
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edge_set.discard((min(a, b), max(a, b)))
            cliques.append(clique_list)
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = greedy_partition(n, k, edges)
        # Convert to lists of ints
        result = [list(c) for c in cliques]
        return result

    def construct_inner():
        specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
        results = []
        for spec in specs:
            results.append(solve_instance(*spec))
        return results

    deadline = time.time() + 140.0
    try:
        return construct_inner()
    except Exception:
        # Fallback: return edge-based partition (always valid)
        results = []
        for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in sorted(edges)])
        return results
# EVOLVE-BLOCK-END
