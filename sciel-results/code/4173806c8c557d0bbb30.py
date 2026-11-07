import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # All edges as frozensets for quick lookup
        all_edges = set(frozenset((u, v)) for u, v in edges)

        # Edge weights: start at 1.0, will be updated
        edge_weights = {e: 1.0 for e in all_edges}

        # Uncovered edges
        uncovered = set(all_edges)

        # Build a helper: for a given set of vertices, find all edges among them that are in uncovered
        def edges_in_vertex_set(vertices):
            vs = set(vertices)
            result = []
            for e in uncovered:
                a, b = tuple(e)
                if a in vs and b in vs:
                    result.append(e)
            return result

        # Greedy: find a maximal clique in the residual graph (using only uncovered edges)
        # that has high weight-sum
        def find_best_clique(uncovered_edges, adj, n):
            if not uncovered_edges:
                return None

            # Compute score for each edge: sum of weights of edges it could be part of
            # We'll try each edge as a seed, extend to maximal clique, compute weight sum
            # To be efficient, we'll use a heuristic: pick edge with highest combined weight
            # of its endpoints' incident uncovered edges

            # Build a map: vertex -> set of uncovered edges incident to it
            vertex_edges = {i: set() for i in range(n)}
            for e in uncovered_edges:
                a, b = tuple(e)
                vertex_edges[a].add(e)
                vertex_edges[b].add(e)

            # Score each edge by sum of weights of edges in a potential clique
            # Heuristic: for edge (u,v), score = weight(u,v) + sum of weights of edges (u,w) and (v,w)
            # where w is common neighbor and (u,w),(v,w) in uncovered
            best_clique = None
            best_score = -1

            # To limit computation, sample edges if too many
            edge_list = list(uncovered_edges)
            if len(edge_list) > 500:
                random.seed(seed)
                edge_list = random.sample(edge_list, 500)

            for e in edge_list:
                u, v = tuple(e)
                # Potential clique: {u, v} plus common neighbors w where (u,w) and (v,w) in uncovered
                common = (adj[u] & adj[v])
                # Filter: w must have (u,w) and (v,w) in uncovered
                candidates = []
                for w in common:
                    if w == u or w == v:
                        continue
                    e_uw = frozenset((u, w))
                    e_vw = frozenset((v, w))
                    if e_uw in uncovered_edges and e_vw in uncovered_edges:
                        candidates.append(w)

                # Greedily build maximal clique starting from {u, v}
                clique_set = {u, v}
                changed = True
                while changed:
                    changed = False
                    # Find all vertices that can be added
                    addable = []
                    for w in candidates:
                        if w in clique_set:
                            continue
                        # Check if w is connected to all in clique_set via uncovered edges
                        ok = True
                        for x in clique_set:
                            e_xw = frozenset((x, w))
                            if e_xw not in uncovered_edges:
                                ok = False
                                break
                        if ok:
                            addable.append(w)

                    if addable:
                        # Pick the one that adds the most weight
                        best_w = None
                        best_w_score = -1
                        for w in addable:
                            score = 0.0
                            for x in clique_set:
                                e_xw = frozenset((x, w))
                                if e_xw in edge_weights:
                                    score += edge_weights[e_xw]
                            if score > best_w_score:
                                best_w_score = score
                                best_w = w
                        clique_set.add(best_w)
                        changed = True
                        # Update candidates: remove best_w
                        candidates = [w for w in candidates if w != best_w]
                        # Recompute addable for remaining
                        # Actually, we should recompute from scratch each iteration for correctness
                        # But for speed, let's just break and do a simpler approach

                    break  # One pass for speed

                # Compute weight sum of this clique
                clique_list = list(clique_set)
                weight_sum = 0.0
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        e = frozenset((clique_list[i], clique_list[j]))
                        if e in edge_weights:
                            weight_sum += edge_weights[e]

                if len(clique_set) >= 2 and weight_sum > best_score:
                    best_score = weight_sum
                    best_clique = clique_set

            return best_clique

        cliques = []
        max_iterations = 100000

        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if not uncovered:
                break

            # Find best clique
            clique_set = find_best_clique(uncovered, adj, n)

            if clique_set is None or len(clique_set) < 2:
                # Fallback: pick any edge
                e = next(iter(uncovered))
                u, v = tuple(e)
                clique_set = {u, v}

            # Convert to list
            clique_list = list(clique_set)
            cliques.append(clique_list)

            # Remove all edges within this clique from uncovered
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e = frozenset((clique_list[i], clique_list[j]))
                    if e in uncovered:
                        uncovered.discard(e)
                    # Update weight: increase weight of uncovered edges to prioritize them
                    # Actually, we want to prioritize edges that are still uncovered
                    # So we can keep weights as-is, or increase them slightly
                    # Let's increase weights of uncovered edges to guide future selections
                    # But this is expensive. Instead, let's just rely on the greedy.

            # Update weights: increase weight of uncovered edges to make them more attractive
            # This is the Lagrangian-like feedback
            for e in uncovered:
                edge_weights[e] *= 1.05  # Slight increase

        # Verify: every edge must be covered exactly once
        # This is guaranteed by construction if we cover all edges
        # But let's check if we covered all
        if uncovered:
            # Fallback: cover remaining edges one by one
            while uncovered:
                if time.time() > deadline:
                    break
                e = next(iter(uncovered))
                u, v = tuple(e)
                cliques.append([u, v])
                uncovered.discard(e)

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback
            cliques = []
            edges = instance(*spec)
            for u, v in edges:
                cliques.append([u, v])
            result.append(cliques)
            continue

        cliques = solve_split(*spec)
        result.append(cliques)

    return result
