"""Clique partition of two fixed dense random graphs via greedy set-cover."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Build adjacency sets and edge set
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        # For each vertex, precompute common neighbors with other vertices
        # common[v][w] = number of vertices x such that x is adjacent to both v and w
        # and (v,x) and (w,x) are in the graph
        # Actually we need: for a candidate clique C containing edge (u,v),
        # we want to maximize |C|. A vertex x can be added to {u,v} iff
        # x is adjacent to all vertices in the current clique.

        # Greedy approach:
        # 1. While there are uncovered edges:
        #    a. Pick an uncovered edge (u,v) with highest "potential"
        #       (potential = number of vertices that could extend the clique)
        #    b. Build the largest clique containing (u,v) using only uncovered edges
        #       and vertices that can be added while maintaining clique property
        #    c. Add this clique to the partition, remove its edges from uncovered set

        uncovered = set(edge_set)
        if not uncovered:
            return []

        cliques = []

        while uncovered:
            if time.time() > DEADLINE:
                break

            # Pick the uncovered edge with the most "clique density"
            # For edge (u,v), potential = number of vertices w such that
            # w is adjacent to both u and v, and all edges among {u,v,w} are in uncovered
            # More generally, we want edges that are part of large cliques in the
            # remaining graph.

            # Heuristic: score = number of common neighbors in the remaining graph
            # (i.e., vertices w where (u,w) and (v,w) are both in uncovered)
            best_edge = None
            best_score = -1

            # To speed up, we can sample or use a priority approach
            # For n=64, edge count is at most 2016, so iterating all is fine
            for e in uncovered:
                u, v = e
                # Count common neighbors w such that (u,w) and (v,w) are in uncovered
                # w must be different from u and v
                common = 0
                # We can compute this by intersecting adj[u] and adj[v]
                # but we need to check if the edges are in uncovered
                # adj[u] ∩ adj[v] gives candidates; check if (u,w) and (v,w) in uncovered
                for w in adj[u]:
                    if w == v:
                        continue
                    if w in adj[v]:
                        e1 = (u, w) if u < w else (w, u)
                        e2 = (v, w) if v < w else (w, v)
                        if e1 in uncovered and e2 in uncovered:
                            common += 1
                # Score: prefer edges with high common neighbor count
                # Also consider the "density" - but for greedy set cover,
                # we want to cover as many edges as possible per clique
                score = common
                if score > best_score:
                    best_score = score
                    best_edge = e

            if best_edge is None:
                break

            u, v = best_edge

            # Build the maximum clique containing (u,v) in the subgraph induced by uncovered edges
            # Start with {u, v}
            clique = {u, v}
            # Candidate vertices that can be added: must be adjacent to all in clique
            # and all edges must be in uncovered
            candidates = set()
            for w in adj[u]:
                if w == v:
                    continue
                if w in adj[v]:
                    # Check if (u,w) and (v,w) are in uncovered
                    e1 = (u, w) if u < w else (w, u)
                    e2 = (v, w) if v < w else (w, v)
                    if e1 in uncovered and e2 in uncovered:
                        candidates.add(w)

            # Greedily add candidates that maintain clique property
            # Sort candidates by degree within the candidate set (or just iterate)
            # Use a greedy approach: repeatedly add the vertex with the most connections
            # to the current clique and other candidates
            changed = True
            while changed:
                changed = False
                # For each candidate, check if it can be added (adjacent to all in clique)
                # and count how many other candidates it's adjacent to
                addable = []
                for w in list(candidates):
                    # Check if w is adjacent to all vertices in clique
                    can_add = True
                    for x in clique:
                        if w not in adj[x]:
                            can_add = False
                            break
                        # Check edge is in uncovered
                        e = (w, x) if w < x else (x, w)
                        if e not in uncovered:
                            can_add = False
                            break
                    if can_add:
                        # Count connections to other candidates
                        conn = 0
                        for y in candidates:
                            if y == w:
                                continue
                            if w in adj[y]:
                                e = (w, y) if w < y else (y, w)
                                if e in uncovered:
                                    conn += 1
                        addable.append((conn, w))
                if addable:
                    # Add the one with most connections
                    addable.sort(reverse=True)
                    best_conn, best_w = addable[0]
                    clique.add(best_w)
                    candidates.discard(best_w)
                    changed = True

            # Remove all edges within the clique from uncovered
            clique_list = sorted(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (a, b) if a < b else (b, a)
                    uncovered.discard(e)

            cliques.append(clique_list)

        # If we timed out, we need to handle remaining uncovered edges
        # Each remaining edge becomes its own clique of size 2
        for e in uncovered:
            cliques.append(list(e))

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > DEADLINE:
            # Fallback: return trivial partition
            edges = random_graph(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
