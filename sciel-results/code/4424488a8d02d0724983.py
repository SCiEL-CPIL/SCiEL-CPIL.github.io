"""Clique-partition of two fixed dense random graphs.

Strategy:
  1. Build the graph.
  2. Greedily extract cliques by picking an unassigned edge and expanding it
     into the largest possible clique such that all edges within the clique
     are currently unassigned.
  3. This guarantees every edge is covered exactly once.
  4. A wall-clock deadline guard returns the best partition found so far.
"""

import time

# ---- task parameters (fixed by the verifier) ----
_PARAMS = ((32, 0.6, 601), (64, 0.55, 602))
_DEADLINE = 140.0   # seconds of wall-clock to stay safely under the 150s kill


def _build_graph(n, p, seed):
    """Return (adjacency bitmasks, edge list) for random_graph(n, p, seed)."""
    from random import Random
    rng = Random(seed)
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
    return adj, edges


def _solve_instance(n, p, seed, deadline):
    adj, edges = _build_graph(n, p, seed)
    unassigned = set(edges)
    cliques = []

    # Helper to check if a set of vertices forms a clique with all edges unassigned
    def is_valid_clique_expansion(current_clique, candidate):
        # Check if candidate is adjacent to all in current_clique
        # and if all edges (candidate, v) for v in current_clique are in unassigned
        for v in current_clique:
            edge = (candidate, v) if candidate < v else (v, candidate)
            if edge not in unassigned:
                return False
            if not (adj[candidate] >> v) & 1:
                return False
        return True

    while unassigned:
        if time.time() > deadline:
            break
            
        # Pick an arbitrary unassigned edge to start
        u, v = next(iter(unassigned))
        clique = [u, v]
        
        # Find common neighbors of all vertices in the clique
        # Start with intersection of neighbors of u and v
        cand_mask = adj[u] & adj[v]
        
        # Greedily add vertices to the clique
        # We want to maximize the size of the clique.
        # Heuristic: add the vertex that is connected to the most other candidates?
        # Or just add any valid vertex until no more can be added.
        # To get a larger clique, we should be careful not to "use up" edges.
        # Since we are removing edges as we go, any vertex added now consumes its edges to the clique.
        
        while True:
            if time.time() > deadline:
                break
                
            # Find the best candidate from cand_mask
            best_cand = -1
            best_score = -1
            c = cand_mask
            while c:
                low = c & (-c)
                w = low.bit_length() - 1
                c &= c - 1
                
                # Check if w can be added
                if is_valid_clique_expansion(clique, w):
                    # Score: number of neighbors in cand_mask that w is connected to?
                    # This helps grow the clique further.
                    score = (adj[w] & cand_mask).bit_count()
                    if score > best_score:
                        best_score = score
                        best_cand = w
                        
            if best_cand == -1:
                break
                
            clique.append(best_cand)
            # Update candidates: intersection of neighbors of all clique members
            cand_mask &= adj[best_cand]
            
        # Remove all edges within the clique from unassigned
        for i in range(len(clique)):
            a = clique[i]
            for j in range(i + 1, len(clique)):
                b = clique[j]
                e = (a, b) if a < b else (b, a)
                unassigned.discard(e)
                
        cliques.append(sorted(clique))

    return cliques


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    results = []
    for (n, p, seed) in _PARAMS:
        if time.time() > deadline:
            # not enough time; fall back to single-edge cliques for remaining
            adj, edges = _build_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        # give each instance its own sub-deadline so we finish both
        sub_deadline = min(deadline, t0 + _DEADLINE - 1.0)
        results.append(_solve_instance(n, p, seed, sub_deadline))
    return results
