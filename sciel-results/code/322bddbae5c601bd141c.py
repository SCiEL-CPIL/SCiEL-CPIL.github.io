"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _random_graph(n, p, seed):
    """Generate edge list matching the verifier's random_graph exactly."""
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    m = len(edges)
    if m == 0:
        return []

    # Build adjacency as sets for fast intersection
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Map edge to index for quick lookup
    edge_set = set(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    # Track which edges are covered
    covered = [False] * m
    cliques = []

    # Precompute common neighbors for each edge to prioritize "hard" edges
    # An edge (u,v) has common neighbors = adj[u] & adj[v]
    # Edges with fewer common neighbors are harder to cover with large cliques
    edge_common_neighbors = []
    for e in edges:
        u, v = e
        cn = adj[u] & adj[v]
        edge_common_neighbors.append(cn)

    # Strategy: greedily form cliques, prioritizing edges that are "harder" to cover
    # i.e., edges with fewer common neighbors (they can't be in many potential cliques)
    # This prevents fragmentation of sparse regions.

    # We'll use a priority: process edges with fewer common neighbors first
    # But we also want large cliques, so we balance.

    # Greedy approach:
    # 1. Find an uncovered edge
    # 2. Try to extend it into the largest possible clique
    # 3. Add that clique, mark all its edges as covered
    # 4. Repeat

    # To be efficient, we maintain a list of uncovered edges and pick strategically.

    # Sort edges by number of common neighbors (ascending) - hard edges first
    edge_order = sorted(range(m), key=lambda i: len(edge_common_neighbors[i]))

    # Pointer to next candidate in edge_order
    ptr = 0
    while ptr < m:
        if _elapsed() > _DEADLINE:
            break
        i = edge_order[ptr]
        ptr += 1
        if covered[i]:
            continue

        u, v = edges[i]
        # Start with the edge's two vertices
        clique = [u, v]
        # Candidate vertices that can join the clique: common neighbors of all current clique members
        # Start with common neighbors of u and v
        candidates = set(edge_common_neighbors[i])

        # Try to add vertices to grow the clique
        # Sort candidates by degree within the candidate set for better greedy choice
        # A vertex with high degree among candidates is more likely to connect to others
        # Actually, we want vertices that connect to ALL current clique members AND to many other candidates
        # Let's just greedily add vertices that are connected to all current clique members

        # Better approach: repeatedly find a vertex in candidates that has the highest degree
        # within candidates (to keep the subgraph dense), add it, and update candidates.

        changed = True
        while changed:
            if _elapsed() > _DEADLINE:
                break
            changed = False
            if not candidates:
                break

            # Find the candidate with the highest number of connections to other candidates
            # This tends to produce denser cliques
            best_v = None
            best_deg = -1
            for w in candidates:
                deg = len(candidates & adj[w])
                if deg > best_deg:
                    best_deg = deg
                    best_v = w

            if best_v is not None and best_deg >= len(candidates) - 1:
                # best_v connects to all other candidates, safe to add
                clique.append(best_v)
                candidates.discard(best_v)
                # Update candidates: must also connect to best_v
                candidates = candidates & adj[best_v]
                changed = True
            else:
                # No single vertex connects to all others; try adding the one with highest degree
                # but this might not form a clique with all current members... 
                # Actually, all candidates already connect to all current clique members.
                # We need candidates that also connect to the new vertex.
                # If no vertex connects to all others, we can still add one and intersect.
                if best_v is not None:
                    clique.append(best_v)
                    candidates.discard(best_v)
                    candidates = candidates & adj[best_v]
                    changed = True
                else:
                    break

        # Add the clique
        if len(clique) >= 2:
            cliques.append(clique)
            # Mark all edges within the clique as covered
            for idx_a in range(len(clique)):
                for idx_b in range(idx_a + 1, len(clique)):
                    a, b = clique[idx_a], clique[idx_b]
                    e = (min(a, b), max(a, b))
                    if e in edge_to_idx:
                        covered[edge_to_idx[e]] = True

    # Verify: every edge must be covered. If not, add remaining edges as 2-cliques.
    for i in range(m):
        if not covered[i]:
            cliques.append(list(edges[i]))

    return cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _elapsed() > _DEADLINE:
            # Fallback: just return edge lists
            edges = _random_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = _solve_instance(n, p, seed)
        results.append(cliques)
    return results
# EVOLVE-BLOCK-END
