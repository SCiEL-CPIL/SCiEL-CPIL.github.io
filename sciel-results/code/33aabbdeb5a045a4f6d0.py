"""Clique partition of two fixed dense random graphs: triangle decomposition + single edges."""
import time

_DEADLINE = 140.0  # wall-clock safety margin below the 150s kill
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge set for (n, p, seed)."""
    import random
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _validate(n, edges, cliques):
    """Every listed set is a clique (>=2) and every edge lies in exactly one clique."""
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set


def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    edge_set = set(edges)
    if not edges:
        return []

    # Adjacency bitsets (int masks)
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Greedy maximal clique packing: try to form largest possible cliques
    # Strategy: process edges, for each unused edge try to extend to a maximal clique
    # that minimizes the number of cliques needed.
    
    # Better approach: use a greedy algorithm that picks edges and tries to find
    # the largest clique containing that edge among remaining edges.
    
    # First, let's do a simple greedy: for each edge, find if it can be part of a
    # triangle, and if so, use the triangle. Process edges in order of "scarcity"
    # (edges that are in fewer potential triangles first).
    
    # Count how many triangles each edge could be in
    edge_tri_count = {}
    for (u, v) in edges:
        if u > v:
            u, v = v, u
        cand = adj[u] & adj[v]
        cand &= ~((1 << (v + 1)) - 1)  # only w > v
        cnt = cand.bit_count()
        edge_tri_count[(u, v)] = cnt

    # Sort edges: those with fewer triangle options first (more constrained)
    sorted_edges = sorted(edges, key=lambda e: edge_tri_count.get((min(e[0],e[1]), max(e[0],e[1])), 0))
    
    used = set()
    cliques = []
    
    for (u, v) in sorted_edges:
        if _elapsed() > _DEADLINE:
            break
        if (u, v) in used:
            continue
        # Try to extend this edge to the largest possible clique
        # Start with the edge {u, v}
        current_clique = [u, v]
        # Find common neighbors that are greater than max(u,v) to avoid duplicates
        max_uv = max(u, v)
        cand = adj[u] & adj[v]
        cand &= ~((1 << (max_uv + 1)) - 1)  # only w > max_uv
        
        # Try to greedily add vertices to form a larger clique
        # We want to maximize clique size while covering edges
        # A vertex w can be added if it's adjacent to all current clique members
        while cand and _elapsed() < _DEADLINE:
            # Pick the first candidate
            w = (cand & -cand).bit_length() - 1
            if w < 0:
                break
            # Check if w is adjacent to all current clique members
            can_add = True
            for x in current_clique:
                if not (adj[w] >> x) & 1:
                    can_add = False
                    break
            if can_add:
                current_clique.append(w)
                # Update candidates: must be adjacent to all clique members and > max
                new_max = max(current_clique)
                cand = cand & adj[w]
                cand &= ~((1 << (new_max + 1)) - 1)
            else:
                # Remove w from candidates
                cand &= ~(1 << w)
        
        # Now we have a clique. Mark all its edges as used.
        k = len(current_clique)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = current_clique[i], current_clique[j]
                e = (a, b) if a < b else (b, a)
                used.add(e)
        
        cliques.append(current_clique)
    
    # Remaining edges become single-edge cliques
    for (u, v) in edges:
        if (u, v) not in used:
            cliques.append([u, v])
    
    # Safety: verify the partition; fall back to all single-edge cliques if invalid
    if not _validate(n, edges, cliques):
        cliques = [[u, v] for (u, v) in edges]
    return cliques


def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
