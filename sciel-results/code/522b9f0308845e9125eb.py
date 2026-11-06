"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _bron_kerbosch_max_cliques(adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot selection.
    adj: list of sets, adj[i] = set of neighbors of i.
    Returns list of frozensets (maximal cliques).
    """
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection: choose u in P ∪ X with max |P ∩ N(u)|
        candidates = P.union(X)
        if candidates:
            pivot = max(candidates, key=lambda u: len(P.intersection(adj[u])))
            candidates_to_iter = P.difference(adj[pivot])
        else:
            candidates_to_iter = set()
        
        for v in list(candidates_to_iter):
            R.add(v)
            P_new = P.intersection(adj[v])
            X_new = X.intersection(adj[v])
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _build_adjacency(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _solve_hitting_set(adj, max_cliques, time_limit=45):
    """
    Find a minimum hitting set for max_cliques (each clique size >= 2).
    Uses greedy + local search / LP relaxation if available.
    Returns a list of vertices.
    """
    start_time = time.time()
    
    if not max_cliques:
        return []
    
    # Get all vertices that appear in any clique
    all_verts = set()
    for c in max_cliques:
        all_verts.update(c)
    all_verts = sorted(all_verts)
    vert_index = {v: i for i, v in enumerate(all_verts)}
    m = len(all_verts)
    
    # If no cliques, return empty
    if not max_cliques:
        return []
    
    # Build incidence: for each clique, list of vertex indices
    clique_verts = []
    for c in max_cliques:
        vs = [vert_index[v] for v in c]
        clique_verts.append(vs)
    
    # Simple greedy: pick vertex covering most uncovered cliques
    uncovered = set(range(len(clique_verts)))
    selected = set()
    
    # Precompute for each vertex, which cliques it covers
    vert_covers = [set() for _ in range(m)]
    for ci, vs in enumerate(clique_verts):
        for vi in vs:
            vert_covers[vi].add(ci)
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        # Pick vertex covering most uncovered cliques
        best_v = -1
        best_count = -1
        for v in range(m):
            if v in selected:
                continue
            count = len(uncovered.intersection(vert_covers[v]))
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count <= 0:
            break
        selected.add(best_v)
        uncovered -= vert_covers[best_v]
    
    # Local search: try to remove vertices
    # For each selected vertex, check if removing it still covers all cliques
    improved = True
    while improved:
        improved = False
        for v in list(selected):
            # Check if v is the only vertex in selected for any clique
            critical = False
            for ci, vs in enumerate(clique_verts):
                if v in vs:
                    # Check if any other selected vertex is in this clique
                    covered = False
                    for w in vs:
                        if w != v and w in selected:
                            covered = True
                            break
                    if not covered:
                        critical = True
                        break
            if not critical:
                selected.remove(v)
                improved = True
    
    # Try to swap: replace a selected vertex with an unselected one to potentially reduce size
    # This is complex; for now, return the greedy + pruning result
    result = [all_verts[v] for v in sorted(selected)]
    return result

def construct():
    out = []
    total_deadline = time.time() + 140  # safety margin
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > total_deadline - 30:
            # Not enough time; use trivial fallback
            rng = Random(seed)
            touched = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        touched.add(u)
                        touched.add(v)
            out.append(sorted(touched))
            continue
        
        adj = _build_adjacency(n, p, seed)
        max_cliques = _bron_kerbosch_max_cliques(adj)
        
        # Filter cliques with at least 2 vertices (already done in bron_kerbosch)
        result = _solve_hitting_set(adj, max_cliques, time_limit=40)
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
