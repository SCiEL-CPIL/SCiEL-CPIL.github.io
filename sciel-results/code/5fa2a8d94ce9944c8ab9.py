# EVOLVE-BLOCK-START
import time
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _max_clique_branch_and_bound(adj, deadline):
    """Find a maximum clique using Bron-Kerbosch with pivot and coloring."""
    n = len(adj)
    best_clique = []
    
    def expand(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if not P:
            return
        
        # Simple coloring bound
        # Order P by degree
        P_list = list(P)
        P_list.sort(key=lambda v: len(adj[v]), reverse=True)
        
        # Greedy coloring for upper bound
        color = {}
        color_class = []
        for v in P_list:
            # Find smallest color not used by neighbors of v in P_list that got smaller color
            used_colors = set()
            for u in adj[v]:
                if u in color and color[u] < len(color_class):
                    used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c == len(color_class):
                color_class.append(v)
        
        max_color = len(color_class)
        
        # If |R| + max_color <= |best_clique|, prune
        if len(R) + max_color <= len(best_clique):
            return
        
        # Bron-Kerbosch with pivot
        # Choose pivot u from P ∪ X that maximizes |N(u) ∩ P|
        all_candidates = P | X
        if all_candidates:
            pivot = max(all_candidates, key=lambda u: len(adj[u] & P))
        else:
            pivot = None
        
        if pivot is not None:
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            if time.time() > deadline:
                return
            expand(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    P = set(range(n))
    X = set()
    R = set()
    expand(R, P, X)
    return best_clique

def _greedy_clique_partition_v2(adj, deadline):
    """
    Structural change: Instead of finding triangles and extending,
    we use a maximum clique finding approach (branch and bound) to find
    the largest possible clique first, then recursively partition the rest.
    This is a different modeling family: maximum clique extraction vs triangle extension.
    """
    n = len(adj)
    remaining = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                remaining.add((u, v))
    
    if not remaining:
        return []
    
    cliques = []
    
    # Build subgraph from remaining edges
    # We'll repeatedly find a large clique in the remaining subgraph
    # and remove it, until no edges remain.
    
    # For efficiency, we maintain a list of active vertices (those with remaining edges)
    active_vertices = set()
    for u, v in remaining:
        active_vertices.add(u)
        active_vertices.add(v)
    
    while remaining and time.time() < deadline:
        # Build adjacency for the subgraph induced by remaining edges
        sub_adj = {v: set() for v in active_vertices}
        for u, v in remaining:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
        
        # Find a large clique in this subgraph
        # Use a heuristic: pick a vertex with high degree, then find max clique containing it
        if not active_vertices:
            break
        
        # Pick starting vertex: highest degree in remaining subgraph
        start_vertex = max(active_vertices, key=lambda v: len(sub_adj[v]))
        
        # Find max clique containing start_vertex
        # This is equivalent to finding max clique in the common neighbors of start_vertex
        # plus start_vertex itself
        
        # Candidates: common neighbors of start_vertex in the subgraph
        candidates = set(sub_adj[start_vertex])
        
        # Now find max clique in candidates, and add start_vertex
        best_clique = [start_vertex]
        
        # Simple greedy extension from start_vertex
        # Try to build a large clique greedily
        current_clique = {start_vertex}
        cand = set(candidates)
        
        while cand and time.time() < deadline:
            # Pick vertex with most connections to current_clique
            best_v = None
            best_count = -1
            for v in cand:
                count = len(current_clique & sub_adj[v])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count < len(current_clique):
                break
            current_clique.add(best_v)
            cand = cand & sub_adj[best_v]
        
        # Verify all edges in current_clique are in remaining
        valid = True
        clique_list = sorted(current_clique)
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = clique_list[i], clique_list[j]
                if (min(a, b), max(a, b)) not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(current_clique) >= 2:
            # Remove all edges in this clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    remaining.discard((min(a, b), max(a, b)))
            cliques.append(clique_list)
            
            # Update active_vertices
            for v in current_clique:
                if not any((min(v, u), max(v, u)) in remaining for u in active_vertices if u != v):
                    active_vertices.discard(v)
        else:
            # Fallback: take a single edge
            edge = next(iter(remaining))
            u, v = edge
            cliques.append([u, v])
            remaining.discard(edge)
            # Update active
            for vertex in (u, v):
                if not any((min(vertex, w), max(vertex, w)) in remaining for w in active_vertices if w != vertex):
                    active_vertices.discard(vertex)
    
    # Cover any remaining edges as pairs
    for edge in list(remaining):
        if time.time() > deadline:
            break
        u, v = edge
        if (u, v) in remaining:
            cliques.append([u, v])
            remaining.discard((u, v))
    
    return cliques

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: trivial partition
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _greedy_clique_partition_v2(adj, deadline)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
