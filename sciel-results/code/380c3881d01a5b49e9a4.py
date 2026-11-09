"""Rewritten program: Structural clique cover for split graphs."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

_DEADLINE = time.time() + 120.0
_MARGIN = 10.0


def _time_ok():
    return (time.time() - _DEADLINE) < -_MARGIN


def _build_adj(n, k, p, seed):
    """Reconstruct the split graph adjacency."""
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    # Clique part: 0 to k-1
    for u in range(k):
        for v in range(u + 1, k):
            adj[u].add(v)
            adj[v].add(u)
    # Cross edges: k to n-1 vs 0 to k-1
    for v in range(k, n):
        for u in range(k):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _solve_split(n, k, p, seed):
    """
    Solves the clique partition for a split graph.
    1. Group independent vertices by their neighborhood in the clique part.
    2. Form cliques from these groups.
    3. Cover remaining edges in the clique part with minimal cliques.
    """
    adj = _build_adj(n, k, p, seed)
    
    # 1. Analyze independent set vertices
    # Map: frozenset(neighbors_in_C) -> list of independent vertices
    groups = {}
    for v in range(k, n):
        # Get neighbors in C
        nbrs = frozenset(adj[v]) # adj[v] contains only C vertices for v >= k? 
        # Wait, adj[v] for v >= k contains neighbors in C.
        # It does NOT contain neighbors in I (since I is independent).
        # So adj[v] is exactly N(v) intersect C.
        if nbrs not in groups:
            groups[nbrs] = []
        groups[nbrs].append(v)
        
    cliques = []
    covered_edges_in_C = set()
    
    # 2. Form cliques for each group
    for nbrs, verts in groups.items():
        if not verts:
            continue
        # The clique is nbrs U verts
        # nbrs is a frozenset of ints (vertices in C)
        # verts is a list of ints (vertices in I)
        clique = list(nbrs) + verts
        if len(clique) >= 2:
            cliques.append(clique)
            
        # Mark edges in C covered by this clique
        # These are all pairs within nbrs
        nbrs_list = list(nbrs)
        for i in range(len(nbrs_list)):
            u = nbrs_list[i]
            for j in range(i + 1, len(nbrs_list)):
                v = nbrs_list[j]
                covered_edges_in_C.add((u, v) if u < v else (v, u))
                
    # 3. Find remaining edges in C
    # Total edges in C are all pairs (u, v) with u, v < k
    remaining_edges = []
    for u in range(k):
        for v in range(u + 1, k):
            if (u, v) not in covered_edges_in_C:
                remaining_edges.append((u, v))
                
    # 4. Cover remaining edges in C
    # We have a graph H on vertices 0..k-1 with edges remaining_edges
    # We need to partition these edges into cliques.
    # Greedy:
    # Build adjacency for H
    H_adj = [set() for _ in range(k)]
    H_edges_set = set(remaining_edges)
    for u, v in remaining_edges:
        H_adj[u].add(v)
        H_adj[v].add(u)
        
    # Greedy clique cover for H
    # We maintain the set of active edges
    active_edges = set(remaining_edges)
    
    while active_edges:
        if not _time_ok():
            # Fallback: cover each remaining edge with a size-2 clique
            # This is always valid
            for u, v in active_edges:
                cliques.append([u, v])
            break
            
        # Pick an edge
        # Heuristic: pick edge with max sum of degrees? Or just first?
        # To be fast, pick first. To be better, pick max degree endpoint.
        # Let's pick the edge incident to the vertex with highest degree in active_edges.
        deg = [0] * k
        for u, v in active_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Find vertex with max deg
        max_deg = -1
        start_u = -1
        start_v = -1
        for u, v in active_edges:
            if deg[u] > max_deg:
                max_deg = deg[u]
                start_u = u
                start_v = v
            if deg[v] > max_deg:
                max_deg = deg[v]
                start_u = v
                start_v = u
                
        if start_u == -1:
            break
            
        # Expand to maximal clique containing start_u, start_v
        # Candidates: vertices adjacent to both start_u and start_v in H
        # And not yet in the clique
        clique_set = {start_u, start_v}
        
        # Find all common neighbors of start_u and start_v in H
        common_nbrs = H_adj[start_u].intersection(H_adj[start_v])
        
        # We can add any subset of common_nbrs that forms a clique with each other and with start_u, start_v.
        # Since H is a subgraph of a complete graph (C is a clique), 
        # any set of vertices is a clique in the original graph C.
        # But in H, they must be connected.
        # So we need to find a maximal clique in H within common_nbrs U {start_u, start_v}.
        
        # Greedy expansion:
        # Sort common_nbrs by degree in H (descending) to prefer high connectivity
        sorted_nbrs = sorted(list(common_nbrs), key=lambda x: len(H_adj[x]), reverse=True)
        
        for w in sorted_nbrs:
            # Check if w is adjacent to all current members in clique_set
            # Since w is in common_nbrs, it is adjacent to start_u and start_v.
            # We need to check if w is adjacent to other added vertices.
            is_adj_to_all = True
            for member in clique_set:
                if member == w:
                    continue
                if w not in H_adj[member]:
                    is_adj_to_all = False
                    break
            if is_adj_to_all:
                clique_set.add(w)
                
        # Convert to list
        new_clique = list(clique_set)
        if len(new_clique) >= 2:
            cliques.append(new_clique)
            
        # Remove edges covered by this clique from active_edges
        # Edges within new_clique
        for i in range(len(new_clique)):
            u = new_clique[i]
            for j in range(i + 1, len(new_clique)):
                v = new_clique[j]
                edge = (u, v) if u < v else (v, u)
                if edge in active_edges:
                    active_edges.remove(edge)
                    
    return cliques


def _is_clique_set(cliques, adj, n):
    # every clique must be a true clique
    for c in cliques:
        if len(c) < 2:
            return False
        for i in range(len(c)):
            ci = c[i]
            for j in range(i + 1, len(c)):
                cj = c[j]
                if cj not in adj[ci]:
                    return False
    # every edge must be covered exactly once
    seen = set()
    for c in cliques:
        L = len(c)
        for i in range(L):
            ci = c[i]
            for j in range(i + 1, L):
                cj = c[j]
                key = (ci, cj) if ci < cj else (cj, ci)
                if key in seen:
                    return False
                seen.add(key)
    # check all edges in graph are covered
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if (u, v) not in seen:
                    return False
    return True


def construct():
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        if not _time_ok():
            # Fallback to trivial cover if time is up
            # This might be slow to generate but is valid
            # We can just return empty? No, must be valid.
            # If time is up, we can't even build adj safely? 
            # _build_adj is fast.
            adj = _build_adj(n, k, p, seed)
            # Trivial cover: one clique per edge
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        cliques.append([u, v])
            results.append(cliques)
            continue
            
        cliques = _solve_split(n, k, p, seed)
        
        # Verify
        adj = _build_adj(n, k, p, seed)
        if not _is_clique_set(cliques, adj, n):
            # If invalid, fallback to trivial cover
            # This should not happen with the new algorithm
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        cliques.append([u, v])
                        
        # Ensure types
        cliques = [[int(v) for v in c] for c in cliques]
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
