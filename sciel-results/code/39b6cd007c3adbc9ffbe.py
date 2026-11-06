"""
Clique partition via maximal-clique enumeration + greedy set cover.
Design family: enumerate all maximal cliques (Bron-Kerbosch with pivot),
then greedily select cliques to cover all edges, prioritizing the clique
that covers the most uncovered edges.
"""
import time
import random
import sys

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_partition(cliques, edge_list, adj):
    if not cliques:
        return len(edge_list) == 0
    edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    for clique in cliques:
        if len(clique) < 2:
            return False
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
                e = _edge_key(clique[a], clique[b])
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _bron_kerbosch(adj_mask, n):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    adj_mask: list of bitmasks for adjacency.
    Returns list of frozensets (maximal cliques).
    """
    cliques = []
    
    def bk(R, P, X):
        if _remaining() < 5:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u in P ∪ X maximizing |P ∩ N(u)|
        union = P | X
        if union:
            # Find pivot
            best_u = -1
            best_count = -1
            # Iterate over bits in union
            temp = union
            while temp:
                u = temp & (-temp)
                u_idx = u.bit_length() - 1
                temp ^= u
                count = (P & adj_mask[u_idx]).bit_count()
                if count > best_count:
                    best_count = count
                    best_u = u_idx
                if best_count == P.bit_count():
                    break  # can't do better
            
            candidates = P & ~adj_mask[best_u] if best_u >= 0 else P
        else:
            candidates = P
        
        # Iterate over candidates
        temp = candidates
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            
            R_new = R | v_bit
            P_new = P & adj_mask[v]
            X_new = X & adj_mask[v]
            
            bk(R_new, P_new, X_new)
            
            P ^= v_bit
            X |= v_bit
    
    full_mask = (1 << n) - 1
    bk(0, full_mask, 0)
    return cliques

def _solve_one_graph(adj, edge_list, time_budget=55.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []

    # Build adjacency bitmask
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

    m = len(edge_list)
    all_edges = set(_edge_key(u, v) for u, v in edge_list)

    # Enumerate all maximal cliques
    max_cliques = _bron_kerbosch(adj_mask, n)
    
    # Filter: keep only cliques with at least 2 vertices
    max_cliques = [c for c in max_cliques if len(c) >= 2]
    
    # If no maximal cliques found (shouldn't happen for dense graph), fallback
    if not max_cliques:
        return [[u, v] for u, v in edge_list]
    
    # For each maximal clique, compute its edge set
    clique_edge_sets = []
    for clique in max_cliques:
        verts = list(clique)
        es = set()
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                es.add(_edge_key(verts[i], verts[j]))
        # Only keep edges that are actually in the graph
        es = es & all_edges
        if es:
            clique_edge_sets.append(es)
    
    # Greedy set cover: repeatedly pick the clique covering most uncovered edges
    uncovered = set(all_edges)
    selected_cliques = []
    # Precompute for each clique its edge set for quick access
    clique_info = list(zip(max_cliques, clique_edge_sets))
    
    while uncovered and _remaining() > 5:
        best_idx = -1
        best_count = 0
        for idx, (clique, es) in enumerate(clique_info):
            if not es:
                continue
            # Count newly covered edges
            new_covered = len(es & uncovered)
            if new_covered > best_count:
                best_count = new_covered
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            # Fallback: cover remaining edges as 2-cliques
            for ek in uncovered:
                selected_cliques.append(sorted(ek))
            uncovered = set()
            break
        
        clique, es = clique_info[best_idx]
        selected_cliques.append(sorted(clique))
        uncovered -= es
        # Clear this clique's edge set so it's not picked again
        clique_edge_sets[best_idx] = set()
    
    # Handle any leftover edges
    for ek in uncovered:
        selected_cliques.append(sorted(ek))
    
    # Verify
    if not _verify_partition(selected_cliques, edge_list, adj):
        return [[u, v] for u, v in edge_list]
    
    return selected_cliques

def construct():
    global _START
    _START = time.time()

    results = []

    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue

        adj, edge_list = _generate_graph(n, p, seed)

        time_budget = min(60.0, _remaining() - 15)
        if time_budget < 5:
            time_budget = 5

        cliques = _solve_one_graph(adj, edge_list, time_budget)

        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]

        results.append(cliques)

    return results
