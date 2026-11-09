"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def _random_graph_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot selection.
    Returns list of frozensets of vertices.
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        P_union_X = P | X
        if P_union_X:
            # Pick pivot with max |P ∩ N(u)|
            best_u = max(P_union_X, key=lambda u: len(P & adj[u]))
        else:
            return
        
        # Iterate over P \ N(u)
        for v in list(P - adj[best_u]):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _chvatal_greedy_mis(n, adj_list):
    """
    Chvátal's greedy algorithm for Maximum Independent Set (approximation).
    Returns a set of vertices that form an independent set.
    """
    remaining = set(range(n))
    independent = set()
    
    while remaining:
        # Pick vertex with minimum degree in the induced subgraph on remaining
        min_deg = float('inf')
        chosen = None
        for v in remaining:
            deg = len(adj_list[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        independent.add(chosen)
        remaining.remove(chosen)
        # Remove all neighbors of chosen from remaining
        remaining -= adj_list[chosen]
    
    return independent

def _solve_hitting_set_via_conflict_graph(maximal_cliques, n):
    """
    1. Build auxiliary conflict graph H where vertices are maximal cliques.
       Edge (C1, C2) if C1 ∩ C2 = ∅.
    2. Find Maximum Independent Set (MIS) in H using Chvatal's Greedy.
    3. Let S be the union of cliques in MIS.
    4. If S does not hit all maximal cliques, augment S greedily.
    5. Prune S if possible.
    """
    if not maximal_cliques:
        return set()
    
    # Filter to cliques of size >= 2
    cliques_to_hit = [c for c in maximal_cliques if len(c) >= 2]
    
    if not cliques_to_hit:
        return set()
    
    m = len(cliques_to_hit)
    
    # Build Disjointness Graph H
    # Vertices are indices 0..m-1
    # Adjacency list for H
    H_adj = [set() for _ in range(m)]
    
    # Optimization: Precompute vertex to clique indices mapping?
    # Or just check intersections.
    # For small m (maximal cliques in sparse graphs are not too many, usually < 1000), O(m^2) is fine.
    for i in range(m):
        for j in range(i + 1, m):
            if not (cliques_to_hit[i] & cliques_to_hit[j]):
                H_adj[i].add(j)
                H_adj[j].add(i)
    
    # Find MIS in H
    mis_indices = _chvatal_greedy_mis(m, H_adj)
    
    # Union of cliques in MIS
    S = set()
    for idx in mis_indices:
        S.update(cliques_to_hit[idx])
    
    # Check if S is a hitting set
    # We need to hit all cliques in cliques_to_hit
    uncovered_indices = []
    for i in range(m):
        if not (S & cliques_to_hit[i]):
            uncovered_indices.append(i)
            
    # If there are uncovered cliques, augment S
    if uncovered_indices:
        # Greedy augmentation
        # We want to add vertices to S to cover uncovered cliques
        # Note: S already contains some vertices.
        # We can treat this as a standard set cover problem on the remaining cliques,
        # but we can reuse vertices already in S (cost 0) or add new ones (cost 1).
        # Actually, since we are minimizing |S|, adding a vertex that is already in S costs 0.
        # But we are constructing S. We just need to ensure every clique is hit.
        
        # Let's just run a greedy set cover on the uncovered cliques,
        # but we can "buy" vertices already in S for free?
        # No, S is fixed so far. We can add more vertices.
        # To minimize total size, we should prefer vertices that cover many uncovered cliques.
        
        remaining_uncovered = set(uncovered_indices)
        
        while remaining_uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in S:
                    # If v is already in S, it covers any clique containing it.
                    # But if it was in S, those cliques should have been covered in the first check?
                    # Yes, if v in S and v in C, then C is hit.
                    # So any uncovered clique does NOT contain any v in S.
                    # So we must pick a v NOT in S.
                    continue
                    
                count = 0
                for idx in remaining_uncovered:
                    if v in cliques_to_hit[idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
            
            S.add(best_v)
            
            new_remaining = set()
            for idx in remaining_uncovered:
                if best_v not in cliques_to_hit[idx]:
                    new_remaining.add(idx)
            remaining_uncovered = new_remaining

    # Pruning: Try to remove vertices from S
    # A vertex v can be removed if no clique is hit ONLY by v.
    # i.e., for every clique C, if v in C, then C contains some other vertex in S.
    
    # To make it efficient, we can just iterate a few times.
    changed = True
    while changed:
        changed = False
        for v in list(S):
            # Check if v is essential
            essential = False
            for i in range(m):
                C = cliques_to_hit[i]
                if v in C:
                    # Check if C has any other vertex in S
                    if len(C & (S - {v})) == 0:
                        essential = True
                        break
            if not essential:
                S.remove(v)
                changed = True
                
    return S

def construct():
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        maximal_cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        # Filter to cliques of size >= 2
        cliques_to_hit = [c for c in maximal_cliques if len(c) >= 2]
        
        if not cliques_to_hit:
            results.append([])
            continue
        
        # Use the new conflict graph approach
        hitting_set = _solve_hitting_set_via_conflict_graph(cliques_to_hit, n)
        results.append(sorted(hitting_set))
    
    return results
# EVOLVE-BLOCK-END
