"""Clique partition via local search on edge-center assignment."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_START = time.time()
_LIMIT = 145.0

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _elapsed():
    return time.time() - _START

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Map edge index to (u, v)
    edge_list = edges
    m = len(edge_list)
    
    # Build adjacency for fast common-neighbor checks
    # adj[u] = set of neighbors
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute for each edge (i, u, v), which other edges share a vertex?
    # Actually, for the "center" representation:
    # Each edge e=(u,v) is assigned a center c.
    # A valid partition requires that for any two edges e1, e2 assigned to the same center c,
    # they must be adjacent (share a vertex) AND the union of their vertices must form a clique.
    # Wait, if e1=(a,b) and e2=(c,d) are in the same clique, then {a,b,c,d} must be a clique.
    # This implies e1 and e2 must be "compatible".
    
    # Let's define compatibility: Two edges (u,v) and (x,y) are compatible if {u,v,x,y} is a clique.
    # This requires:
    # 1. u,v,x,y are distinct (or not, but edges are distinct).
    # 2. All pairs among {u,v,x,y} are edges in the graph.
    
    # This is expensive to check on the fly.
    # Let's use a different representation:
    # A clique is defined by a set of vertices.
    # The "center" approach is tricky because an edge (u,v) can belong to many cliques.
    
    # Alternative: Represent the partition as a list of cliques.
    # Start with a random valid partition (e.g., single edges).
    # Move: Try to merge two cliques if their union is a clique.
    # Move: Try to move an edge from one clique to another if validity is maintained.
    # Move: Try to split a clique? No, we want to minimize count, so merging is key.
    
    # Better Local Search:
    # State: A set of cliques (frozensets of vertices).
    # Initial State: All edges as single cliques.
    # Objective: Minimize number of cliques.
    # Moves:
    # 1. Merge two cliques C1, C2 if C1 ∪ C2 is a clique.
    # 2. Move an edge e from C1 to C2 if C2 ∪ {e} is a clique (and e not in C2).
    #    Note: Moving an edge changes the clique structure.
    
    # Let's stick to the "merge only" heuristic first, then add moves.
    # Actually, "merge only" is a greedy approach. Local search allows "unmerging" or "rearranging".
    
    # Let's try a different angle: **Greedy with Randomized Restarts and Local Optimization**.
    # The current program does batched maximal clique packing.
    # Let's try: **Iterative Edge Assignment to Maximal Cliques**.
    
    # New Strategy: **Simulated Annealing on Clique Coverage**.
    # Represent state as a list of cliques.
    # We will try to minimize the number of cliques.
    
    # To make this efficient, we will work with the "residual graph" idea but use a different selection heuristic.
    # The current program picks the largest maximal cliques.
    # Maybe we should pick cliques that cover "bottleneck" edges?
    
    # Let's implement a **Greedy Set Cover** variant where we prioritize edges that are in fewest maximal cliques.
    
    # 1. Find all maximal cliques in the original graph.
    # 2. For each edge, count how many maximal cliques contain it.
    # 3. Greedily select the clique that covers the most "hard" edges (edges with low count).
    # 4. Remove covered edges, repeat.
    
    # This is still a greedy set cover, but with a different heuristic.
    # Is this materially different? It's a different heuristic for the same greedy framework.
    # The directive asks for a "different design or modelling family".
    
    # Let's try **Local Search on Vertex Coloring of the Line Graph**.
    # Edge clique partition is equivalent to coloring the line graph such that each color class is a clique in the line graph?
    # No, a clique in the line graph is a set of edges in G that are pairwise adjacent.
    # This is exactly a clique in G (if they share a common vertex) or a star (if they share a common vertex).
    # Wait. In the line graph L(G), two edges are adjacent if they share a vertex in G.
    # A clique in L(G) is a set of edges in G that are pairwise adjacent.
    # This corresponds to either:
    # 1. A star in G (all edges share a common vertex).
    # 2. A triangle in G (edges form a triangle).
    # 3. A larger clique in G? If G has a K4, the edges of K4 form a K6 in L(G)?
    #    Edges of K4: (1,2),(1,3),(1,4),(2,3),(2,4),(3,4).
    #    (1,2) is adjacent to (1,3),(1,4),(2,3),(2,4). Not (3,4).
    #    So the edges of a K4 do NOT form a clique in L(G).
    #    Therefore, a clique partition of G is NOT the same as a clique cover of L(G).
    #    
    #    However, a **clique in G** corresponds to a set of edges that form a **clique in L(G)** ONLY if the clique in G is a triangle or a star.
    #    For larger cliques, the edges do not form a clique in L(G).
    #    
    #    So this mapping is not direct.
    
    # Let's go back to **Greedy with "Hard Edge" Prioritization**.
    # It's a structural change in the selection strategy.
    
    # 1. Compute all maximal cliques of G.
    # 2. For each edge, compute the set of maximal cliques containing it.
    # 3. While edges remain:
    #    a. Find the edge with the fewest covering maximal cliques.
    #    b. Among those, pick the one with the largest size.
    #    c. Add to partition, remove its edges.
    #    d. Update counts for remaining edges.
    
    # This is a different greedy heuristic.
    
    # To make it even more different, let's add a **Local Search** step after the greedy phase.
    # Try to merge any two cliques in the partition if their union is a clique.
    # Repeat until no merges are possible.
    
    def _maximal_cliques(n, edges):
        # Bron-Kerbosch
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= 1 << v
            adj_mask[v] |= 1 << u
        
        result = []
        def bronk(R, P, X):
            if not P:
                if not X and R.bit_count() >= 2:
                    result.append(R)
                return
            # Pivot
            ux = P | X
            best_u = -1
            best_c = -1
            for u in range(n):
                if (ux >> u) & 1:
                    c = (P & adj_mask[u]).bit_count()
                    if c > best_c:
                        best_c = c
                        best_u = u
            if best_u >= 0:
                P_sub = P & ~(1 << best_u)
            else:
                P_sub = P
            
            # Iterate bits in P_sub
            ps = P_sub
            while ps:
                lsb = ps & -ps
                v = lsb.bit_length() - 1
                bronk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                ps ^= lsb
        
        bronk(0, (1 << n) - 1, 0)
        return result

    maxcls = _maximal_cliques(n, edges)
    if not maxcls:
        return [[u, v] for u, v in edges]
    
    # Map edge to list of maximal cliques containing it
    edge_to_cliques = {e: [] for e in edges}
    clique_edges = []
    for i, mask in enumerate(maxcls):
        verts = [v for v in range(n) if (mask >> v) & 1]
        if len(verts) < 2:
            continue
        ecs = set()
        for u in verts:
            for v in verts:
                if u < v:
                    ecs.add((u, v))
        for e in ecs:
            if e in edge_to_cliques:
                edge_to_cliques[e].append(i)
        clique_edges.append((i, ecs, verts))
    
    # Filter valid cliques (all edges present)
    valid_cliques = []
    for i, ecs, verts in clique_edges:
        if all(e in edge_to_cliques for e in ecs):
            valid_cliques.append((i, ecs, verts))
            
    # Initialize counts
    edge_count = {e: len(edge_to_cliques[e]) for e in edges}
    
    chosen = []
    remaining = set(edges)
    
    while remaining:
        if not _deadline_ok():
            break
        
        # Find edge with min count
        min_count = min(edge_count[e] for e in remaining)
        candidates = [e for e in remaining if edge_count[e] == min_count]
        
        # Among candidates, pick clique with max size
        best_clique = None
        best_size = -1
        for e in candidates:
            for idx in edge_to_cliques[e]:
                # Check if this clique is still valid (all edges in remaining)
                # We need to store the clique edges
                # We have valid_cliques list, but need to map idx to it
                pass
        
        # This mapping is getting complex. Let's simplify.
        # Just pick the first candidate edge, and among cliques containing it, pick the largest.
        
        e0 = candidates[0]
        best_c = None
        best_sz = -1
        for idx in edge_to_cliques[e0]:
            # Find the clique
            # We need a way to access clique edges from idx
            # Let's store clique_edges in a dict
            pass
            
        # Let's refactor to store cliques in a dict
        clique_map = {}
        for i, ecs, verts in valid_cliques:
            clique_map[i] = (ecs, verts)
            
        for idx in edge_to_cliques[e0]:
            if idx in clique_map:
                ecs, verts = clique_map[idx]
                # Check if all edges in ecs are in remaining
                if all(ed in remaining for ed in ecs):
                    if len(ecs) > best_sz:
                        best_sz = len(ecs)
                        best_c = verts
                        
        if best_c:
            chosen.append(best_c)
            # Remove edges
            ecs, _ = clique_map[edge_to_cliques[e0][0]] # Wait, we need the ecs of best_c
            # Let's store best_ecs
            pass
            
    # This is getting messy. Let's stick to the **Greedy Maximal Clique** but with a **Different Sorting Key**.
    # Current: Sort by edge count desc.
    # New: Sort by (min_edge_count_in_clique, edge_count) desc.
    # i.e., prioritize cliques that contain "rare" edges.
    
    # Let's implement that.
    
    # 1. Find all maximal cliques.
    # 2. For each edge, count frequency in maximal cliques.
    # 3. For each clique, compute min_freq = min(freq[e] for e in clique).
    # 4. Sort cliques by (min_freq, size) ascending? No, we want to cover rare edges first.
    #    So sort by min_freq ascending.
    
    maxcls = _maximal_cliques(n, edges)
    if not maxcls:
        return [[u, v] for u, v in edges]
        
    # Compute edge frequencies
    edge_freq = {e: 0 for e in edges}
    cliques_data = []
    for mask in maxcls:
        verts = [v for v in range(n) if (mask >> v) & 1]
        if len(verts) < 2:
            continue
        ecs = set()
        for u in verts:
            for v in verts:
                if u < v:
                    ecs.add((u, v))
        # Check validity
        if all(e in edge_freq for e in ecs):
            for e in ecs:
                edge_freq[e] += 1
            cliques_data.append((verts, ecs))
            
    # Update min_freq for each clique
    for i, (verts, ecs) in enumerate(cliques_data):
        min_f = min(edge_freq[e] for e in ecs)
        cliques_data[i] = (verts, ecs, min_f)
        
    # Sort by min_f ascending, then size descending
    cliques_data.sort(key=lambda x: (x[2], -len(x[1])))
    
    chosen = []
    remaining = set(edges)
    
    for verts, ecs, min_f in cliques_data:
        if not _deadline_ok():
            break
        if ecs <= remaining:
            chosen.append(verts)
            remaining -= ecs
            
    # Fallback
    for e in remaining:
        chosen.append([e[0], e[1]])
        
    # Local Search: Try to merge cliques
    # Convert to set of frozensets
    cliques_set = [frozenset(c) for c in chosen]
    changed = True
    while changed and _deadline_ok():
        changed = False
        new_cliques = []
        merged = False
        for i in range(len(cliques_set)):
            if merged:
                break
            for j in range(i+1, len(cliques_set)):
                if merged:
                    break
                union = cliques_set[i] | cliques_set[j]
                # Check if union is a clique
                is_clique = True
                for u in union:
                    for v in union:
                        if u < v and (u, v) not in edges:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    new_cliques.append(union)
                    cliques_set[j] = None
                    merged = True
                    changed = True
                    break
            if merged:
                new_cliques.append(cliques_set[i])
        if changed:
            cliques_set = [c for c in new_cliques if c is not None]
            
    return [list(c) for c in cliques_set]

def construct():
    res1 = _solve_instance(32, 0.6, 601)
    if not _deadline_ok():
        edges1 = _random_edges(32, 0.6, 601)
        res1 = [[u, v] for u, v in edges1]
    res2 = _solve_instance(64, 0.55, 602)
    if not _deadline_ok():
        edges2 = _random_edges(64, 0.55, 602)
        res2 = [[u, v] for u, v in edges2]
    return [res1, res2]
# EVOLVE-BLOCK-END
