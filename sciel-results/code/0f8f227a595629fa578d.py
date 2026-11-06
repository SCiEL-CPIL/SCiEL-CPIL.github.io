# EVOLVE-BLOCK-START
import time
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _bron_kerbosch(adj, n, deadline):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P union X that maximizes |P intersect N(u)|
        union_px = P | X
        if not union_px:
            return
            
        # Find pivot
        pivot = None
        max_common = -1
        # Iterate through vertices in union_px
        temp = union_px
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            common = P & adj[u]
            cnt = common.bit_count()
            if cnt > max_common:
                max_common = cnt
                pivot = u
            if cnt == P.bit_count():
                break
                
        if pivot is None:
            return
            
        # Recurse on P \ N(pivot)
        # We need to iterate over vertices in P that are NOT neighbors of pivot
        # Actually, standard BK iterates over P \ N(pivot) to ensure maximality coverage?
        # No, standard BK: for each v in P \ N(u), recurse with R+{v}, P&N(v), X&N(v)
        # Then remove v from P and add to X
        
        # P \ N(pivot)
        candidates = P & ~adj[pivot]
        
        while candidates:
            if time.time() > deadline:
                return
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # Recurse
            new_R = R | (1 << v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bk(new_R, new_P, new_X)
            
            # Update P and X
            P &= ~(1 << v)
            X |= (1 << v)

    initial_P = (1 << n) - 1
    bk(0, initial_P, 0)
    return cliques

def _partition_graph(adj, n, deadline):
    """Partition edges into cliques to minimize count."""
    # 1. Find all maximal cliques
    max_cliques = _bron_kerbosch(adj, n, deadline)
    
    # 2. Calculate score for each clique
    # We want cliques that cover many edges.
    # A clique of size k covers k*(k-1)/2 edges.
    # Cost is 1 clique.
    # We want to maximize edges covered per clique.
    # Let's sort by size descending, as larger cliques are generally better.
    # Or by (size * (size-1)) / 2.
    
    clique_list = []
    for c in max_cliques:
        size = c.bit_count()
        if size >= 2:
            clique_list.append(c)
            
    # Sort by size descending, then by vertex ids for determinism
    clique_list.sort(key=lambda x: (x.bit_count(), x), reverse=True)
    
    # 3. Greedy selection
    # Represent remaining edges as a set of (u, v) or bitmasks?
    # Since n is small (<=64), we can use a set of edges.
    # Initial edges
    remaining_edges = set()
    for u in range(n):
        neighbors = adj[u]
        # Only count edges u-v where v > u to avoid duplicates
        # Iterate through neighbors
        temp = neighbors
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            if v > u:
                remaining_edges.add((u, v))
                
    selected_cliques = []
    
    # To make it faster, we can pre-compute edges for each clique
    clique_edges = []
    for c in clique_list:
        verts = []
        temp = c
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            verts.append(v)
        edges = set()
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                edges.add((min(verts[i], verts[j]), max(verts[i], verts[j])))
        clique_edges.append(edges)
        
    # Sort cliques by number of edges descending (larger cliques first)
    # This is a simple greedy: pick the clique that covers the most remaining edges
    # We need to re-evaluate "remaining edges covered" for each clique dynamically?
    # That would be O(N_cliques * E) per step, which might be slow.
    
    # Alternative: Just pick the largest clique first, then next largest that doesn't conflict?
    # Conflict means sharing an edge.
    # Since we are partitioning, we cannot share edges.
    
    # Let's try: Sort by size desc. Iterate. If all edges of this clique are in remaining, take it.
    # This is fast but might be suboptimal if a slightly smaller clique allows for many other large ones?
    # Given the density, large cliques are likely disjoint or overlapping in ways that large ones dominate.
    
    # Let's refine: 
    # 1. Get all maximal cliques.
    # 2. Sort by size descending.
    # 3. For each clique in sorted order:
    #    - Check if all its edges are in `remaining_edges`.
    #    - If yes, add to result, remove its edges from `remaining_edges`.
    #    - If no, skip.
    # 4. Cover remaining edges as pairs.
    
    # This is O(N_cliques * E_clique).
    # N_cliques can be large, but for n=64, it's manageable.
    
    for i, c in enumerate(clique_list):
        if time.time() > deadline:
            break
        edges = clique_edges[i]
        # Check if all edges are remaining
        # Optimization: if any edge is not in remaining, skip
        valid = True
        for e in edges:
            if e not in remaining_edges:
                valid = False
                break
        if valid:
            selected_cliques.append(c)
            for e in edges:
                remaining_edges.discard(e)
                
    # Cover remaining edges as pairs
    for u, v in remaining_edges:
        if time.time() > deadline:
            break
        # Add as a clique [u, v]
        selected_cliques.append((1 << u) | (1 << v))
        
    # Convert bitmasks to lists
    result = []
    for c in selected_cliques:
        verts = []
        temp = c
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            verts.append(v)
        result.append(sorted(verts))
        
    return result

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                temp = adj[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
            
        adj = _build_graph(n, p, seed)
        cliques = _partition_graph(adj, n, deadline)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
