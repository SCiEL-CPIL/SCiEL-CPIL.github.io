import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    rng = random.Random(42)
    
    def bit_count(x):
        return x.bit_count()
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph adj keeps it valid.
        Assumes (u,v) is not currently an edge.
        A K3,3 is formed if there exist A'={a1,a2} subset of N(v)\{u} and B'={b1,b2} subset of N(u)\{v}
        such that all edges between A' and B' exist.
        This is equivalent to checking if the bipartite graph between N(v)\{u} and N(u)\{v} contains a K2,2.
        Which means: do any two vertices in N(v)\{u} have at least 2 common neighbors in N(u)\{v}?
        """
        # Candidates for A' (neighbors of v, excluding u)
        # Candidates for B' (neighbors of u, excluding v)
        
        # Get neighbors of v excluding u
        nv = adj[v] & ~(1 << u)
        # Get neighbors of u excluding v
        nu = adj[u] & ~(1 << v)
        
        # Iterate over pairs in nv
        # To be efficient, we can iterate over bits in nv
        # Let a1, a2 be two distinct vertices in nv.
        # We need to check if |N(a1) & N(a2) & nu| >= 2.
        
        # List of vertices in nv
        # Since n=20, we can just loop
        verts_nv = []
        mask = nv
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            verts_nv.append(idx)
            mask ^= lsb
            
        for i in range(len(verts_nv)):
            a1 = verts_nv[i]
            for j in range(i + 1, len(verts_nv)):
                a2 = verts_nv[j]
                # Common neighbors of a1 and a2 that are also in nu
                common = adj[a1] & adj[a2] & nu
                if bit_count(common) >= 2:
                    return False
        
        return True

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_existing_edges(adj):
        """Return list of existing edges."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def get_missing_edges(adj):
        """Return list of missing edges."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    edges.append((i, j))
        return edges

    def local_search(adj, max_iters=100):
        """
        Perform local search: try adding edges, then try swaps.
        Returns (adj, edge_count).
        """
        cur_edges = count_edges(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Phase 1: Add-only
            missing = get_missing_edges(adj)
            rng.shuffle(missing)
            for (u, v) in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
            
            # Phase 2: Swap
            if not improved:
                existing = get_existing_edges(adj)
                missing = get_missing_edges(adj)
                rng.shuffle(existing)
                rng.shuffle(missing)
                
                # Try a limited number of swaps to save time
                swap_limit = 50
                swaps_tried = 0
                for e1 in existing:
                    if swaps_tried >= swap_limit:
                        break
                    e1_u, e1_v = e1
                    for e2 in missing:
                        if swaps_tried >= swap_limit:
                            break
                        e2_u, e2_v = e2
                        
                        # Skip if same edge
                        if (e1_u, e1_v) == (e2_u, e2_v) or (e1_u, e1_v) == (e2_v, e2_u):
                            continue
                        
                        # Try swap
                        remove_edge(adj, e1_u, e1_v)
                        if is_valid_add(adj, e2_u, e2_v):
                            add_edge(adj, e2_u, e2_v)
                            improved = True
                            # After one successful swap, restart the loop to re-evaluate
                            break
                        else:
                            add_edge(adj, e1_u, e1_v)
                        swaps_tried += 1
                    if improved:
                        break
                        
        return adj, cur_edges

    best_edges = 0
    best_adj = None
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    num_starts = 100
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # 1. Greedy initialization with random order
        adj = [0] * n
        edge_order = all_edges[:]
        rng.shuffle(edge_order)
        for (u, v) in edge_order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        
        # 2. Local Search
        adj, cur_edges = local_search(adj)
        
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
            
    if best_adj is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
