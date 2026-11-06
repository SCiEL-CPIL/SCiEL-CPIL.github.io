"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 40
    
    def get_neighbors(adj, u):
        """Return list of neighbors of u in adj."""
        mask = adj[u]
        result = []
        while mask:
            b = (mask & -mask).bit_length() - 1
            result.append(b)
            mask ^= (1 << b)
        return result

    def is_c6_free(u, v, adj, n):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 is u-a-b-c-d-v-u.
        We need to check if there exist distinct a,b,c,d such that:
        a in N(u), d in N(v), b in N(a), c in N(d), b-c is an edge.
        And a,b,c,d are distinct and not u,v.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        a_list = get_neighbors(adj, u)
        d_list = get_neighbors(adj, v)
        
        # Filter to only those in the respective masks
        a_list = [a for a in a_list if (nu_mask >> a) & 1]
        d_list = [d for d in d_list if (nv_mask >> d) & 1]
        
        for a in a_list:
            for d in d_list:
                if a == d:
                    continue
                
                # Candidates for b: neighbors of a excluding u, v, d
                b_candidates = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d)
                
                # Candidates for c: neighbors of d excluding u, v, a
                c_candidates = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if any edge exists between b_candidates and c_candidates
                b_mask = b_candidates
                while b_mask:
                    b = (b_mask & -b_mask).bit_length() - 1
                    b_mask ^= (1 << b)
                    if adj[b] & c_candidates:
                        return False
                        
        return True

    def count_edges(adj, n):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def local_search(adj, n, deadline):
        """
        Try to improve the graph by removing edges that block many additions.
        For each edge e, compute how many other edges would become addable if e were removed.
        Remove the edge with highest "block value" if it leads to net gain.
        """
        edge_count = count_edges(adj, n)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Build list of current edges
            current_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        current_edges.append((u, v))
            
            if not current_edges:
                break
            
            # For each edge, compute how many edges it blocks
            # An edge e=(u,v) blocks an edge (x,y) if removing e allows (x,y) to be added
            best_removal = None
            best_gain = 0
            
            # Sample a subset of edges to check for speed
            sample_size = min(len(current_edges), 30)
            sample = random.sample(current_edges, sample_size) if len(current_edges) > sample_size else current_edges
            
            for (u, v) in sample:
                # Temporarily remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Count how many edges can now be added
                added = 0
                for x in range(n):
                    for y in range(x + 1, n):
                        if not (adj[x] & (1 << y)):
                            if is_c6_free(x, y, adj, n):
                                added += 1
                
                # Restore edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Net gain: we lose 1 edge but gain 'added' edges
                # But this is approximate since adding multiple edges may interact
                gain = added - 1
                if gain > best_gain:
                    best_gain = gain
                    best_removal = (u, v)
            
            if best_removal and best_gain > 0:
                u, v = best_removal
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                improved = True
        
        return adj

    def greedy_add_edges(adj, n, deadline):
        """Greedily add all possible edges in random order."""
        remaining_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    remaining_edges.append((u, v))
        
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if is_c6_free(u, v, adj, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj

    best_adj = None
    best_count = 0
    
    # Multiple restarts with different initializations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        
        # Different initialization strategies
        strategy = restart % 4
        
        if strategy == 0:
            # 8 disjoint K5s
            for k in range(8):
                base = 5 * k
                for a in range(5):
                    for b in range(a + 1, 5):
                        u = base + a
                        v = base + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        elif strategy == 1:
            # Random sparse graph
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.15:
                        if is_c6_free(u, v, adj, n):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        elif strategy == 2:
            # Random sparse graph with higher density
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.25:
                        if is_c6_free(u, v, adj, n):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        else:
            # Start empty
            pass
        
        # Greedy add
        adj = greedy_add_edges(adj, n, deadline)
        
        # Local search improvement
        if time.time() < deadline - 5:
            adj = local_search(adj, n, deadline - 2)
            # After local search, try to add more edges
            adj = greedy_add_edges(adj, n, deadline)
        
        current_count = count_edges(adj, n)
        if current_count > best_count:
            best_count = current_count
            best_adj = list(adj)
        
        # Early stop if we hit the theoretical best
        if best_count >= 105:
            break
    
    # Convert adj to edge list
    if best_adj is None:
        return []
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
