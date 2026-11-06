import itertools
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115
    
    def find_k33(adj):
        """Find one K3,3 subgraph in the graph represented by adj bitmasks."""
        # Iterate over all 6-vertex subsets
        for verts in itertools.combinations(range(n), 6):
            # There are 10 ways to partition 6 vertices into two sets of 3
            # We only need to check unique partitions (up to swapping A and B)
            # Partitions of {0,1,2,3,4,5} into {0,1,2} and {3,4,5} etc.
            # Indices in verts: 0..5
            # Pairs of indices (i,j) for A, rest for B.
            # Since A and B are interchangeable, we can just check:
            # A = {v0, v1, v2}, B = {v3, v4, v5}
            # A = {v0, v1, v3}, B = {v2, v4, v5}
            # ...
            # There are C(6,3)/2 = 10 such partitions.
            
            v = verts
            # Generate the 10 partitions
            # We can hardcode the index sets for A
            # 0,1,2 vs 3,4,5
            # 0,1,3 vs 2,4,5
            # 0,1,4 vs 2,3,5
            # 0,1,5 vs 2,3,4
            # 0,2,3 vs 1,4,5
            # 0,2,4 vs 1,3,5
            # 0,2,5 vs 1,3,4
            # 0,3,4 vs 1,2,5
            # 0,3,5 vs 1,2,4
            # 0,4,5 vs 1,2,3
            
            partitions = [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 3], [2, 4, 5]),
                ([0, 1, 4], [2, 3, 5]),
                ([0, 1, 5], [2, 3, 4]),
                ([0, 2, 3], [1, 4, 5]),
                ([0, 2, 4], [1, 3, 5]),
                ([0, 2, 5], [1, 3, 4]),
                ([0, 3, 4], [1, 2, 5]),
                ([0, 3, 5], [1, 2, 4]),
                ([0, 4, 5], [1, 2, 3])
            ]
            
            for a_idx, b_idx in partitions:
                A = [v[i] for i in a_idx]
                B = [v[i] for i in b_idx]
                
                # Check if all edges between A and B exist
                is_k33 = True
                for u in A:
                    for w in B:
                        if not ((adj[u] >> w) & 1):
                            is_k33 = False
                            break
                    if not is_k33:
                        break
                if is_k33:
                    # Return the edges of this K3,3
                    edges = []
                    for u in A:
                        for w in B:
                            edges.append((u, w))
                    return edges
        return None

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        
        bit_u = 1 << u
        bit_v = 1 << v
        
        # Case 1: u, v on opposite sides
        # A = {u, x, y}, B = {v, p, q}
        # x, y in N(v) \ {u}
        # p, q in N(u) \ {v}
        # p, q in N(x) \cap N(y)
        # So | N(u) \cap N(x) \cap N(y) | >= 2
        
        nv = adj[v] & ~bit_u
        nv_list = []
        mask = nv
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            nv_list.append(idx)
            mask ^= lsb
            
        for i in range(len(nv_list)):
            x = nv_list[i]
            for j in range(i+1, len(nv_list)):
                y = nv_list[j]
                common = adj[x] & adj[y] & adj[u]
                if common.bit_count() >= 2:
                    return False
                    
        # Case 2: u, v on same side
        # A = {u, v, z}, B = {p, q, r}
        # p, q, r in N(u) \cap N(v) \cap N(z)
        # So | N(u) \cap N(v) \cap N(z) | >= 3
        
        common_uv = adj[u] & adj[v]
        for z in range(n):
            if z == u or z == v:
                continue
            common = common_uv & adj[z]
            if common.bit_count() >= 3:
                return False
                
        return True

    def rebuild_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return edges

    def count_edges(adj):
        return sum(bin(x).count("1") for x in adj) // 2

    def local_search(adj):
        nonlocal deadline
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add edges
            for u in range(n):
                if time.time() >= deadline - 5:
                    break
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
                if improved:
                    break
            
            # If no edge could be added, try removing an edge and see if we can add more
            if not improved and time.time() < deadline - 10:
                edges = rebuild_edges(adj)
                if not edges:
                    break
                
                # Try up to 5 random removals
                for _ in range(5):
                    if time.time() >= deadline - 10:
                        break
                    u, v = random.choice(edges)
                    adj_copy = adj[:]
                    
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Try to add edges again
                    imp = True
                    while imp and time.time() < deadline - 10:
                        imp = False
                        for u2 in range(n):
                            if time.time() >= deadline - 10:
                                break
                            for v2 in range(u2+1, n):
                                if not ((adj[u2] >> v2) & 1) and can_add_edge(adj, u2, v2):
                                    adj[u2] |= (1 << v2)
                                    adj[v2] |= (1 << u2)
                                    imp = True
                                    break
                            if imp:
                                break
                    
                    new_edges = count_edges(adj)
                    old_edges = count_edges(adj_copy)
                    
                    if new_edges > old_edges:
                        improved = True
                    else:
                        adj = adj_copy
                        break
                        
        return adj

    best_adj = None
    best_count = 0
    
    random.seed(42)
    
    # Multiple restarts with different removal strategies
    for restart in range(50):
        if time.time() >= deadline - 10:
            break
            
        # Start with complete graph
        adj = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v:
                    adj[u] |= (1 << v)
        
        # Greedy removal of K3,3s
        while time.time() < deadline - 20:
            k33_edges = find_k33(adj)
            if k33_edges is None:
                break
            
            # Pick an edge to remove
            # Heuristic: Remove edge with highest sum of degrees
            best_edge = None
            best_score = -1
            for u, v in k33_edges:
                score = bin(adj[u]).count("1") + bin(adj[v]).count("1")
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                break
        
        # Local search to add edges back
        adj = local_search(adj)
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
    if best_adj is None:
        # Fallback
        best_adj = [0] * n
        
    return rebuild_edges(best_adj)
