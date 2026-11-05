import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def can_add_edge(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4."""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        # C4 exists iff ∃w∈N(u) such that N(w) ∩ N(v) ≠ ∅
        mask_u = adj[u]
        mask_v = adj[v]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if adj[w] & mask_v:
                return False
            m ^= lsb
        return True
    
    def greedy_boost(adj, degrees, n, min_deg, deadline, max_iterations=50000):
        """Try to boost degrees to meet min_deg using greedy edge addition."""
        for _ in range(max_iterations):
            if time.time() > deadline:
                return False
            
            # Find vertices below target
            low_verts = [i for i in range(n) if degrees[i] < min_deg]
            if not low_verts:
                return True
            
            # Pick the vertex with lowest degree
            u = min(low_verts, key=lambda i: degrees[i])
            
            # Try all possible partners in random order
            partners = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
            random.shuffle(partners)
            
            for v in partners:
                if can_add_edge(adj, n, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    break
        
        return min(degrees) >= min_deg
    
    def local_search(adj, degrees, n, min_deg, deadline, max_swaps=10000):
        """Local search: try to swap edges to improve minimum degree."""
        # Build edge list from adj
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                m ^= lsb
        
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            swap_count += 1
            
            # Find the vertex with minimum degree
            min_u = min(range(n), key=lambda i: degrees[i])
            
            if degrees[min_u] >= min_deg:
                return True
            
            # Try to find an edge to add for min_u
            added = False
            partners = [v for v in range(n) if v != min_u and not ((adj[min_u] >> v) & 1)]
            random.shuffle(partners)
            
            for v in partners:
                if can_add_edge(adj, n, min_u, v):
                    # Check if this would help: we need to remove some edge first
                    # if it causes issues, but for now just try to add
                    adj[min_u] |= (1 << v)
                    adj[v] |= (1 << min_u)
                    degrees[min_u] += 1
                    degrees[v] += 1
                    improved = True
                    added = True
                    break
            
            if not added:
                # Try removing an edge from a high-degree vertex and adding a different one
                high_verts = [i for i in range(n) if degrees[i] > min_deg]
                if high_verts:
                    high_u = random.choice(high_verts)
                    # Find an edge to remove from high_u
                    m = adj[high_u]
                    removable = []
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        removable.append(w)
                        m ^= lsb
                    
                    if removable:
                        remove_v = random.choice(removable)
                        # Remove edge (high_u, remove_v)
                        adj[high_u] &= ~(1 << remove_v)
                        adj[remove_v] &= ~(1 << high_u)
                        degrees[high_u] -= 1
                        degrees[remove_v] -= 1
                        
                        # Now try to add an edge for min_u
                        partners2 = [v for v in range(n) if v != min_u and not ((adj[min_u] >> v) & 1)]
                        random.shuffle(partners2)
                        
                        for v in partners2:
                            if can_add_edge(adj, n, min_u, v):
                                adj[min_u] |= (1 << v)
                                adj[v] |= (1 << min_u)
                                degrees[min_u] += 1
                                degrees[v] += 1
                                improved = True
                                break
                        else:
                            # Undo the removal
                            adj[high_u] |= (1 << remove_v)
                            adj[remove_v] |= (1 << high_u)
                            degrees[high_u] += 1
                            degrees[remove_v] += 1
        
        return min(degrees) >= min_deg
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        max_restarts = 5000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            degrees = [0] * n
            
            # Initialize with a random sparse graph to break symmetry
            initial_edges = random.randint(0, n * min_deg // 2)
            for _ in range(initial_edges):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and not ((adj[u] >> v) & 1) and can_add_edge(adj, n, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
            
            # Greedy boost
            if greedy_boost(adj, degrees, n, min_deg, deadline):
                # Build edge list
                edges = []
                for u in range(n):
                    m = adj[u]
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        m ^= lsb
                return {'n': n, 'edges': edges}
            
            # Local search
            if local_search(adj, degrees, n, min_deg, deadline):
                # Build edge list
                edges = []
                for u in range(n):
                    m = adj[u]
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        m ^= lsb
                return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    # For n=39, min_deg=0, empty graph works
    return {'n': 39, 'edges': []}
