import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Quick upper bound: if we can't even find 10 vertices with limited connections
        # Use greedy + branch and bound
        
        def backtrack(candidates, size, depth=0):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            # Upper bound: if remaining candidates can't reach 10
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Pick vertex with max degree in candidates (removes most)
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch: include best_v or exclude
            # Include: remove best_v and its neighbors
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1, depth + 1):
                return True
            # Exclude: remove best_v
            if backtrack(candidates & ~(1 << best_v), size, depth + 1):
                return True
            return False
        
        return backtrack(mask_all, 0)
    
    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i] & ((1 << (i + 1)) - 1) & ~((1 << i) - 1)
            # Actually simpler:
            temp = adj[i]
            j = 0
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges
    
    # Try known constructions first
    
    # Andrasfai graph And(k): vertices Z_{3k}, i~j iff (j-i) mod 3k in {1, 3k-1}
    # And(9) has 27 vertices, alpha=9, triangle-free
    # Actually And(k) has 3k vertices, alpha = k, triangle-free
    # So And(9) has 27 vertices, alpha=9 < 10. Good!
    
    # Try And(k) for various k
    for k in range(10, 14):  # 3k = 30, 33, 36, 39 vertices
        n = 3 * k
        if n > 41:
            break
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # And(k) is triangle-free and has alpha = k
        # For k >= 10, alpha >= 10, so we need k < 10 for alpha < 10
        # But k=9 gives alpha=9 < 10, n=27
        # We need larger n with alpha < 10
        
        if k < 10:
            # alpha = k < 10, so valid
            if not has_independent_set_10(adj, n):
                return {'n': n, 'edges': build_edge_list(adj, n)}
    
    # And(9): n=27, alpha=9
    n = 27
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d == 1 or d == n - 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    if not has_independent_set_10(adj, n):
        return {'n': n, 'edges': build_edge_list(adj, n)}
    
    # Try to construct larger graphs using local search
    # Start with a valid graph and try to add vertices/edges
    
    # Strategy: Start with And(9) on 27 vertices, try to extend to larger n
    # by adding vertices and finding appropriate edges
    
    best_n = 27
    best_adj = adj[:]
    
    # Try random construction for n = 28 to 40
    for target_n in range(28, 41):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random attempts
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            adj = [0] * target_n
            
            # Start with a random partial Andrasfai-like structure
            # or random sparse graph, then add edges greedily
            
            # Generate random permutation for vertex ordering
            order = list(range(target_n))
            random.shuffle(order)
            
            # Add edges in random order, maintaining triangle-free
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)
            
            for u, v in edges:
                # Check if adding (u,v) creates a triangle
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check independent set
            if not has_independent_set_10(adj, target_n):
                found = True
                break
        
        if found:
            best_n = target_n
            best_adj = adj
            return {'n': target_n, 'edges': build_edge_list(adj, target_n)}
    
    # If we haven't found a larger graph, try local search from And(9)
    # Try to add one vertex at a time
    
    current_n = 27
    current_adj = adj[:]
    
    for target_n in range(28, 41):
        if time.time() > deadline:
            break
        
        # Try to extend current graph by adding target_n - current_n vertices
        success = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            new_adj = [0] * target_n
            # Copy existing edges
            for i in range(current_n):
                new_adj[i] = current_adj[i]
            
            # For new vertices, try random connections
            for v in range(current_n, target_n):
                # Randomly connect to some existing vertices
                for u in range(v):
                    if random.random() < 0.3:  # 30% chance to add edge
                        # Check triangle-free
                        common = new_adj[u] & new_adj[v]
                        if common == 0:
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
            
            # Now try to add more edges randomly
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)
            for u, v in edges:
                if not (new_adj[u] & (1 << v)):
                    common = new_adj[u] & new_adj[v]
                    if common == 0:
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
            
            if not has_independent_set_10(new_adj, target_n):
                success = True
                break
        
        if success:
            current_n = target_n
            current_adj = new_adj
            if target_n >= best_n:
                best_n = target_n
                best_adj = new_adj
            return {'n': target_n, 'edges': build_edge_list(new_adj, target_n)}
    
    # Return best found
    if best_n >= 27:
        return {'n': best_n, 'edges': build_edge_list(best_adj, best_n)}
    
    # Fallback: And(9)
    n = 27
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d == 1 or d == n - 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': n, 'edges': build_edge_list(adj, n)}
