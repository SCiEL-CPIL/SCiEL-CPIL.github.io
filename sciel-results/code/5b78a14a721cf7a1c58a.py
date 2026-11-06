import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Greedy check first to quickly identify graphs with large alpha
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
            
        if size >= 10:
            return True
            
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
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
            
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            if backtrack(candidates & ~(1 << best_v), size):
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
    
    # Try to find a valid graph for n=39 down to 28
    for target_n in range(39, 27, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            adj = [0] * target_n
            
            # Generate random triangle-free graph
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)
            
            for u, v in edges:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check independent set
            if not has_independent_set_10(adj, target_n):
                found = True
                break
        
        if found:
            return {'n': target_n, 'edges': build_edge_list(adj, target_n)}

    # Fallback: Andrasfai graph And(9) with n=27
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
        
    # Should not reach here
    return {'n': 1, 'edges': []}
