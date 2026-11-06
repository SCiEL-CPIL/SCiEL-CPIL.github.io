import time
import random
import sys

def construct():
    """
    Construct a triangle-free graph with independence number < 10, maximizing n.
    Uses a combination of:
    1. Known constructions (Andrasfai, Clebsch, etc.)
    2. Random search with local improvements
    3. Spectral-inspired initialization
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def build_adj(edges, n):
        """Build adjacency sets from edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def count_triangles(adj, n):
        """Count triangles in graph."""
        count = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            count += 1
        return count
    
    def max_independent_set_size(adj, n, timeout=5.0):
        """
        Branch and bound to find max independent set size.
        Uses bitmask for efficiency when n <= 40.
        """
        if n <= 1:
            return n
        
        # Convert to bitmask representation
        # Bit i is set if vertex i is in the set
        # We want to find maximum size independent set
        
        # Precompute: for each vertex, which other vertices it's NOT adjacent to
        # (i.e., which vertices can be in an independent set with it)
        non_adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and j not in adj[i]:
                    mask |= (1 << j)
            non_adj[i] = mask
        
        best = [0]
        start_time = time.time()
        
        def dfs(remaining_mask, current_size):
            nonlocal best
            if time.time() - start_time > timeout:
                return
            
            if remaining_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + popcount(remaining_mask)
            if current_size + bin(remaining_mask).count('1') <= best[0]:
                return
            
            # Pick vertex with highest degree in remaining set
            # (this helps prune)
            v = -1
            max_deg = -1
            temp_mask = remaining_mask
            while temp_mask:
                bit = temp_mask & (-temp_mask)
                idx = bit.bit_length() - 1
                # Count neighbors of idx in remaining_mask
                deg = bin(adj_mask[idx] & remaining_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp_mask ^= bit
            
            if v == -1:
                return
            
            # Option 1: Include v
            new_remaining = remaining_mask & non_adj[v]
            new_remaining &= ~(1 << v)
            dfs(new_remaining, current_size + 1)
            
            # Option 2: Exclude v
            remaining_mask &= ~(1 << v)
            dfs(remaining_mask, current_size)
        
        # Precompute adjacency masks
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        all_mask = (1 << n) - 1
        dfs(all_mask, 0)
        return best[0]
    
    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        return True
    
    def is_valid(adj, n, alpha_timeout=5.0):
        """Check if graph is triangle-free and alpha < 10."""
        if not is_triangle_free(adj, n):
            return False
        alpha = max_independent_set_size(adj, n, alpha_timeout)
        return alpha < 10
    
    def try_extend(adj, n, new_n, max_attempts=200):
        """
        Try to extend graph from n to new_n vertices.
        Returns (success, new_adj, new_edges) or (False, None, None).
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Strategy: pick a random subset S of existing vertices
            # The new vertex connects to S
            # For triangle-free: S must be independent in G
            # For alpha constraint: need to check
            
            # Generate random independent set
            S = set()
            remaining = list(range(n))
            random.shuffle(remaining)
            
            for v in remaining:
                if all(v not in adj[u] for u in S):
                    S.add(v)
                    if len(S) >= random.randint(1, min(5, n)):
                        break
            
            if not S:
                continue
            
            # Create new adjacency
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(n)
            
            # Check validity
            if is_valid(new_adj, new_n, alpha_timeout=3.0):
                return (True, new_adj, [(n, u) for u in S])
        
        return (False, None, None)
    
    def local_search(adj, n, max_iters=5000):
        """
        Local search: try to improve the graph by adding/removing edges
        to maximize n while maintaining triangle-free and alpha < 10.
        """
        best_n = n
        best_adj = [set(s) for s in adj]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add a vertex
            if n < 40:
                success, new_adj, new_edges = try_extend(adj, n, n + 1, max_attempts=50)
                if success:
                    adj = new_adj
                    n += 1
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    continue
            
            # Try edge modifications: remove an edge to allow more vertices
            if n >= best_n and random.random() < 0.3:
                # Pick a random edge to remove
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if j > i:
                            edges.append((i, j))
                if edges:
                    u, v = random.choice(edges)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    # Check if still valid
                    if is_triangle_free(adj, n) and max_independent_set_size(adj, n, 3.0) < 10:
                        # Try to extend
                        success, new_adj, new_edges = try_extend(adj, n, n + 1, max_attempts=50)
                        if success:
                            adj = new_adj
                            n += 1
                            best_n = n
                            best_adj = [set(s) for s in adj]
                        else:
                            # Revert
                            adj[u].add(v)
                            adj[v].add(u)
                    else:
                        # Revert
                        adj[u].add(v)
                        adj[v].add(u)
        
        return best_n, best_adj
    
    # Start with known constructions
    best_n = 0
    best_adj = None
    best_edges = []
    
    # Construction 1: Andrasfai graph A_9 (n=26)
    n1 = 26
    edges1 = []
    for i in range(n1):
        for j in range(i + 1, n1):
            if (j - i) % 3 == 1:
                edges1.append([i, j])
    adj1 = build_adj(edges1, n1)
    if is_valid(adj1, n1, alpha_timeout=10.0):
        best_n = n1
        best_adj = [set(s) for s in adj1]
        best_edges = edges1[:]
    
    # Construction 2: Clebsch graph (n=16, triangle-free, alpha=5)
    n2 = 16
    edges2 = []
    # Clebsch graph: vertices are binary strings of length 5 with odd weight
    # Two vertices adjacent if Hamming distance is 2
    verts = []
    for mask in range(32):
        if bin(mask).count('1') % 2 == 1:
            verts.append(mask)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            ham = bin(verts[i] ^ verts[j]).count('1')
            if ham == 2:
                edges2.append([i, j])
    adj2 = build_adj(edges2, n2)
    if is_valid(adj2, n2, alpha_timeout=10.0):
        if n2 > best_n:
            best_n = n2
            best_adj = [set(s) for s in adj2]
            best_edges = edges2[:]
    
    # Construction 3: Random triangle-free graphs
    for trial in range(20):
        if time.time() > deadline:
            break
        
        n3 = random.randint(20, 35)
        # Generate random graph and remove triangles
        adj3 = [set() for _ in range(n3)]
        for i in range(n3):
            for j in range(i + 1, n3):
                if random.random() < 0.4:
                    adj3[i].add(j)
                    adj3[j].add(i)
        
        # Remove triangles by deleting edges
        for _ in range(100):
            triangles_found = False
            for i in range(n3):
                for j in list(adj3[i]):
                    if j > i:
                        for k in list(adj3[j]):
                            if k > j and k in adj3[i]:
                                # Triangle i-j-k, remove edge (j,k)
                                adj3[j].discard(k)
                                adj3[k].discard(j)
                                triangles_found = True
            if not triangles_found:
                break
        
        if is_valid(adj3, n3, alpha_timeout=5.0):
            if n3 > best_n:
                best_n = n3
                best_adj = [set(s) for s in adj3]
                best_edges = []
                for i in range(n3):
                    for j in adj3[i]:
                        if j > i:
                            best_edges.append([i, j])
    
    # Now try to extend from the best graph found
    if best_adj is not None:
        adj = [set(s) for s in best_adj]
        n = best_n
        
        # Try direct extension
        for target_n in range(n + 1, 41):
            if time.time() > deadline:
                break
            
            success, new_adj, new_edges = try_extend(adj, n, target_n, max_attempts=300)
            if success:
                adj = new_adj
                n = target_n
                # Update best
                best_n = n
                best_adj = [set(s) for s in adj]
                best_edges.extend(new_edges)
            else:
                break
        
        # Try local search
        if time.time() < deadline - 30:
            ls_n, ls_adj = local_search(adj, n, max_iters=20000)
            if ls_n > best_n:
                best_n = ls_n
                best_adj = [set(s) for s in ls_adj]
                best_edges = []
                for i in range(best_n):
                    for j in best_adj[i]:
                        if j > i:
                            best_edges.append([i, j])
    
    # Final: build edge list from best_adj
    if best_adj is not None:
        final_edges = []
        for i in range(best_n):
            for j in best_adj[i]:
                if j > i:
                    final_edges.append([i, j])
    else:
        # Fallback: trivial valid graph
        best_n = 9
        final_edges = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if (j - i) % 3 != 0:
                    final_edges.append([i, j])
    
    return {'n': best_n, 'edges': final_edges}
