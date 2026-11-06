import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Helper: check if adding edge (u, v) creates a triangle
    def is_valid_add(u, v, adj):
        if (adj[u] >> v) & 1:
            return False
        if (adj[u] & adj[v]):
            return False
        return True
    
    # Helper: exact 3-colorability check using bitmask backtracking
    def is_3_colorable(adj):
        # Try all 3-colorings via backtracking with bitmasks
        colors = [-1] * n
        uncolored = (1 << n) - 1
        
        def backtrack(uncolored):
            if uncolored == 0:
                return True
            
            # Pick vertex with fewest available colors (MRV heuristic)
            best_v = -1
            best_avail = 3
            for v in range(n):
                if (uncolored >> v) & 1:
                    avail = 3
                    for c in range(3):
                        # Check if color c conflicts with any neighbor already colored
                        conflict = False
                        for u in range(n):
                            if (adj[v] >> u) & 1 and colors[u] == c:
                                conflict = True
                                break
                        if not conflict:
                            avail -= 1
                    if avail < best_avail:
                        best_avail = avail
                        best_v = v
                        if avail == 0:
                            return False
            
            if best_v == -1:
                return True
            
            for c in range(3):
                # Check if color c is available for best_v
                ok = True
                for u in range(n):
                    if (adj[best_v] >> u) & 1 and colors[u] == c:
                        ok = False
                        break
                if ok:
                    colors[best_v] = c
                    if backtrack(uncolored & ~(1 << best_v)):
                        return True
                    colors[best_v] = -1
            
            return False
        
        return backtrack(uncolored)
    
    # Strategy: Start from complete graph, remove edges to eliminate triangles
    # Use dynamic triangle counting to guide removals
    
    def greedy_triangle_free(adj_start):
        """Start from adj_start, remove edges to make triangle-free, maximizing remaining edges."""
        adj = adj_start[:]
        # Count triangles containing each edge
        # For edge (u,v), triangle count = popcount(adj[u] & adj[v])
        
        # Build list of all edges with their triangle counts
        edges = {}
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges[(u, v)] = bin(adj[u] & adj[v]).count('1')
        
        # Iteratively remove the edge with the fewest triangles (but > 0)
        # If all edges have 0 triangles, we're done
        while True:
            if time.time() > deadline:
                break
            
            # Find edge with minimum positive triangle count
            min_count = float('inf')
            min_edge = None
            for e, c in edges.items():
                if 0 < c < min_count:
                    min_count = c
                    min_edge = e
            
            if min_edge is None:
                break  # No triangles left
            
            # Remove min_edge
            u, v = min_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            del edges[(u, v)]
            
            # Update triangle counts for affected edges
            # For any edge (u, w) where w != v: new count = old count - [if v was in common neighbors]
            # Actually, triangle count for edge (a,b) = popcount(adj[a] & adj[b])
            # When we remove edge (u,v), we need to update counts for edges involving u or v
            
            # Update edges incident to u
            for w in range(n):
                if w != u and w != v and (adj[u] >> w) & 1:
                    e = (min(u, w), max(u, w))
                    if e in edges:
                        # Old count included v if v was neighbor of w
                        # New count = popcount(adj[u] & adj[w]) after removal
                        edges[e] = bin(adj[u] & adj[w]).count('1')
            
            # Update edges incident to v
            for w in range(n):
                if w != u and w != v and (adj[v] >> w) & 1:
                    e = (min(v, w), max(v, w))
                    if e in edges:
                        edges[e] = bin(adj[v] & adj[w]).count('1')
        
        return adj, edges
    
    # Multiple strategies to try
    
    best_count = 0
    best_edges = []
    
    # Strategy 1: Greedy triangle removal from complete graph
    complete_adj = [(1 << n) - 1] * n
    for i in range(n):
        complete_adj[i] &= ~(1 << i)
    
    adj1, edges1 = greedy_triangle_free(complete_adj)
    count1 = len(edges1)
    if count1 > best_count:
        # Check 3-colorability
        if not is_3_colorable(adj1):
            best_count = count1
            best_edges = [[u, v] for u, v in edges1.keys()]
    
    # Strategy 2: Randomized greedy with different tie-breaking
    for restart in range(500):
        if time.time() > deadline:
            break
        
        adj = complete_adj[:]
        for i in range(n):
            adj[i] &= ~(1 << i)
        
        edges = {}
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges[(u, v)] = bin(adj[u] & adj[v]).count('1')
        
        while True:
            if time.time() > deadline:
                break
            
            # Find all edges with minimum positive triangle count
            min_count = float('inf')
            for e, c in edges.items():
                if 0 < c < min_count:
                    min_count = c
            
            if min_count == float('inf'):
                break
            
            # Randomly pick from edges with min_count
            candidates = [e for e, c in edges.items() if c == min_count]
            if not candidates:
                break
            
            min_edge = random.choice(candidates)
            u, v = min_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            del edges[(u, v)]
            
            # Update triangle counts
            for w in range(n):
                if w != u and w != v and (adj[u] >> w) & 1:
                    e = (min(u, w), max(u, w))
                    if e in edges:
                        edges[e] = bin(adj[u] & adj[w]).count('1')
            
            for w in range(n):
                if w != u and w != v and (adj[v] >> w) & 1:
                    e = (min(v, w), max(v, w))
                    if e in edges:
                        edges[e] = bin(adj[v] & adj[w]).count('1')
        
        count = len(edges)
        if count > best_count:
            if not is_3_colorable(adj):
                best_count = count
                best_edges = [[u, v] for u, v in edges.keys()]
    
    # Strategy 3: Start from Grötzsch graph seed and add edges
    seed_edges = []
    for i in range(5):
        seed_edges.append((i, (i + 1) % 5))
    for u, w in [(i, (i + 1) % 5) for i in range(5)]:
        seed_edges.append((u, 5 + w))
        seed_edges.append((w, 5 + u))
    for u in range(5):
        seed_edges.append((5 + u, 10))
        
    adj_seed = [0] * n
    for u, v in seed_edges:
        adj_seed[u] |= 1 << v
        adj_seed[v] |= 1 << u
        
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        current_adj = adj_seed[:]
        current_edges = set(seed_edges)
        
        candidates = []
        for u, v in all_edges:
            if (u, v) not in current_edges:
                if is_valid_add(u, v, current_adj):
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        for u, v in candidates:
            if is_valid_add(u, v, current_adj):
                current_adj[u] |= 1 << v
                current_adj[v] |= 1 << u
                current_edges.add((u, v))
        
        count = len(current_edges)
        if count > best_count:
            if not is_3_colorable(current_adj):
                best_count = count
                best_edges = [[u, v] for u, v in current_edges]
    
    # If no valid graph found, return empty (shouldn't happen)
    if not best_edges:
        # Fallback: return Grötzsch graph extended with isolated vertices
        return [[u, v] for u, v in seed_edges]
    
    return best_edges
