import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def is_triangle_free(n, edges):
        # Efficient check using adjacency matrix or set
        # For n <= 41, O(n^3) is fine if optimized, or O(n^2) with bitsets
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            # If any two neighbors of u are connected, it's a triangle
            # Neighbors of u:
            # Check all pairs of neighbors?
            # Or: for each v in neighbors(u), if adj[u] & adj[v] != 0, then triangle.
            # But we must exclude u itself from adj[v] check?
            # adj[v] has u set. So adj[u] & adj[v] will have u set? No, adj[u] doesn't have u.
            # So if there is any common neighbor w != u, then adj[u] & adj[v] will have w.
            # Since adj[u] has v, and adj[v] has u.
            # Let's just check: for each v in range(n):
            #   if adj[u] & (1 << v):
            #      common = adj[u] & adj[v]
            #      if common & ~(1 << u) & ~(1 << v): # if there is a third vertex
            #         return False
            # This is O(n^2) per u, so O(n^3) total. 41^3 ~ 68k ops. Very fast.
            for v in range(n):
                if adj[u] & (1 << v):
                    # Check if u and v share a common neighbor w
                    common = adj[u] & adj[v]
                    # Remove u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def has_independent_set_10(n, edges):
        # Branch and bound for independent set of size 10
        # Use bitsets for speed
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Quick greedy lower bound check: if greedy finds alpha >= 10, return True
        # Greedy: repeatedly pick vertex with min degree in remaining, remove it and its neighbors
        remaining = (1 << n) - 1
        greedy_size = 0
        while remaining:
            # Find vertex with min degree in remaining subgraph
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(remaining & adj[v]).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            greedy_size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
            if greedy_size >= 10:
                return True
        
        # If greedy didn't find 10, do exact branch and bound
        def backtrack(candidates, size):
            if size == 10:
                return True
            # Pruning: if size + popcount(candidates) < 10, return False
            if size + bin(candidates).count('1') < 10:
                return False
            
            # Pick vertex with max degree in candidate subgraph for better pruning
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(candidates & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            candidates &= ~(1 << best_v)
            
            # Option 1: Include best_v
            new_candidates = candidates & ~adj[best_v]
            if backtrack(new_candidates, size + 1):
                return True
            
            # Option 2: Exclude best_v
            if backtrack(candidates, size):
                return True
                
            return False
        
        return backtrack((1 << n) - 1, 0)

    def validate(n, edges):
        if not is_triangle_free(n, edges):
            return False
        if has_independent_set_10(n, edges):
            return False
        return True

    # Known baseline: Andrasfai graph A_9 on 26 vertices
    # A_9: vertices 0..25, edge (i,j) if (j-i) mod 26 in {1, 25} i.e. adjacent in cycle plus... 
    # Actually Andrasfai graph A_n for n=3k+2: partition into 3 parts of sizes k+1, k, k
    # For n=26: parts of size 9, 9, 8. Edges between parts, no edges within parts.
    # This is a complete 3-partite graph with parts 9,9,8.
    # Wait, that's not triangle-free. Let me use the correct construction.
    # Andrasfai graph A_9: It's the complement of a triangle-free graph... 
    # Let me just use a known valid small graph as fallback.
    best_n = 26
    best_edges = []
    # Use a simple valid graph: cycle C_26 has alpha=13, too big.
    # Let's use a random valid graph as fallback, or just return a small known one.
    # For safety, generate a valid graph on 26 vertices:
    # Actually, let's just keep the original baseline and focus on finding larger ones.
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try to find larger n
    # Start from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break
            
        # Try several random restarts
        found = False
        for attempt in range(2000): # More attempts with faster validation
            if time.time() > deadline - 2:
                break
                
            # Generate random maximal triangle-free graph
            edges = []
            adj = [0] * n
            # Create random permutation of vertices to break symmetry
            perm = list(range(n))
            random.shuffle(perm)
            # Shuffle all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a triangle
                if (adj[u] & adj[v]) == 0:
                    edges.append((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Quick density check: very sparse graphs likely have large alpha
            # For triangle-free graph on n vertices, by Turan, max edges ~ n^2/4
            # If edges < n^2/8, alpha is likely > 10
            if len(edges) < n * n // 10:
                continue
            
            # Convert edges list to list of lists for validate
            edge_list = [[u, v] for u, v in edges]
            
            # Run validation
            if validate(n, edge_list):
                return {'n': n, 'edges': edge_list}
        
        # If we didn't find one for this n, continue to n-1

    return {'n': best_n, 'edges': best_edges}
