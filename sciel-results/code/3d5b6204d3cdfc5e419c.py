import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Fallback: 8 disjoint K4s (n=32, alpha=8, C5-free)
    def get_fallback():
        n = 32
        edges = []
        for b in range(8):
            offset = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([offset + i, offset + j])
        return {'n': n, 'edges': edges}
    
    # Try to construct a valid 33-vertex graph
    # Strategy: 8 K4s + vertex 32 connected to all of K4_0..K4_3 and one vertex of K4_4..K4_7
    n = 33
    edges = []
    
    # Add K4 cliques
    for b in range(8):
        offset = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([offset + i, offset + j])
    
    # Connect vertex 32 to all 4 vertices of K4_0, K4_1, K4_2, K4_3
    for b in range(4):
        for i in range(4):
            edges.append([32, 4 * b + i])
    
    # Connect vertex 32 to one vertex (the first) of K4_4, K4_5, K4_6, K4_7
    for b in range(4, 8):
        edges.append([32, 4 * b])
    
    # Verify the graph
    # Build adjacency matrix
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    # Check for C5: iterate over all 5-tuples
    # For n=33, C(33,5) = 237,336 which is manageable
    found_c5 = False
    for a in range(n - 4):
        if time.time() > deadline:
            return get_fallback()
        for b in range(a + 1, n - 3):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n - 2):
                if not adj[b][c]:
                    continue
                for d in range(c + 1, n - 1):
                    if not adj[c][d]:
                        continue
                    for e in range(d + 1, n):
                        if not adj[d][e]:
                            continue
                        if adj[e][a]:
                            found_c5 = True
                            break
                    if found_c5:
                        break
                if found_c5:
                    break
            if found_c5:
                break
        if found_c5:
            break
    
    if found_c5:
        # This construction has a C5, fall back to 32
        return get_fallback()
    
    # Check independence number using branch and bound
    # We need alpha < 9, i.e., alpha <= 8
    # Use a simple exact check via complement graph coloring or direct search
    
    def has_independent_set_of_size_k(adj, n, k):
        """Check if there's an independent set of size k."""
        # Greedy + backtracking
        remaining = list(range(n))
        
        def backtrack(remaining, current_size, target):
            if current_size >= target:
                return True
            if len(remaining) < target - current_size:
                return False
            
            # Pick a vertex
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: include v
            # Remove v and its neighbors from consideration
            new_remaining = []
            for u in rest:
                if not adj[v][u]:
                    new_remaining.append(u)
            if backtrack(new_remaining, current_size + 1, target):
                return True
            
            # Option 2: exclude v
            if backtrack(rest, current_size, target):
                return True
            
            return False
        
        return backtrack(remaining, 0, k)
    
    if has_independent_set_of_size_k(adj, n, 9):
        # Has independent set of size 9, fall back
        return get_fallback()
    
    # Valid 33-vertex graph found!
    return {'n': n, 'edges': edges}
