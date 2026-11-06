import random
import time

def construct():
    n = 16
    vertices = list(range(n))
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    # Precompute neighbors for fast triangle check
    # We will maintain adjacency sets
    
    def is_3_colorable(adj):
        # Backtracking search for 3-coloring
        colors = [-1] * n
        # Order vertices by degree (descending) to prune faster
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if color c is used by any neighbor
                conflict = False
                for u in adj[v]:
                    if colors[u] == c:
                        conflict = True
                        break
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    def creates_triangle(adj, u, v):
        # Check if there is a common neighbor
        # adj[u] and adj[v] are sets
        return bool(adj[u] & adj[v])

    best_edges = []
    best_count = 0
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    while time.time() < deadline:
        adj = [set() for _ in range(n)]
        edges = set()
        
        # Random greedy construction
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not creates_triangle(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
        
        # Check if 3-colorable
        if not is_3_colorable(adj):
            # It is 4-chromatic (or higher, but triangle-free implies max chi is high? 
            # Actually, triangle-free graphs can have high chromatic number. 
            # But if it's not 3-colorable, it's at least 4.)
            # Now, greedily add all possible edges
            changed = True
            while changed:
                changed = False
                # Iterate through all possible edges not in edges
                # To make it faster, we can just iterate all_edges and check
                for u, v in all_edges:
                    if (u, v) not in edges:
                        if not creates_triangle(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            edges.add((u, v))
                            changed = True
                            
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                
        # Optional: Local search if not 3-colorable?
        # The hint suggests 1-1 swaps.
        # If the graph is 3-colorable, we might try to swap edges to make it 4-chromatic.
        # But for now, let's see if random greedy finds any.
        
    return best_edges
