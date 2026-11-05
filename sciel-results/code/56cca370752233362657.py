import random
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Helper to check 3-colorability
    def is_3_colorable(adj):
        # adj is a list of bitmasks
        # Order vertices by degree
        vertices = list(range(n))
        vertices.sort(key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        colors = [0] * n # 0: uncolored, 1-3: colors
        
        def backtrack(idx):
            if idx == n:
                return True
            v = vertices[idx]
            # Available colors
            avail = 7
            for u in range(n):
                if (adj[v] >> u) & 1:
                    avail &= ~(1 << colors[u])
            if avail == 0:
                return False
            
            # Try each available color
            # Heuristic: try color with fewest options? Or just iterate.
            for c in range(3):
                if (avail >> c) & 1:
                    colors[v] = c + 1
                    if backtrack(idx + 1):
                        return True
                    colors[v] = 0
            return False
        
        return backtrack(0)
    
    # Random search
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        edge_list = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a triangle
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            if common == 0:
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_list.append([u, v])
        
        # Count edges
        count = len(edge_list)
        
        # Only check colorability if we might improve
        if count > best_count:
            if not is_3_colorable(adj):
                best_count = count
                best_edges = [e[:] for e in edge_list]
                
                # If we reached a very high number, we might stop or continue
                # Max possible is 64. 3-colorable max is lower.
                # Let's keep searching for a bit more to see if we can go higher.
    
    if best_edges is None:
        # Fallback to Grötzsch if no better found (should not happen)
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    return best_edges
