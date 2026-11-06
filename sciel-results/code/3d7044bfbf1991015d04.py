import random
import time

def construct():
    n = 31
    target_edges = 90
    deadline = time.time() + 110  # Safety margin
    
    # Function to check if adding (u, v) creates a C4
    # It creates a C4 if u and v have >= 2 common neighbors
    def can_add(adj, u, v):
        if v in adj[u]:
            return False
        # Count common neighbors
        common = 0
        for w in adj[u]:
            if w in adj[v]:
                common += 1
                if common >= 2:
                    return False
        return True

    # Greedy construction with randomization
    best_edges = None
    
    while time.time() < deadline:
        adj = [set() for _ in range(n)]
        edges = []
        
        # Shuffle vertices to randomize the order of edge addition
        # We consider pairs (i, j) with i < j
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if len(edges) >= target_edges:
                break
            if can_add(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
        
        if len(edges) == target_edges:
            # Found a C4-free graph with 90 edges
            # Add one more edge to get 91
            # Any non-edge will do, as it will create 0 C4s
            # Find a non-edge
            for u, v in pairs:
                if v not in adj[u]:
                    edges.append((u, v))
                    break
            best_edges = edges
            break
        else:
            # Failed to reach 90 edges, retry
            continue
            
    if best_edges is None:
        # Fallback: return a random set of 91 edges (likely high C4 count)
        # But we should try harder
        # For now, return the last attempt or a simple structure
        # Let's just return the first 91 pairs as in the seed, to be safe
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
        
    # Convert to list of lists
    return [list(e) for e in best_edges]
