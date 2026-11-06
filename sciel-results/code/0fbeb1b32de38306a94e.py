import time
import random
from copy import deepcopy

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and (w in adj[u]):
                    return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks."""
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    colors = [None] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Determine which colors are forbidden
        forbidden = 0
        for nb in adj[v]:
            c = colors[nb]
            if c is not None:
                forbidden |= (1 << c)
        
        # Try available colors
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = None
        
        return False
    
    return backtrack(0)

def _find_mis(adj, exclude):
    """
    Find a Maximum Independent Set in the graph defined by adj,
    excluding the vertex 'exclude'.
    Uses a simple greedy heuristic with randomization for speed.
    """
    n = len(adj)
    # Create a list of available vertices
    available = [i for i in range(n) if i != exclude]
    
    # Shuffle to get different results on each call (if called in a loop with restarts)
    # But here we want a good MIS. A greedy approach:
    # 1. Sort by degree in the subgraph (lowest first)
    # 2. Pick vertex, add to set, remove it and its neighbors
    
    # To make it robust, we can try a few different orderings
    best_set = set()
    
    # We'll do a few random shuffles and greedy picks
    for _ in range(5):
        current_available = available.copy()
        random.shuffle(current_available)
        
        # Sort by degree in the induced subgraph of current_available
        # This is O(n^2) per step, total O(n^3) which is fine for n=16
        current_set = set()
        remaining = set(current_available)
        
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_v = None
            min_deg = float('inf')
            for v in remaining:
                deg = len([u for u in remaining if u in adj[v]])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            if min_v is None:
                break
                
            current_set.add(min_v)
            remaining.remove(min_v)
            # Remove neighbors of min_v from remaining
            for u in list(remaining):
                if u in adj[min_v]:
                    remaining.remove(u)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin for safety/verification
    
    # Build Grötzsch graph on vertices 0-10
    # C5: 0-1-2-3-4-0
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
            
    for i in range(5):
        add_edge(i, (i + 1) % 5)
        
    # Mycielski layer 5-9
    for i in range(5):
        add_edge(5 + i, (i + 1) % 5)
        add_edge(5 + i, (i - 1) % 5)
        
    # Apex 10
    for i in range(5):
        add_edge(10, 5 + i)
        
    new_vertices = [11, 12, 13, 14, 15]
    best_edge_count = -1
    best_adj = None
    
    # Run multiple restarts
    # Each restart shuffles the order of adding new vertices
    # and uses randomized greedy MIS
    
    max_iterations = 2000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        current_adj = [set(s) for s in adj]
        
        # Shuffle the order in which we process new vertices
        order = new_vertices.copy()
        random.shuffle(order)
        
        for v in order:
            # Find a large independent set in the graph formed by all vertices except v
            # using the current_adj state
            mis = _find_mis(current_adj, v)
            
            # Connect v to all vertices in mis
            for u in mis:
                current_adj[v].add(u)
                current_adj[u].add(v)
                
        # Count edges
        edge_count = sum(len(s) for s in current_adj) // 2
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = current_adj
    
    # Use the best found
    if best_adj is None:
        # Fallback to just the Grötzsch graph (should not happen)
        final_adj = adj
    else:
        final_adj = best_adj
        
    # Final verification (optional but good for safety)
    # if not _is_triangle_free(final_adj):
    #     # Should not happen
    #     pass
    # if _is_3_colorable(final_adj, n):
    #     # Should not happen
    #     pass

    edges = []
    for u in range(n):
        for v in final_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges
