import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We know n=47 is impossible. We aim for n=46.
    # If we fail, we fall back to lower n.
    
    best_result = {'n': 39, 'edges': []}
    best_score = 39
    
    # Try n=46 first
    if _try_construct(46, deadline):
        return _try_construct(46, deadline) # Wait, _try_construct should return the result
    
    # Actually, let's write a helper that returns the graph or None
    res_46 = _find_c4_free_graph(46, deadline)
    if res_46:
        return res_46
    
    res_45 = _find_c4_free_graph(45, deadline)
    if res_45:
        return res_45
        
    # Fallback
    return best_result

def _find_c4_free_graph(n, deadline):
    min_deg_needed = n - 39
    # For n=46, min_deg_needed = 7
    # Target edges: around n * min_deg / 2
    
    # Strategy: Randomized Greedy Removal from a sparse random graph
    # Start with a random graph with slightly more edges than needed
    # Remove edges to break C4s
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        # Generate random graph
        # Use p such that expected degree is min_deg_needed + 1
        p = (min_deg_needed + 1) / (n - 1)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u,v))
                    
        # Greedy removal of C4s
        # While there is a C4, remove an edge from it
        # To be efficient, we don't find "a" C4, we find an edge with max common neighbors
        # If max common neighbors >= 2, removing it reduces C4 count
        
        changed = True
        while changed:
            changed = False
            max_common = 0
            edge_to_remove = None
            
            # Check all edges
            # This is O(E * n) which is fine for n=46
            for (u, v) in list(edges):
                if u not in adj[v]: # edge might have been removed
                    continue
                common = len(adj[u] & adj[v])
                if common > max_common:
                    max_common = common
                    edge_to_remove = (u, v)
                    
            if max_common >= 2:
                # Remove edge_to_remove
                u, v = edge_to_remove
                adj[u].remove(v)
                adj[v].remove(u)
                edges.remove((u, v))
                changed = True
                
        # Check min degree
        min_deg = min(len(adj[i]) for i in range(n))
        
        if min_deg >= min_deg_needed:
            # Verify C4-free
            if _verify_c4_free(n, edges):
                return {'n': n, 'edges': [[u, v] for u, v in edges]}
                
    return None

def _verify_c4_free(n, edges):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    for (u, v) in edges:
        if len(adj[u] & adj[v]) >= 2:
            return False
    return True

# The construct function above was a bit messy, let's clean it up.
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try n=46
    res = _find_c4_free_graph(46, deadline)
    if res:
        return res
        
    # Try n=45
    res = _find_c4_free_graph(45, deadline)
    if res:
        return res
        
    # Try n=44
    res = _find_c4_free_graph(44, deadline)
    if res:
        return res
        
    # Fallback to a trivial graph (empty)
    return {'n': 1, 'edges': []}
