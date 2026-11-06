# EVOLVE-BLOCK-START
import time
import random

def _creates_c4(adj, u, v):
    """
    Check if adding edge (u, v) to the graph with adjacency sets 'adj' creates a C4.
    A C4 is u-x-y-v-u.
    This exists if there is an edge (x, y) such that x in adj[u] and y in adj[v].
    """
    nu = adj[u]
    nv = adj[v]
    # Optimize: iterate over the smaller set
    if len(nu) > len(nv):
        nu, nv = nv, nu
    for x in nu:
        if adj[x] & nv:
            return True
    return False

def _construct_c4_free(n, min_deg_required, time_limit):
    """
    Construct a C4-free graph on n vertices with min degree >= min_deg_required.
    Uses a greedy approach with local repair (edge removal) to escape dead ends.
    """
    start_time = time.time()
    
    # Try multiple random instances
    max_attempts = 200 # Increased attempts
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        adj = [set() for _ in range(n)]
        # Track edges for removal
        edges = set()
        
        # We will try to build the graph
        # Strategy: repeatedly pick a vertex with min degree, try to connect it to a valid neighbor.
        # If no valid neighbor, remove a random edge to "open up" the graph.
        
        # To avoid infinite loops in repair, we limit repair steps per attempt
        max_repair_steps = 100 # Increased repair steps
        
        for step in range(n * n * 3): # Max steps safeguard
            if time.time() - start_time > time_limit:
                return None
                
            # Check success
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_required:
                # Success
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append([u, v])
                return {'n': n, 'edges': edge_list}
                
            # Find vertex with minimum degree
            min_deg_vertex = min(range(n), key=lambda i: len(adj[i]))
            
            # Try to find a valid neighbor for min_deg_vertex
            found = False
            # Randomize neighbor order to avoid bias
            candidates = [v for v in range(n) if v != min_deg_vertex and v not in adj[min_deg_vertex]]
            random.shuffle(candidates)
            
            for v in candidates:
                if not _creates_c4(adj, min_deg_vertex, v):
                    # Add edge
                    adj[min_deg_vertex].add(v)
                    adj[v].add(min_deg_vertex)
                    edges.add((min(min_deg_vertex, v), max(min_deg_vertex, v)))
                    found = True
                    break
                    
            if not found:
                # Stuck. Try to repair by removing a random edge.
                if max_repair_steps <= 0:
                    break # Restart attempt
                max_repair_steps -= 1
                
                if not edges:
                    break # Can't repair empty graph
                    
                # Remove an edge. 
                # Heuristic: Remove an edge incident to a high degree vertex?
                # Or an edge that is part of many potential C4s?
                # Simple: Random edge.
                # Better: Edge with max degree sum?
                edge_to_remove = random.choice(list(edges))
                u, v = edge_to_remove
                adj[u].remove(v)
                adj[v].remove(u)
                edges.remove(edge_to_remove)
                
    return None

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin before 120s limit
    
    # Try from largest n down
    # n=47 is impossible (proven above).
    # n=46 requires min degree 7.
    # n=45 requires min degree 6.
    
    time_budgets = {
        46: 60.0,
        45: 30.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(46, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Cap the time for this n
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
            
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
            
    # Fallback
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
