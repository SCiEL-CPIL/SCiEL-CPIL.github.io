import time
import random

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # We know n=46 is possible. Try 47 first, then 46, etc.
    # The task says value is known to be 46 or 47.
    # If we find 47, score is higher.
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n
        result = _fast_greedy_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _is_safe(adj, u, v):
    """
    Check if adding edge (u,v) to graph defined by adjacency sets adj creates a C4.
    adj is a list of sets.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We need to check if there is an edge (x,y) in E
    # such that x in N(u), y in N(v), and x != y.
    # This creates cycle u-x-y-v-u.
    
    for x in nu:
        # Check if any neighbor of x is in nv
        # If x is in nv, we must exclude x from the check for y
        # because y != x
        common = adj[x] & nv
        if common:
            # If x is in common, it means x is a neighbor of itself? No, x in nv and x in adj[x] is false.
            # Wait, adj[x] contains neighbors of x. x is not in adj[x].
            # So if x in nv, x is not in adj[x].
            # So common will not contain x.
            # Therefore, if common is not empty, there is some y in nv such that (x,y) in E.
            # And since x not in adj[x], y != x is automatically satisfied?
            # No, y is in nv. x is in nu.
            # If x is in nv, then x is a common neighbor.
            # But x is not in adj[x]. So x is not in common.
            # So any y in common is distinct from x.
            return False
            
    return True

def _fast_greedy_construct(n, required_min_deg, deadline):
    """
    Attempts to build a C4-free graph with min degree >= required_min_deg.
    Uses multiple random restarts with a greedy strategy.
    """
    max_restarts = 1000  # Set-based check is fast, so we can do many restarts
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random seed for this attempt
        random.seed(attempt)
        
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        
        success = False
        
        # We need to add edges until min(degrees) >= required_min_deg
        # Total edges needed approx n * required_min_deg / 2
        
        while time.time() < deadline:
            min_deg = min(degrees)
            if min_deg >= required_min_deg:
                success = True
                break
                
            # Find vertices with min degree
            min_vertices = [i for i in range(n) if degrees[i] == min_deg]
            
            # Pick one randomly
            u = random.choice(min_vertices)
            
            # Get non-neighbors of u
            non_neighbors = [v for v in range(n) if v not in adj[u] and v != u]
            
            if not non_neighbors:
                # u is connected to everyone, but degree is low? Impossible if n is large enough
                # Or we are stuck
                break
                
            # Shuffle to try random order
            random.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if _is_safe(adj, u, v):
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added = True
                    break  # Restart loop to re-evaluate degrees
            
            if not added:
                # Could not add any edge to u without creating C4
                # Try a different strategy or restart
                # For now, break and restart
                break
                
        if success:
            edges = []
            for u in range(n):
                for v in sorted(adj[u]):
                    if u < v:
                        edges.append([u, v])
            return (n, edges)
            
    return None
