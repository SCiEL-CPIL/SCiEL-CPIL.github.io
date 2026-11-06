"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_c4_free(n, adj):
    """Check if graph with adjacency list adj on n vertices has no C4."""
    # adj is a list of sets or lists. Let's assume sets for O(1) lookup.
    # C4 exists if any two vertices have > 1 common neighbor.
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) > 1:
                return False
    return True

def _min_degree(n, adj):
    return min(len(adj[i]) for i in range(n))

def _try_construct(n, time_limit=10):
    """Try to construct a C4-free graph on n vertices with min degree >= n-39."""
    if n <= 39:
        return {'n': n, 'edges': []}
    
    min_deg_required = n - 39
    if min_deg_required < 1:
        min_deg_required = 1
        
    start_time = time.time()
    
    # We will try random graphs and repair C4s
    for attempt in range(20000):
        if time.time() - start_time > time_limit:
            break
        
        # Choose p such that expected degree is slightly above min_deg_required
        # p * (n-1) ~ min_deg_required + buffer
        p = (min_deg_required + 2) / (n - 1)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate random graph
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
        
        # Check min degree first (cheap)
        if _min_degree(n, adj) < min_deg_required:
            continue
            
        # Check C4
        if _is_c4_free(n, adj):
            # Success
            edge_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_list.append([u, v])
            return {'n': n, 'edges': edge_list}
            
        # If C4 exists, try to repair by removing edges from C4s
        # This is a heuristic repair
        repaired = False
        for _ in range(50): # max repair steps
             if _min_degree(n, adj) < min_deg_required:
                 break
             # Find a C4
             c4_found = False
             for u in range(n):
                 if c4_found: break
                 for v in range(u + 1, n):
                     common = adj[u] & adj[v]
                     if len(common) > 1:
                         # Found C4 involving u, v and two common neighbors
                         # Remove one edge to break it.
                         # Pick a random edge from the C4.
                         # The C4 is u - w1 - v - w2 - u
                         # Edges: (u,w1), (w1,v), (v,w2), (w2,u)
                         # Let's remove (u, w1) or similar.
                         # Actually, just remove a random edge incident to u or v that is part of the C4.
                         # Simpler: remove a random edge from the whole graph? No, targeted.
                         # Let's remove edge (u, w1) where w1 is a common neighbor.
                         w1 = random.choice(list(common))
                         # Remove edge (u, w1)
                         if w1 in adj[u]:
                             adj[u].remove(w1)
                             adj[w1].remove(u)
                             # Update edges list? We can rebuild at the end.
                             # But we need to check min degree again.
                             c4_found = True
                             break
                 if c4_found: break
             if not c4_found:
                 # No C4 found? It should be C4 free now.
                 if _is_c4_free(n, adj) and _min_degree(n, adj) >= min_deg_required:
                     edge_list = []
                     for u in range(n):
                         for v in adj[u]:
                             if u < v:
                                 edge_list.append([u, v])
                     return {'n': n, 'edges': edge_list}
                 break
                 
    return None

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    # Try from largest n down
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
        # Allocate more time to larger n? Or equal?
        # n=46 is the target.
        result = _try_construct(n, time_limit=remaining_time)
        if result is not None:
            return result
    
    # Fallback: Construct a known small C4-free graph
    # For n=39, min degree 0 is fine.
    # But we can do better.
    # Let's just return a valid small graph if we fail.
    # The task says "Maximize n". If we fail 46, we should try to return the best we found.
    # But _try_construct returns None on failure.
    # We should keep track of the best result.
    
    # Let's re-structure to keep best result.
    
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
