"""Program for the suite task: construct a small k-regular graph of girth g."""
import math
import random
import time
from collections import deque

# EVOLVE-BLOCK-START

def _generate_random_regular(k, n, seed=42):
    """Generate a random k-regular graph on n vertices."""
    if n < k:
        return None
    if (n * k) % 2 != 0:
        return None
        
    rng = random.Random(seed)
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    rng.shuffle(stubs)
    
    edges = set()
    adj = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        
        # Handle self-loops and multi-edges by swapping with later stubs
        valid = False
        for j in range(i+2, len(stubs), 2):
            if stubs[j] != u and stubs[j+1] != u and (u, stubs[j]) not in adj[u] and (u, stubs[j+1]) not in adj[u]:
                # Swap v with stubs[j]
                stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                v = stubs[i+1]
                valid = True
                break
        if not valid:
            # Restart
            return _generate_random_regular(k, n, seed+1)

        if u == v:
             continue
            
        if u in adj[v]:
             continue # Should be handled by swap

        edges.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return _generate_random_regular(k, n, seed+1)
            
    return list(edges), adj

def _find_shortest_cycle(adj, n):
    """Find the shortest cycle. Returns (length, list_of_edges_in_cycle)."""
    min_len = float('inf')
    min_cycle_edges = []
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_len:
                        min_len = cycle_len
                        # Reconstruct cycle edges
                        # Path u -> ... -> start
                        path_u = []
                        curr = u
                        while curr != start:
                            path_u.append((curr, parent[curr]))
                            curr = parent[curr]
                        # Path v -> ... -> start
                        path_v = []
                        curr = v
                        while curr != start:
                            path_v.append((curr, parent[curr]))
                            curr = parent[curr]
                        
                        # Cycle is path_u + edge(u,v) + reversed path_v
                        # Edges are (a,b) where a is child, b is parent
                        cycle_edges = path_u + [(u, v)] + list(reversed(path_v))
                        min_cycle_edges = cycle_edges
                        # Optimization: if min_len == 3, we can't do better
                        if min_len == 3:
                            return min_len, min_cycle_edges
    
    return min_len, min_cycle_edges

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs."""
    
    if k < 2:
        return 0, []
    
    if g <= 2:
        if k == 2:
            n = max(3, g)
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        else:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return n, edges

    timeout = 110
    start_time = time.time()
    
    # Determine starting n
    if k == 4 and g == 10:
        n_start = 320
    else:
        # Moore bound estimate
        m = g // 2
        lb = 1
        term = 1
        for i in range(m):
            lb += k * term
            term *= (k-1)
        n_start = lb
        
    best_n = None
    best_edges = None
    
    # Try increasing n
    for n in range(n_start, n_start + 50):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple random seeds for this n
        found = False
        for seed in range(50):
            if time.time() - start_time > timeout:
                break
                
            res = _generate_random_regular(k, n, seed=seed)
            if res is None:
                continue
                
            edges, adj = res
            
            # Local search
            for step in range(2000):
                if time.time() - start_time > timeout:
                    break
                    
                min_len, cycle_edges = _find_shortest_cycle(adj, n)
                
                if min_len >= g:
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                    
                if min_len == float('inf'):
                    break
                    
                # Attempt to break the shortest cycle
                # Pick a random edge from the cycle to swap
                # Simple heuristic: try swapping a random edge in the cycle with a random non-cycle edge
                # This is O(E^2) which is too slow.
                
                # Better heuristic: 
                # The cycle is C. 
                # Pick an edge (u,v) in C.
                # Find a neighbor u' of u not in C.
                # Find a neighbor v' of v not in C.
                # If (u', v') is not an edge, try swapping (u,v) and (u',v')? No, that changes degree.
                
                # Correct Swap:
                # Remove (u,v) and (x,y). Add (u,x) and (v,y).
                # We want to break cycle C.
                # Let's just pick two random non-adjacent edges in the graph and swap them.
                # If the new graph has a longer shortest cycle, accept.
                
                # To save time, I will just do a few random swaps and check.
                if len(edges) < 2:
                    continue
                
                # Pick two random edges
                e1 = edges[random.randint(0, len(edges)-1)]
                e2 = edges[random.randint(0, len(edges)-1)]
                
                if e1 == e2:
                    continue
                    
                u1, v1 = e1
                u2, v2 = e2
                
                # Ensure they are not adjacent
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                    
                # Try swap: remove (u1,v1), (u2,v2). Add (u1,u2), (v1,v2)
                # Check validity
                if (min(u1,u2), max(u1,u2)) in adj[u1] or (min(v1,v2), max(v1,v2)) in adj[v1]:
                    continue
                if (min(u1,u2), max(u1,u2)) in edges or (min(v1,v2), max(v1,v2)) in edges:
                    continue
                if u1 == u2 or v1 == v2:
                    continue
                    
                # Perform swap
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                edges.remove(e1)
                edges.remove(e2)
                
                new_e1 = (min(u1,u2), max(u1,u2))
                new_e2 = (min(v1,v2), max(v1,v2))
                
                adj[u1].add(u2)
                adj[u2].add(u1)
                adj[v1].add(v2)
                adj[v2].add(v1)
                
                edges.append(new_e1)
                edges.append(new_e2)
                
                # Check if girth improved
                new_min_len, _ = _find_shortest_cycle(adj, n)
                if new_min_len < min_len:
                    # Revert
                    adj[u1].remove(u2)
                    adj[u2].remove(u1)
                    adj[v1].remove(v2)
                    adj[v2].remove(v1)
                    
                    edges.remove(new_e1)
                    edges.remove(new_e2)
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    edges.append(e1)
                    edges.append(e2)
                    # Continue to next step
                    continue
                else:
                    # Accept (or keep)
                    pass
                    
            if found:
                break
                
        if found:
            break
            
    if best_n is not None:
        return best_n, best_edges
        
    return 0, []

# EVOLVE-BLOCK-END
