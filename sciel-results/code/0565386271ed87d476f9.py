import time
import random
import collections
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # k is expected to be 3, g to be 16
    # Search range for N
    # Lower bound 512, Upper bound 936
    # Step 2 because 3-regular graphs must have even number of vertices
    
    for N in range(512, 937, 2):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this N
        # Number of attempts per N
        # We can scale attempts based on remaining time
        # But simple loop is fine
        
        while time.time() < deadline - 5.0:
            # Generate random 3-regular graph
            edges = _generate_random_3_reg(N)
            if edges is None:
                continue
                
            # Check girth
            girth = _check_girth(edges, N)
            
            if girth >= g:
                return N, edges
            
            # If girth is close to g, we might keep trying
            # But if girth is very small, we just continue
            
    # If we reach here, we failed to find a graph within time/range
    # This is a failure case
    # As a last resort, return a known small graph? 
    # But it must have girth >= g.
    # If we can't find one, we raise exception.
    raise Exception("Failed to construct graph with required girth")

def _generate_random_3_reg(N):
    # Configuration model
    half_edges = []
    for i in range(N):
        half_edges.extend([i, i, i])
        
    random.shuffle(half_edges)
    
    edges_set = set()
    for i in range(0, 3*N, 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None
        if (u, v) in edges_set or (v, u) in edges_set:
            return None
        edges_set.add((u, v))
        
    return list(edges_set)

def _check_girth(edges, N):
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    # For each edge, remove it and find shortest path between its endpoints
    for u, v in edges:
        # BFS from u to v, avoiding edge (u,v)
        dist = [-1] * N
        dist[u] = 0
        q = collections.deque([u])
        found = False
        
        while q:
            curr = q.popleft()
            for nei in adj[curr]:
                # Skip the edge (u,v) in both directions
                if (curr == u and nei == v) or (curr == v and nei == u):
                    continue
                if dist[nei] == -1:
                    dist[nei] = dist[curr] + 1
                    if nei == v:
                        cycle_len = dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < 16:
                                return min_girth
                        found = True
                        break
                    q.append(nei)
            if found:
                break
                
    return min_girth
