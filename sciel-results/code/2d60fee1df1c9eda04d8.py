import random
import time
from collections import deque

def construct(k, g):
    if k != 5 or g != 9:
        # Fallback for other cases: use a simple heuristic or return None
        # For this task, we focus on 5,9
        return _fallback_construct(k, g)
        
    N = 1116
    edges, success = _solve_sa(k, g, N, time_limit=110)
    if success:
        return N, edges
        
    # If 1116 fails, try 1200
    edges, success = _solve_sa(k, g, 1200, time_limit=100)
    if success:
        return 1200, edges
        
    # If all fails, return a valid graph from a different simple construction?
    # We can't reuse LUW.
    # We might have to return None, which is bad.
    # But 1116 should be findable.
    return None, None

def _random_regular_graph(N, k):
    # Configuration model
    while True:
        stubs = [i for i in range(N) for _ in range(k)]
        random.shuffle(stubs)
        adj = [set() for _ in range(N)]
        edges = []
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        if valid:
            return adj, edges

def _count_short_cycles(adj, N, g):
    # Count cycles of length < g
    # g=9, so lengths 3..8
    count = 0
    for start in range(N):
        # BFS
        # dist[v] = distance
        # parent[v] = parent in BFS tree
        # To detect cycles, we check edges (u,v) where dist[u] and dist[v] are close
        # Actually, simpler: 
        # For each vertex, do BFS up to depth (g-1)//2
        # If we see a vertex at distance d, and it has a neighbor at distance d-1 that is not its parent, 
        # and the path length forms a cycle < g.
        
        # More robust: 
        # Find all simple cycles of length < g? Too slow.
        
        # Use the property: A cycle of length L < g exists iff 
        # there is a vertex v and two distinct neighbors u1, u2 such that 
        # the shortest path between u1 and u2 not passing through v is < L-2?
        
        # Let's just do BFS and check for "chords" or back edges.
        # For undirected graph, BFS tree.
        # Any edge (u,v) where dist[u] == dist[v] or |dist[u]-dist[v]| == 1 (and not parent) 
        # forms a cycle.
        # Cycle length = dist[u] + dist[v] + 1 (if dist[u]==dist[v]) 
        # or dist[u] + dist[v] + 1 (if dist differs by 1)?
        # If dist[u] == dist[v] = d, cycle len = 2d + 1.
        # If dist[u] = d, dist[v] = d+1, and v is not child of u, cycle len = d + (d+1) + 1 = 2d + 2.
        
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # If v is already visited
                    # Check if this edge creates a short cycle
                    if v == parent[u]:
                        continue
                    # Cycle length
                    # If visited[u] == visited[v]:
                    #    len = 2 * visited[u] + 1
                    # Else if abs(visited[u] - visited[v]) == 1:
                    #    len = visited[u] + visited[v] + 1
                    # Else:
                    #    The cycle is longer or not detected by simple BFS edge check?
                    #    Actually, BFS detects the shortest cycle through the tree edge?
                    #    No, BFS detects the cycle formed by the tree path and the edge.
                    
                    d1 = visited[u]
                    d2 = visited[v]
                    if abs(d1 - d2) <= 1:
                        cycle_len = d1 + d2 + 1
                        if cycle_len < g:
                            count += 1
                            # We can break early if we just want "exists", but we want count for SA
                            # But counting all is expensive.
                            # Let's just return 1 if found? 
                            # No, SA needs a cost function.
                            # Maybe just count the number of such edges?
                            pass
    return count

def _solve_sa(k, g, N, time_limit):
    adj, edges = _random_regular_graph(N, k)
    if adj is None:
        return None, False
        
    badness = _count_short_cycles(adj, N, g)
    if badness == 0:
        return edges, True
        
    T = 1.0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick two random edges
        e1 = random.choice(edges)
        e2 = random.choice(edges)
        if e1 == e2:
            continue
            
        u, v = e1
        x, y = e2
        
        # Check non-adjacency
        if u == x or u == y or v == x or v == y:
            continue
            
        # Try swap 1: (u,x) and (v,y)
        # Check if edges exist
        if x not in adj[u] and y not in adj[v]:
            # Perform swap
            adj[u].remove(v); adj[v].remove(u)
            adj[x].remove(y); adj[y].remove(x)
            
            adj[u].add(x); adj[x].add(u)
            adj[v].add(y); adj[y].add(v)
            
            edges.remove(e1); edges.remove(e2)
            edges.append((u, x)); edges.append((v, y))
            
            # Recalculate badness
            new_badness = _count_short_cycles(adj, N, g)
            
            delta = new_badness - badness
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                badness = new_badness
                if badness == 0:
                    return edges, True
            else:
                # Revert
                adj[u].remove(x); adj[x].remove(u)
                adj[v].remove(y); adj[y].remove(v)
                
                adj[u].add(v); adj[v].add(u)
                adj[x].add(y); adj[y].add(x)
                
                edges.remove((u, x)); edges.remove((v, y))
                edges.append(e1); edges.append(e2)
                
        # Try swap 2: (u,y) and (v,x)
        elif y not in adj[u] and x not in adj[v]:
            # Perform swap
            adj[u].remove(v); adj[v].remove(u)
            adj[x].remove(y); adj[y].remove(x)
            
            adj[u].add(y); adj[y].add(u)
            adj[v].add(x); adj[x].add(v)
            
            edges.remove(e1); edges.remove(e2)
            edges.append((u, y)); edges.append((v, x))
            
            new_badness = _count_short_cycles(adj, N, g)
            
            delta = new_badness - badness
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                badness = new_badness
                if badness == 0:
                    return edges, True
            else:
                # Revert
                adj[u].remove(y); adj[y].remove(u)
                adj[v].remove(x); adj[x].remove(v)
                
                adj[u].add(v); adj[v].add(u)
                adj[x].add(y); adj[y].add(x)
                
                edges.remove((u, y)); edges.remove((v, x))
                edges.append(e1); edges.append(e2)

        T *= 0.999
        
    return None, False
