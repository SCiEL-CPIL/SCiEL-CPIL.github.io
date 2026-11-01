import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    m = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(m))
    
    max_n = moore_bound + 100
    
    best_result = None
    
    for n in range(moore_bound, max_n + 1):
        if time.time() > deadline - 10.0:
            break
        
        result = local_search(k, g, n, 2.0)
        if result is not None:
            best_result = result
            break
            
    if best_result is not None:
        return best_result
        
    return construct_luw(k, g)

def local_search(k, g, n, time_limit):
    start_time = time.time()
    
    adj = generate_random_k_regular(n, k)
    if adj is None:
        return None
        
    # We need to track edges to ensure simplicity during swaps.
    # adj is list of sets.
    
    while time.time() - start_time < time_limit:
        # Check if valid
        if verify_girth_fast(adj, g, n):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
        # Find a short cycle to break
        cycle = find_short_cycle(adj, g, n)
        if cycle is not None:
            # Try to break it
            if not break_cycle(adj, n, k, cycle):
                # Random swap
                random_swap(adj, n, k)
        else:
            # No cycle found by fast check, but verify_girth_fast failed?
            # This implies a cycle exists but wasn't found by the sample.
            # Just do a random swap.
            random_swap(adj, n, k)
            
    return None

def generate_random_k_regular(n, k):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    
    # Pair stubs
    # We process stubs in pairs.
    # To avoid loops and multi-edges, we use a simple retry logic.
    
    # This is O(n^2) worst case, but for n=320, it's fine.
    
    i = 0
    while i < len(stubs):
        u = stubs[i]
        v = stubs[i+1]
        
        if u == v:
            # Find another stub for u
            j = i + 2
            while j < len(stubs):
                if stubs[j] != u:
                    break
                j += 1
            if j == len(stubs):
                return None # Failed
            # Swap stubs[i+1] and stubs[j]
            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
            continue
            
        if v in adj[u]:
            # Multi-edge
            # Find another stub for u
            j = i + 2
            while j < len(stubs):
                if stubs[j] != u and stubs[j] not in adj[u]:
                    break
                j += 1
            if j == len(stubs):
                return None
            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
            continue
            
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        i += 2
        
    return adj

def verify_girth_fast(adj, g, n):
    # Check from 10 random vertices
    for _ in range(10):
        s = random.randint(0, n-1)
        if not check_girth_from(adj, s, g, n):
            return False
    return True

def check_girth_from(adj, s, g, n):
    visited = [-1] * n
    parent = [-1] * n
    visited[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        if visited[u] >= g - 1:
            continue
        for v in adj[u]:
            if visited[v] == -1:
                visited[v] = visited[u] + 1
                parent[v] = u
                q.append(v)
            else:
                if u != v and parent[v] != u and parent[u] != v:
                    # Cycle found
                    # Length is at least 3
                    # We need to check if it is < g
                    # The BFS tree gives the shortest path from s to v and s to u.
                    # The cycle length is dist[u] + dist[v] + 1.
                    # But this is only valid if the path from s to u and s to v are disjoint (except s).
                    # In BFS, they are.
                    # So if dist[u] + dist[v] + 1 < g, we have a short cycle.
                    if visited[u] + visited[v] + 1 < g:
                        return False
    return True

def find_short_cycle(adj, g, n):
    # Return a list of nodes in a short cycle
    # Use BFS from a few vertices
    for _ in range(5):
        s = random.randint(0, n-1)
        visited = [-1] * n
        parent = [-1] * n
        visited[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if visited[u] >= g - 1:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if u != v and parent[v] != u and parent[u] != v:
                        if visited[u] + visited[v] + 1 < g:
                            # Reconstruct cycle
                            # Path s -> ... -> u and s -> ... -> v
                            # Cycle is u -> ... -> v -> s -> ... -> u? No.
                            # The cycle is formed by the back edge (u,v) and the tree paths.
                            # We just need one edge to break.
                            return [u, v]
    return None

def break_cycle(adj, n, k, cycle):
    # Pick an edge in the cycle, e.g., (u, v)
    # Try to swap it with a random edge (x, y)
    u, v = cycle
    # Pick random x, y
    for _ in range(10):
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y: continue
        if x in (u, v) or y in (u, v): continue
        if x not in adj[y]: continue
        
        # Try swap (u,v) and (x,y) with (u,x) and (v,y)
        # Check validity
        if u == x or u == y or v == x or v == y:
            continue
        if x in adj[u] or y in adj[v]:
            continue
            
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(y)
        adj[y].add(v)
        
        return True
    return False

def random_swap(adj, n, k):
    # Pick two random edges
    for _ in range(10):
        u = random.randint(0, n-1)
        if not adj[u]: continue
        v = random.choice(list(adj[u]))
        
        x = random.randint(0, n-1)
        if not adj[x]: continue
        y = random.choice(list(adj[x]))
        
        if u == x or u == y or v == x or v == y:
            continue
        if x in adj[u] or y in adj[v]:
            continue
            
        # Swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(y)
        adj[y].add(v)
        
        return True
    return False

def construct_luw(k, g):
    # ... (LUW code)
    pass
