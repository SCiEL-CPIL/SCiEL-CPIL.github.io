import time
import random
import math
from collections import deque

def construct(k, g):
    if k != 4:
        # Fallback for other k: use LUW
        return luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Target girth is g (10)
    # Try to find a graph with n=320
    n_target = 320
    
    # Try a few restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        edges = random_4_regular(n_target)
        if not edges:
            continue
            
        adj = [[] for _ in range(n_target)]
        adj_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            if u > v:
                u, v = v, u
            adj_set.add((u, v))
            
        # Local search to minimize short cycles
        success = local_search_min_cycles(adj, adj_set, n_target, g, deadline)
        if success:
            # Verify girth
            if get_girth(adj, g) >= g:
                return n_target, list(adj_set)
                
    # Fallback to LUW
    return luw_construct(4, g)

def luw_construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        
    n = len(order)
    return n, edges

def random_4_regular(n):
    # Generate a random 4-regular graph using perfect matching composition
    # 4-regular = union of two 2-factors (perfect matchings)
    # Or just random pairing of stubs.
    
    # Stubs: each vertex has 4 stubs
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    # Pair them up
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u != v:
            if u > v:
                u, v = v, u
            edges.add((u, v))
            
    # Check if it's simple and 4-regular
    # If self-loop or multi-edge, reject and retry?
    # For large n, probability of multi-edge is low.
    # But we need to ensure no loops.
    
    # Let's use a more robust method: 
    # 1. Create list of vertices with 4 stubs each.
    # 2. Shuffle stubs.
    # 3. Pair consecutive stubs.
    # 4. If any pair is (u,u) or duplicate edge, discard and retry.
    
    for _ in range(100):
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
            
        if valid:
            return list(edges)
            
    # Fallback: return empty or raise error? 
    # Should not happen for n >= 10.
    return []

def local_search_min_cycles(adj, adj_set, n, g, deadline):
    # We want to minimize cycles of length < g.
    # Since g=10, we care about 4, 6, 8.
    # Let's focus on 4-cycles first.
    
    def count_4_cycles(adj):
        count = 0
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i+1, len(adj[u])):
                    v1 = adj[u][i]
                    v2 = adj[u][j]
                    # Count common neighbors of v1 and v2
                    common = 0
                    for w in adj[v1]:
                        if w in adj[v2] and w != u:
                            common += 1
                    count += common
        return count
        
    current_4_cycles = count_4_cycles(adj)
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        # Pick two random edges
        e1 = random.choice(list(adj_set))
        e2 = random.choice(list(adj_set))
        if e1 == e2:
            continue
            
        u, v = e1
        x, y = e2
        
        # Try swap (u,v) and (x,y) -> (u,x) and (v,y)
        if u == x or v == y:
            continue
            
        ux = (min(u,x), max(u,x))
        vy = (min(v,y), max(v,y))
        
        if ux in adj_set or vy in adj_set:
            continue
            
        # Perform swap
        adj_set.remove(e1)
        adj_set.remove(e2)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj_set.add(ux)
        adj_set.add(vy)
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
        new_4_cycles = count_4_cycles(adj)
        
        # Accept if 4-cycles decrease
        if new_4_cycles < current_4_cycles:
            current_4_cycles = new_4_cycles
        else:
            # Revert
            adj_set.remove(ux)
            adj_set.remove(vy)
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            
            adj_set.add(e1)
            adj_set.add(e2)
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
            # If we have no 4-cycles, check for 6-cycles
            if current_4_cycles == 0:
                # Check girth
                if get_girth(adj, g) >= g:
                    return True
                # If girth < g, try to break the short cycle
                # For now, just continue searching
        
    # Final check
    if get_girth(adj, g) >= g:
        return True
    return False

def get_girth(adj, g_limit):
    # Returns girth if < g_limit, else g_limit
    # BFS from each vertex
    n = len(adj)
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    # But this is not necessarily the shortest cycle through u and v
                    # However, if dist[u] + dist[v] + 1 < g_limit, we found a short cycle.
                    # Note: This detects *some* cycle, not necessarily the girth.
                    # But if *any* cycle is < g_limit, we return < g_limit.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return cycle_len
                        
    return g_limit

def construct(k, g):
    if k != 4:
        return luw_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We will try to find a valid graph for n=320
    # Using a more efficient local search
    
    n_target = 320
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        edges = random_4_regular(n_target)
        if not edges:
            continue
            
        adj = [[] for _ in range(n_target)]
        adj_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            if u > v:
                u, v = v, u
            adj_set.add((u, v))
            
        current_girth = get_girth(adj, g)
        
        if current_girth >= g:
            return n_target, edges
            
        # Local search
        # We will perform 1000 swaps
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Pick two random edges
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
                
            u, v = e1
            x, y = e2
            
            # Try swap: (u,x) and (v,y)
            # Check if u==x or v==y (loop)
            if u == x or v == y:
                continue
                
            # Check if (u,x) or (v,y) in adj_set
            ux = (min(u,x), max(u,x))
            vy = (min(v,y), max(v,y))
            
            if ux in adj_set or vy in adj_set:
                continue
                
            # Perform swap
            # Remove (u,v) and (x,y)
            adj_set.remove(e1)
            adj_set.remove(e2)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add (u,x) and (v,y)
            adj_set.add(ux)
            adj_set.add(vy)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Update edges list
            edges.remove(e1)
            edges.remove(e2)
            edges.append(ux)
            edges.append(vy)
            
            # Check girth
            new_girth = get_girth(adj, g)
            
            if new_girth >= g:
                return n_target, edges
            elif new_girth > current_girth:
                current_girth = new_girth
            else:
                # Revert
                adj_set.remove(ux)
                adj_set.remove(vy)
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj_set.add(e1)
                adj_set.add(e2)
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                
                edges.remove(ux)
                edges.remove(vy)
                edges.append(e1)
                edges.append(e2)

    # If we didn't find one for 320, try 330
    # ...
    # For now, fall back to LUW if we run out of time or fail
    return luw_construct(4, g)
