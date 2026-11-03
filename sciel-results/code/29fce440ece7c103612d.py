import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g on as few vertices as possible.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Target n=320 (known cage)
    # First, try to find a circulant graph C_n(s1, s2)
    # This is fast to check.
    
    def check_circulant_girth(n, s1, s2):
        # Build adj
        adj = [[] for _ in range(n)]
        for i in range(n):
            adj[i].append((i + s1) % n)
            adj[i].append((i - s1) % n)
            adj[i].append((i + s2) % n)
            adj[i].append((i - s2) % n)
        
        # Check simplicity
        for i in range(n):
            if len(set(adj[i])) != 4:
                return False
            if i in adj[i]:
                return False

        # BFS for girth
        min_girth = 10**9
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < g:
                                return False
            if min_girth < g:
                return False
        return True

    def get_circulant_edges(n, s1, s2):
        edges = set()
        for i in range(n):
            for j in [(i + s1) % n, (i - s1) % n, (i + s2) % n, (i - s2) % n]:
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
        return list(edges)

    # Search for circulant
    # We know n=320 is the target. Try nearby n.
    # We will try n from 243 to 400 first with random s1, s2.
    # Then expand range if time permits.
    
    found = False
    # Try specific n values first
    candidates = list(range(320, 321)) + list(range(243, 320)) + list(range(321, 400))
    
    for n in candidates:
        if time.time() > deadline - 10:
            break
        # Try random pairs
        attempts = 0
        while time.time() < deadline - 10 and attempts < 500:
            s1 = random.randint(1, n//2)
            s2 = random.randint(1, n//2)
            if s1 == s2 or s1 + s2 == n:
                continue
            if check_circulant_girth(n, s1, s2):
                return n, get_circulant_edges(n, s1, s2)
            attempts += 1
            
    # If no circulant found, try Random Regular Graphs with Local Search
    # for n in [320, 400]
    
    def generate_random_4reg(n):
        # Configuration model
        stumps = []
        for i in range(n):
            for _ in range(4):
                stumps.append(i)
        random.shuffle(stumps)
        adj = [[] for _ in range(n)]
        for i in range(0, len(stumps), 2):
            u, v = stumps[i], stumps[i+1]
            if u == v:
                # Self-loop, retry or fix
                # For simplicity, if self-loop, swap with another stump
                # This is a simplified generator, might need retries
                pass 
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for multi-edges and self-loops
        valid = True
        for i in range(n):
            if len(adj[i]) != 4:
                valid = False
                break
            if len(set(adj[i])) != 4:
                valid = False
                break
        return adj if valid else None

    def local_search_4reg(adj, n, g, time_limit):
        # Try to increase girth by breaking short cycles
        # Find a short cycle
        # If found, pick two edges (u,v) and (x,y) on or near the cycle and swap
        # i.e., remove (u,v), (x,y) and add (u,x), (v,y)
        
        start = time.time()
        while time.time() - start < time_limit:
            # Find shortest cycle
            min_cycle_len = 10**9
            min_cycle_edges = []
            
            # BFS from each node to find shortest cycle
            # This is O(n * (n+E)) which is ok for n=320
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                found_short = False
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            # Cycle found
                            # Reconstruct path
                            # This is expensive to do for every edge.
                            # Optimization: only check if dist[u] + dist[v] + 1 < min_cycle_len
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_cycle_len:
                                min_cycle_len = cycle_len
                                # Store edges of this cycle?
                                # We just need to know a short cycle exists.
                                # To repair, we need the edges.
                                # Let's just record that a short cycle exists and pick random edges to swap?
                                # Better: record the cycle edges.
                                if cycle_len < g:
                                    # Found a short cycle
                                    # We need to break it.
                                    # Pick two edges on the cycle to swap with random other edges?
                                    # Or swap edges within the cycle?
                                    # A 2-switch on the cycle itself might just rotate it.
                                    # We should swap an edge on the cycle with an edge NOT on the cycle.
                                    
                                    # For now, let's just do random 2-switches and re-check.
                                    # It's stochastic.
                                    
                                    # Pick two random edges (u1, v1) and (u2, v2)
                                    # Ensure they don't create multi-edges or loops
                                    # Remove them, add (u1, u2), (v1, v2)
                                    
                                    # We need a list of edges
                                    edge_set = set()
                                    for i in range(n):
                                        for j in adj[i]:
                                            if i < j:
                                                edge_set.add((i, j))
                                    edges = list(edge_set)
                                    
                                    if len(edges) < 2:
                                        break
                                        
                                    e1 = random.choice(edges)
                                    e2 = random.choice(edges)
                                    if e1 == e2:
                                        continue
                                    
                                    u1, v1 = e1
                                    u2, v2 = e2
                                    
                                    # Check validity of new edges
                                    new_e1 = (u1, u2)
                                    new_e2 = (v1, v2)
                                    
                                    if u1 == u2 or v1 == v2:
                                        continue
                                    if new_e1 in edge_set or new_e2 in edge_set:
                                        continue
                                    
                                    # Apply swap
                                    adj[u1].remove(v1)
                                    adj[v1].remove(u1)
                                    adj[u2].remove(v2)
                                    adj[v2].remove(u2)
                                    
                                    adj[u1].append(u2)
                                    adj[u2].append(u1)
                                    adj[v1].append(v2)
                                    adj[v2].append(v1)
                                    
                                    # Re-check girth (expensive)
                                    # Instead of full BFS, we can just continue loop and check at end of batch?
                                    # No, we need to know if we fixed it.
                                    # Let's check girth every few swaps.
                                    
                                    if min_cycle_len >= g:
                                        # Check full girth
                                        if check_girth(adj, n, g):
                                            return True
                                    
            # If we finish BFS and min_cycle_len >= g, we are done
            if min_cycle_len >= g:
                return True
                
        return False

    def check_girth(adj, n, g):
        min_girth = 10**9
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < g:
                                return False
        return min_girth >= g

    # Try random graphs for n in [320, 400]
    for n in range(320, 400, 2):
        if time.time() > deadline - 5:
            break
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            adj = generate_random_4reg(n)
            if adj is None:
                continue
            if check_girth(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            
            # Try local search
            # Limit time per local search
            t_start = time.time()
            if local_search_4reg(adj, n, g, 0.5):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback: LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    # d is the "depth" parameter. Girth is roughly 2d+2.
    # We want girth >= g. So 2d+2 >= g => d >= (g-2)/2.
    d = max(3, (g - 2) // 2)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
            
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
