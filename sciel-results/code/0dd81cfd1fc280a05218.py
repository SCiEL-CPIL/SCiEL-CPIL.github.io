import time
import sys
import random
from collections import deque

def _has_short_cycle(n, adj, g):
    """Check if graph has a cycle of length < g using non-backtracking BFS."""
    for start in range(n):
        dist = {}
        queue = deque()
        for first in adj[start]:
            queue.append((first, start, 1))
            dist[(first, start)] = 1
        
        while queue:
            curr, prev, length = queue.popleft()
            if length >= g - 1:
                continue
            
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                if nxt == start:
                    if 3 <= length + 1 < g:
                        return True
                    continue
                state = (nxt, curr)
                if state not in dist:
                    dist[state] = length + 1
                    queue.append((nxt, curr, length + 1))
    return False

def _get_shortest_cycle(n, adj):
    """Find a shortest cycle in the graph. Returns list of vertices or None."""
    # BFS from each vertex
    best_cycle = None
    for start in range(n):
        dist = {start: -1}
        parent = {}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nxt in adj[curr]:
                if dist.get(nxt, -1) == -1:
                    dist[nxt] = dist[curr] + 1
                    parent[nxt] = curr
                    queue.append(nxt)
                elif nxt != parent.get(curr, -1):
                    # Found a back edge
                    # Reconstruct cycle
                    cycle = [nxt, curr]
                    curr_node = curr
                    while curr_node != nxt:
                        curr_node = parent[curr_node]
                        cycle.append(curr_node)
                    if best_cycle is None or len(cycle) < len(best_cycle):
                        best_cycle = cycle
    return best_cycle

def _random_4_regular(n, seed=42):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    random.seed(seed)
    
    adj = [set() for _ in range(n)]
    for attempt in range(100):
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        success = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                success = False
                break
            if v in adj[u]:
                success = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if success and all(len(adj[i]) == 4 for i in range(n)):
            break
    
    if not all(len(adj[i]) == 4 for i in range(n)):
        return None
    
    # Convert to list of lists for faster access
    adj_list = [list(s) for s in adj]
    return adj_list

def _try_local_search(n, g, deadline):
    """Try to find a graph of size n with girth >= g using local search."""
    best_adj = None
    best_girth = 0
    
    for seed in range(50): # Try a few random starts
        if time.time() > deadline - 5:
            break
            
        adj = _random_4_regular(n, seed)
        if adj is None:
            continue
            
        # Initial girth check
        if not _has_short_cycle(n, adj, g):
            return adj
            
        # Local search
        current_adj = adj
        for step in range(500): # Max steps per start
            if time.time() > deadline - 2:
                break
                
            cycle = _get_shortest_cycle(n, current_adj)
            if cycle is None:
                return current_adj
            
            cycle_len = len(cycle)
            if cycle_len >= g:
                return current_adj
                
            # Try to break the cycle
            # Pick a random edge in the cycle
            edge_idx = random.randint(0, len(cycle) - 1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % len(cycle)]
            
            # Try to swap (u, v) with another edge
            # Find a candidate edge (w, x) to swap with
            # We need to remove (u,v) and (w,x) and add (u,w) and (v,x) or (u,x) and (v,w)
            # Let's try random w, x
            
            swapped = False
            for _ in range(20): # 20 random swap attempts
                w = random.randint(0, n - 1)
                x = random.randint(0, n - 1)
                
                # Conditions for valid swap:
                # 1. w, x distinct from u, v
                # 2. (w,x) is an edge
                # 3. (u,w) and (v,x) are not edges (and not loops)
                # 4. (u,x) and (v,w) are not edges (if we choose that pairing)
                
                if w in [u, v] or x in [u, v]:
                    continue
                if w == x:
                    continue
                if x not in current_adj[w]:
                    continue
                
                # Check if we can form (u,w) and (v,x)
                if w in current_adj[u] or x in current_adj[v]:
                    continue
                if w == u or x == v: # Loop check
                    continue
                
                # Perform swap
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                current_adj[w].remove(x)
                current_adj[x].remove(w)
                
                current_adj[u].append(w)
                current_adj[w].append(u)
                current_adj[v].append(x)
                current_adj[x].append(v)
                
                # Check if this created a multi-edge or loop (should be prevented by checks)
                # And check if girth improved or stayed same
                # We just accept it and re-evaluate in next loop
                swapped = True
                break
                
            if not swapped:
                # If no swap found, try a different edge in cycle or break
                if random.random() < 0.5:
                    break

    return best_adj

def _luw_construct(k, g):
    """LUW construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 4:
        return _luw_construct(k, g)
    
    # k=4, g=10
    # Target: 320 vertices
    
    # 1. Try local search on 320 vertices
    n_target = 320
    adj = _try_local_search(n_target, g, deadline)
    
    if adj is not None:
        # Verify
        if not _has_short_cycle(n_target, adj, g):
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_target, edges
    
    # 2. Fallback to LUW
    return _luw_construct(k, g)
