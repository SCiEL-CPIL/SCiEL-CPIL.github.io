import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k == 4 and g == 10:
        # Try to find the (4,10)-cage (n=320)
        # Allocate time to each attempt, decreasing as we go
        attempts = [320, 322, 324, 326, 328, 330, 332, 334]
        total_time = deadline - start_time - 15  # reserve 15s for fallback
        time_per_attempt = total_time / len(attempts)
        
        for i, n_target in enumerate(attempts):
            remaining = deadline - start_time - 15
            if remaining < 5:
                break
            # Allocate proportional time, more for smaller n
            alloc = min(time_per_attempt, remaining)
            sub_deadline = min(deadline - 15, start_time + time.time() - start_time + alloc)
            result = find_small_graph(k, g, n_target, sub_deadline)
            if result:
                return result
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

def find_small_graph(k, g, n, deadline):
    """Simulated Annealing to find a k-regular graph with girth >= g."""
    # Try multiple random initial graphs, keep the best one
    best_girth = 0
    best_adj = None
    best_adj_set = None
    
    # Generate a few candidate initial graphs and pick the best
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        edges = random_k_regular(n, k)
        if not edges:
            continue
            
        adj = [[] for _ in range(n)]
        adj_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            key = (min(u, v), max(u, v))
            adj_set.add(key)
        
        girth = get_girth(adj, g)
        if girth > best_girth:
            best_girth = girth
            best_adj = adj
            best_adj_set = adj_set
            if girth >= g:
                return n, list(adj_set)
    
    if best_adj is None:
        return None
        
    adj = best_adj
    adj_set = best_adj_set
    current_girth = best_girth
    
    if current_girth >= g:
        return n, list(adj_set)
        
    # SA Parameters - tuned for girth maximization
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 5000
    
    # Precompute edge list for fast access
    edge_list = list(adj_set)
    
    while time.time() < deadline - 3:
        # Pick random edges
        e1 = random.choice(edge_list)
        e2 = random.choice(edge_list)
        if e1 == e2:
            continue
            
        u, v = e1
        x, y = e2
        
        # Try swap: remove (u,v) and (x,y), add (u,x) and (v,y)
        # Check validity
        if u == x or v == y:
            continue
        ux_key = (min(u, x), max(u, x))
        vy_key = (min(v, y), max(v, y))
        if ux_key in adj_set or vy_key in adj_set:
            continue
        # Also check we're not creating a loop
        if u == x or v == y:
            continue
            
        # Apply swap
        adj_set.discard(e1)
        adj_set.discard(e2)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj_set.add(ux_key)
        adj_set.add(vy_key)
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
        # Update edge_list (lazy: remove old, add new)
        try:
            edge_list.remove(e1)
        except ValueError:
            pass
        try:
            edge_list.remove(e2)
        except ValueError:
            pass
        edge_list.append(ux_key)
        edge_list.append(vy_key)
        
        # Evaluate
        new_girth = get_girth(adj, g)
        
        if new_girth >= g:
            return n, list(adj_set)
            
        delta = new_girth - current_girth
        if delta > 0:
            current_girth = new_girth
            no_improve_count = 0
        else:
            # Accept with probability exp(delta/T)
            if random.random() < math.exp(delta / T):
                current_girth = new_girth
                no_improve_count += 1
            else:
                # Revert
                adj_set.discard(ux_key)
                adj_set.discard(vy_key)
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
                
                try:
                    edge_list.remove(ux_key)
                except ValueError:
                    pass
                try:
                    edge_list.remove(vy_key)
                except ValueError:
                    pass
                edge_list.append(e1)
                edge_list.append(e2)
                no_improve_count += 1
        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            T = 1.0
    
    return None

def get_girth(adj, g_limit):
    """Fast girth check: returns min(girth, g_limit).
    Uses depth-limited BFS with early termination."""
    n = len(adj)
    max_depth = g_limit // 2  # we only need to detect cycles shorter than g_limit
    
    for start in range(n):
        # BFS with parent tracking to avoid counting the edge back to parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge: cycle length = dist[u] + dist[v] + 1
                    # But we need to make sure this isn't just the tree edge
                    cycle_len = d + dist[v] + 1
                    if cycle_len < g_limit:
                        return cycle_len
                        
    return g_limit

def random_k_regular(n, k):
    """Generate a random k-regular graph using the configuration model with rejection.
    For k=4, we use a smarter approach: generate a random permutation and pair stubs
    to minimize short cycles."""
    if k == 4:
        # Use a 2-regular base (cycle) and add edges to make it 4-regular
        # Start with two disjoint Hamiltonian cycles
        perm1 = list(range(n))
        random.shuffle(perm1)
        perm2 = list(range(n))
        random.shuffle(perm2)
        
        edges = set()
        # First cycle
        for i in range(n):
            u = perm1[i]
            v = perm1[(i + 1) % n]
            if u > v:
                u, v = v, u
            if u == v:
                return random_k_regular(n, k)  # retry
            edges.add((u, v))
        
        # Second cycle
        for i in range(n):
            u = perm2[i]
            v = perm2[(i + 1) % n]
            if u > v:
                u, v = v, u
            if u == v:
                return random_k_regular(n, k)  # retry
            if (u, v) in edges:
                return random_k_regular(n, k)  # retry, duplicate edge
            edges.add((u, v))
        
        return list(edges)
    else:
        # General case: configuration model
        for _ in range(1000):
            stubs = []
            for i in range(n):
                for _ in range(k):
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
        return []

def luw_construct(k, g):
    # Same as seed, but ensure it works
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
