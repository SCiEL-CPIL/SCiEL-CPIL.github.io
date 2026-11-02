import time
import random
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _compute_girth(adjacency, n):
    """Compute the girth of the graph using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        if min_girth == 3:
            break
    return min_girth if min_girth != float('inf') else -1

def _find_short_cycle(adjacency, n, max_girth):
    """Find a cycle of length <= max_girth if one exists. Returns list of vertices or None."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len <= max_girth:
                        # Reconstruct cycle
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                        
                        if lca is None:
                            continue
                            
                        idx_u = path_u.index(lca)
                        idx_v = path_v.index(lca)
                        cycle = path_u[:idx_u+1][::-1] + path_v[:idx_v][::-1]
                        
                        if cycle[0] == cycle[-1]:
                            cycle = cycle[:-1]
                        
                        if len(cycle) > 1:
                            return cycle
    return None

def _has_short_path(adj, src, dst, max_dist, excluded_edges):
    """Check if there's a path from src to dst of length <= max_dist avoiding excluded_edges."""
    if src == dst:
        return 0
    visited = {src: 0}
    queue = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for neighbor in adj[node]:
            if (node, neighbor) in excluded_edges or (neighbor, node) in excluded_edges:
                continue
            if neighbor == dst:
                return dist + 1
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    return -1

def _is_2_switch_valid(adjacency, u, v, w, x, g):
    """Check if performing 2-switch (u,v)->(u,w),(v,x) doesn't create short cycles."""
    excluded = {(u, v), (v, u), (w, x), (x, w)}
    
    # Check if u-w path exists with length < g-1
    if _has_short_path(adjacency, u, w, g - 2, excluded) != -1:
        return False
    
    # Check if v-x path exists with length < g-1
    if _has_short_path(adjacency, v, x, g - 2, excluded) != -1:
        return False
    
    return True

def _generate_random_k_regular(n, k):
    """Generate a random simple k-regular graph using configuration model with 2-switches."""
    # Try multiple times to get a valid simple k-regular graph
    for attempt in range(100):
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair up stubs
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if not valid:
            continue
        
        # Check all vertices have degree k
        valid = True
        for i in range(n):
            if len(adjacency[i]) != k:
                valid = False
                break
        
        if valid:
            return adjacency
    
    return None

def _switching_construct(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate a random k-regular graph
    adjacency = _generate_random_k_regular(n, k)
    if adjacency is None:
        return None
    
    # Check initial girth
    current_girth = _compute_girth(adjacency, n)
    
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Try to increase girth using 2-switches
    max_iterations = 20000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Find a short cycle
        short_cycle = _find_short_cycle(adjacency, n, g - 1)
        if short_cycle is None:
            # No short cycle found, we're done
            edges = []
            for i in range(n):
                for j in adjacency[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        # Try to break the short cycle using a 2-switch
        cycle_len = len(short_cycle)
        
        if cycle_len >= 4:
            # Pick two non-adjacent edges in the cycle
            attempts = 0
            while attempts < 10:
                attempts += 1
                edge1_idx = random.randint(0, cycle_len - 1)
                edge2_idx = random.randint(0, cycle_len - 1)
                
                u1 = short_cycle[edge1_idx]
                v1 = short_cycle[(edge1_idx + 1) % cycle_len]
                u2 = short_cycle[edge2_idx]
                v2 = short_cycle[(edge2_idx + 1) % cycle_len]
                
                # Check that these edges are non-adjacent (no shared vertices)
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue
                
                # Check that (u1, u2) and (v1, v2) are not already edges
                if u2 in adjacency[u1] or v2 in adjacency[v1]:
                    continue
                
                # Check if switch is valid
                if _is_2_switch_valid(adjacency, u1, v1, u2, v2, g):
                    # Perform switch
                    adjacency[u1].remove(v1)
                    adjacency[v1].remove(u1)
                    adjacency[u2].remove(v2)
                    adjacency[v2].remove(u2)
                    adjacency[u1].add(u2)
                    adjacency[u2].add(u1)
                    adjacency[v1].add(v2)
                    adjacency[v2].add(v1)
                    break
        else:
            # Triangle, try random 2-switch
            u = random.randint(0, n - 1)
            v = random.choice(list(adjacency[u]))
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                for x in adjacency[w]:
                    if x == u or x == v or x == w:
                        continue
                    if x in adjacency[u] or w in adjacency[v]:
                        continue
                    candidates.append((w, x))
            if not candidates:
                continue
            w, x = random.choice(candidates)
            
            if _is_2_switch_valid(adjacency, u, v, w, x, g):
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                adjacency[w].remove(x)
                adjacency[x].remove(w)
                adjacency[u].add(w)
                adjacency[w].add(u)
                adjacency[v].add(x)
                adjacency[x].add(v)
    
    # Check final girth
    final_girth = _compute_girth(adjacency, n)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, try switching algorithm
    if k == 5 and g == 9:
        # Try switching algorithm with various n values
        # Start near Moore bound and go up
        # Moore bound for (5,9): 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5 * 85 = 426
        # But the problem says lower bound is 428.
        
        # Let's try n values starting from around 428 and going up
        target_ns = [428, 450, 500, 600, 800, 1000, 1200, 1500, 2000]
        
        # Time budget: 100s total
        total_time = 100.0
        start_time = time.time()
        
        for n in target_ns:
            remaining = total_time - (time.time() - start_time)
            if remaining < 5:
                break
            time_per_attempt = min(15.0, remaining / 2)
            
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
