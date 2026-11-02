import time
import random
import heapq
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

def _generate_random_k_regular(n, k, time_limit=5.0):
    """Generate a random simple k-regular graph using configuration model + 2-switches."""
    start_time = time.time()
    if n * k % 2 != 0:
        return None
    
    # Configuration model
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    random.shuffle(half_edges)
    
    # Pair up
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    
    if not valid:
        # Try to fix with 2-switches
        for _ in range(50000):
            if time.time() - start_time > time_limit:
                return None
            
            # Find all edges
            edge_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_list.append((u, v))
            
            if len(edge_list) == 0:
                return None
            
            # Pick two random edges
            idx1 = random.randint(0, len(edge_list) - 1)
            idx2 = random.randint(0, len(edge_list) - 1)
            if idx1 == idx2:
                continue
            
            a, b = edge_list[idx1]
            c, d = edge_list[idx2]
            
            # Check if these edges share a vertex
            if a == c or a == d or b == c or b == d:
                continue
            
            # Try switch: remove (a,b) and (c,d), add (a,c) and (b,d)
            if c in adj[a] or d in adj[b]:
                continue
            
            # Perform the switch
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[a].add(c)
            adj[c].add(a)
            adj[b].add(d)
            adj[d].add(b)
            
            # Check if this fixed all multi-edges
            is_regular = all(len(adj[u]) == k for u in range(n))
            if is_regular:
                return adj
    
    return None

def _check_girth_fast(adj, n, g, time_limit=10.0):
    """Check if graph has girth >= g using non-backtracking walks from each vertex."""
    start_time = time.time()
    
    for start in range(n):
        if time.time() - start_time > time_limit:
            return None
        
        # BFS tracking (node, prev_node, depth)
        visited = {}
        queue = deque()
        for neighbor in adj[start]:
            queue.append((neighbor, start, 1))
            visited[(neighbor, start)] = 1
        
        while queue:
            if time.time() - start_time > time_limit:
                return None
            node, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue  # non-backtracking
                if neighbor == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                if (neighbor, node) not in visited:
                    visited[(neighbor, node)] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return True

def _find_short_cycle(adj, n, g, time_limit=5.0):
    """Find a cycle shorter than g, or return None if girth >= g."""
    start_time = time.time()
    
    for start in range(n):
        if time.time() - start_time > time_limit:
            return None
        
        # BFS tracking (node, prev_node, depth, path)
        visited = {}
        queue = deque()
        for neighbor in adj[start]:
            queue.append((neighbor, start, 1, [start, neighbor]))
            visited[(neighbor, start)] = 1
        
        while queue:
            if time.time() - start_time > time_limit:
                return None
            node, prev, depth, path = queue.popleft()
            if depth >= g - 1:
                continue
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue  # non-backtracking
                if neighbor == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return path + [start]
                if (neighbor, node) not in visited:
                    visited[(neighbor, node)] = depth + 1
                    queue.append((neighbor, node, depth + 1, path + [neighbor]))
    
    return None

def _girth_switching_construct(k, g, n, time_limit=80.0):
    """Try to construct a k-regular graph of girth >= g using edge-switching."""
    start_time = time.time()
    
    # Generate initial random k-regular graph
    adj = None
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            return None
        adj = _generate_random_k_regular(n, k, time_limit=2.0)
        if adj is not None:
            break
    
    if adj is None:
        return None
    
    # Check initial girth
    girth_ok = _check_girth_fast(adj, n, g, time_limit=5.0)
    if girth_ok == True:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Build edge list for easy switching
    def build_edge_list():
        edge_list = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edge_list.append((i, j))
        return edge_list
    
    edge_list = build_edge_list()
    
    # Greedy switching: try to eliminate short cycles
    for step in range(1000000):
        if time.time() - start_time > time_limit - 10.0:
            # Do a final full check
            girth_ok = _check_girth_fast(adj, n, g, time_limit=5.0)
            if girth_ok == True:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            return None
        
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, g, time_limit=1.0)
        
        if cycle is None:
            # No short cycle found, do a full check
            girth_ok = _check_girth_fast(adj, n, g, time_limit=3.0)
            if girth_ok == True:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            elif girth_ok == None:
                continue
            else:
                # There's a short cycle but we didn't find it, try again
                continue
        
        # We found a short cycle. Try to break it by switching an edge in the cycle
        # Pick a random edge from the cycle
        cycle_edges = []
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            if u < v:
                cycle_edges.append((u, v))
            else:
                cycle_edges.append((v, u))
        
        # Find an edge in the cycle that we can switch
        switched = False
        for edge in cycle_edges:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            a, b = edge
            
            # Find another edge (c, d) that doesn't share vertices with (a, b)
            for _ in range(10):
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                idx2 = random.randint(0, len(edge_list) - 1)
                c, d = edge_list[idx2]
                
                # Check if these edges share a vertex
                if a == c or a == d or b == c or b == d:
                    continue
                
                # Try switch: remove (a,b) and (c,d), add (a,c) and (b,d)
                if c in adj[a] or d in adj[b]:
                    continue
                
                # Perform the switch
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                adj[a].add(c)
                adj[c].add(a)
                adj[b].add(d)
                adj[d].add(b)
                
                edge_list = build_edge_list()
                switched = True
                break
            
            if switched:
                break
        
        if not switched:
            # Try random switching
            if len(edge_list) < 2:
                continue
            
            idx1 = random.randint(0, len(edge_list) - 1)
            idx2 = random.randint(0, len(edge_list) - 1)
            if idx1 == idx2:
                continue
            
            a, b = edge_list[idx1]
            c, d = edge_list[idx2]
            
            if a == c or a == d or b == c or b == d:
                continue
            
            if c in adj[a] or d in adj[b]:
                continue
            
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[a].add(c)
            adj[c].add(a)
            adj[b].add(d)
            adj[d].add(b)
            
            edge_list = build_edge_list()
    
    # Final check
    girth_ok = _check_girth_fast(adj, n, g, time_limit=5.0)
    if girth_ok == True:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, try edge-switching
    if k == 5 and g == 9:
        # Try sizes near known best
        target_ns = [1116, 1120, 1150, 1200, 1300, 1500]
        
        time_budget = 100.0
        time_per_attempt = time_budget / len(target_ns)
        
        for n in target_ns:
            res = _girth_switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
    # Fallback to LUW
    return _luw_construct(k, g)
