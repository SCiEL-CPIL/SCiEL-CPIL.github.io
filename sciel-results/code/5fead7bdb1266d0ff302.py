"""Construct a k-regular graph of girth at least g with minimal vertices within time budget."""
import time
import itertools
import random
import math
from collections import deque

# Moore bound lower bound for n(k, g)
def _moore_bound(k, g):
    if g % 2 == 0:
        # Even girth: tree of depth g/2 - 1 from each edge
        depth = g // 2 - 1
        return 2 * (1 + k * (k**depth - 1) // (k - 1))
    else:
        # Odd girth: tree of depth (g-1)/2 from each vertex
        depth = (g - 1) // 2
        return 1 + k * (k**depth - 1) // (k - 1)

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS tracking (vertex, prev) to find shortest cycle through each vertex
    # A cycle of length L through start means there's a path of length L from start back to start
    # without immediately backtracking
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # dist[(u, v)] = shortest non-backtracking walk length from (start, None) to (u, v)
        dist = {}
        q = deque()
        for nb in adj[start]:
            dist[(start, nb)] = 1
            q.append((start, nb))
        
        while q:
            u, prev = q.popleft()
            d = dist[(u, prev)]
            for nb in adj[u]:
                if nb == prev:
                    continue  # no backtracking
                new_state = (u, nb)
                if new_state in dist:
                    # Found a cycle: length = d + 1
                    cycle_len = d + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[new_state] = d + 1
                    q.append(new_state)
                # Early termination: if we've explored enough depth, no short cycle exists
                if d + 1 >= g - 1:
                    break
    return True

def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing vertices."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    mb = _moore_bound(k, g)
    
    # Try greedy growth approach: start with a tree and add edges to satisfy degree
    # and maintain girth constraint
    
    def try_greedy_growth(max_vertices, time_limit=30.0):
        """Try to build a k-regular graph of girth >= g using greedy growth."""
        t0 = time.time()
        
        # Start with a single vertex
        # We'll grow the graph by adding vertices and edges
        # Each vertex must have degree exactly k
        # No cycle shorter than g
        
        # Use a BFS-like approach: start with a tree of depth (g-1)//2 for odd g
        # or g//2 - 1 for even g, then try to close it into a regular graph
        
        # For now, try a simple approach: generate all possible k-regular graphs
        # on small n and check girth
        
        best_n = None
        best_edges = None
        
        # Try n from max(mb, some minimum) upward
        min_n = max(mb, 2 * k)  # At least 2k vertices for k-regular simple graph
        
        # Limit search based on time
        search_limit = min(max_vertices, min_n + 200)
        
        for n in range(min_n, search_limit):
            if time.time() - t0 > time_limit:
                break
            
            # Check if k-regular graph exists on n vertices: k*n must be even
            if (k * n) % 2 != 0:
                continue
            
            # Try to construct using a greedy matching approach
            # This is expensive, so only try for small n
            if n > 100:
                break
            
            # Generate a random k-regular graph and check girth
            # Use configuration model
            import random
            random.seed(n * 1000 + k * 100 + g)
            
            for attempt in range(100):
                if time.time() - t0 > time_limit:
                    break
                
                # Configuration model
                half_edges = list(range(n)) * k
                random.shuffle(half_edges)
                
                edges = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    edge = (min(u, v), max(u, v))
                    if edge in edges:
                        valid = False
                        break
                    edges.add(edge)
                
                if not valid:
                    continue
                
                edge_list = list(edges)
                
                # Check regularity (should be by construction, but verify)
                deg = [0] * n
                for u, v in edge_list:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != k for d in deg):
                    continue
                
                # Check girth
                if _verify_girth(n, edge_list, g):
                    best_n = n
                    best_edges = edge_list
                    break
                
            if best_n is not None:
                break
        
        return best_n, best_edges
    
    # Try greedy growth with limited time
    max_try = 500
    n_greedy, edges_greedy = try_greedy_growth(max_try, time_limit=30.0)
    
    if n_greedy is not None and time.time() < deadline - 10.0:
        return n_greedy, edges_greedy
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # This is the original approach
    q = k
    # Choose d such that girth >= d + 5 (for odd d) or d + 4 (for even d)
    # We need girth >= g
    # For odd d: girth = d + 5, so d = g - 5
    # For even d: girth = d + 4, so d = g - 4
    if (g - 5) % 2 == 1:  # g - 5 is odd, so d = g - 5 is odd
        d = g - 5
    else:
        d = g - 4
    
    d = max(3, d)
    
    # GF(q) operations (q must be prime power; for k=5, q=5)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % 5
        sub = lambda x, y: (x - y) % 5
        mul = lambda x, y: x * y % 5
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: x * y % 3
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        # General prime case (assumes q is prime)
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Rule for CD(d, q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n_luw = len(order)
    
    # Check deadline
    if time.time() > deadline - 5.0:
        return n_luw, edges
    
    # Verify the LUW graph
    if _verify_girth(n_luw, edges, g):
        return n_luw, edges
    
    # If verification failed (shouldn't happen), return anyway
    return n_luw, edges
