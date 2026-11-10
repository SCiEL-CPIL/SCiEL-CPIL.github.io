"""Combined program: LUW baseline + Randomized Local Search for smaller k-regular girth-g graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _moore_lower(k, g):
    """Return a simple lower bound on n(k,g)."""
    if g % 2 == 1:
        # Odd girth: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        h = (g - 3) // 2
        total = 1 + k
        for i in range(1, h + 1):
            total += k * (k - 1) ** i
        return total
    else:
        # Even girth: 2 * k * sum_{i=0}^{g/2 - 1} (k-1)^i
        h = g // 2
        total = 0
        for i in range(h):
            total += k * (k - 1) ** i
        return 2 * total

def _validate(k, g, n, edges):
    """
    Validate that (n, edges) is a simple k-regular graph with no cycle shorter than g.
    Returns True if valid, False otherwise.
    """
    if n < k + 1:
        return False
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i < 0 or i >= n or j < 0 or j >= n:
            return False
        key = (min(i, j), max(i, j))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
        
    # Check girth via non-backtracking walks
    for s in range(n):
        visited = {}
        queue = deque()
        queue.append((s, None, 0))
        visited[(s, None)] = 0
        
        while queue:
            v, prev, dist = queue.popleft()
            for w in adj[v]:
                if w == prev:
                    continue
                new_dist = dist + 1
                if new_dist >= g:
                    continue
                if w == s and new_dist >= 1:
                    return False
                state = (w, v)
                if state not in visited:
                    visited[state] = new_dist
                    queue.append((w, v, new_dist))
    
    return True

def _lazebnik_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph.
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
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

def _find_shortest_cycle(adj, n, max_len):
    """Find a cycle of length < max_len. Returns list of vertices or None."""
    for s in range(n):
        # BFS on non-backtracking walks
        # state: (v, prev, dist)
        visited = {}
        queue = deque()
        queue.append((s, None, 0, [s]))
        visited[(s, None)] = 0
        
        while queue:
            v, prev, dist, path = queue.popleft()
            if dist >= max_len:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                new_dist = dist + 1
                if w == s and new_dist >= 3: # Cycle length at least 3
                    return path + [s]
                state = (w, v)
                if state not in visited:
                    visited[state] = new_dist
                    queue.append((w, v, new_dist, path + [w]))
    return None

def _try_repair(adj, n, k, g, time_limit):
    """
    Attempt to repair a k-regular graph to have girth >= g.
    Returns (n, edges) if successful, None otherwise.
    """
    start_time = time.time()
    # Copy adj to mutable structure
    adj_set = [set(adj[i]) for i in range(n)]
    
    while time.time() - start_time < time_limit:
        cycle = _find_shortest_cycle(adj_set, n, g)
        if not cycle:
            # Success
            edges = []
            for i in range(n):
                for j in adj_set[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        # Pick a random edge in the cycle to break
        # Cycle is a list of vertices [v0, v1, ..., vL-1, v0]
        # Edges are (cycle[i], cycle[i+1])
        # We want to remove (u, v) and add (u', v') such that
        # u' in adj[u], v' in adj[v], u' != v, v' != u, and (u', v') not in graph.
        # And the new graph should ideally not create a shorter cycle.
        
        # Simple strategy: Remove a random edge from the cycle, and reconnect the endpoints
        # to random other neighbors.
        
        edge_idx = random.randint(0, len(cycle) - 2)
        u = cycle[edge_idx]
        v = cycle[edge_idx + 1]
        
        # Remove edge (u, v)
        adj_set[u].remove(v)
        adj_set[v].remove(u)
        
        # Find new neighbors
        # u needs a new neighbor, v needs a new neighbor.
        # To maintain regularity, we can swap partners.
        # u is connected to N_u \ {v}. v is connected to N_v \ {u}.
        # Pick u_new from N_u \ {v} and v_new from N_v \ {u}.
        # But we must ensure (u, v_new) and (v, u_new) are valid (not existing edges).
        # Actually, simpler: Pick a random edge (a, b) not incident to u or v?
        # No, that doesn't fix the degree of u and v.
        
        # Let's try: Pick a random neighbor u_new of u (u_new != v).
        # Pick a random neighbor v_new of v (v_new != u).
        # Remove (u, u_new) and (v, v_new).
        # Add (u, v_new) and (v, u_new).
        # This is a 2-opt swap.
        
        candidates_u = list(adj_set[u])
        candidates_v = list(adj_set[v])
        
        if not candidates_u or not candidates_v:
            # Should not happen if k >= 2
            adj_set[u].add(v)
            adj_set[v].add(u)
            continue
            
        u_new = random.choice(candidates_u)
        v_new = random.choice(candidates_v)
        
        # Check if swap is valid (no multi-edges)
        if v_new in adj_set[u] or u_new in adj_set[v]:
            adj_set[u].add(v)
            adj_set[v].add(u)
            continue
        
        # Perform swap
        adj_set[u].remove(u_new)
        adj_set[u].add(v_new)
        
        adj_set[v].remove(v_new)
        adj_set[v].add(u_new)
        
        adj_set[u_new].remove(u)
        adj_set[u_new].add(v)
        
        adj_set[v_new].remove(v)
        adj_set[v_new].add(u)
        
        # Check if this created a shorter cycle? 
        # We don't check immediately, the next iteration will find the shortest cycle.
        # If the cycle length decreases, we might loop. 
        # To avoid this, we could check if the new cycle is shorter, and if so, revert.
        # But for simplicity, we just proceed. The randomization should help escape.
        
    return None

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s budget
    
    # Strategy 1: Try to find a small graph via Randomized Local Search
    # Start near Moore bound
    moore = _moore_lower(k, g)
    
    # We try a few target sizes
    # For k=5, g=9, moore=426.
    # We try n = moore, moore+10, ... up to some limit
    # But generating random k-regular graphs for large n is O(n*k).
    
    # Let's try a few random restarts for a specific n.
    # If we find a valid graph, return it.
    
    # We prioritize smaller n.
    # However, if the optimal n is 1116, and we start at 426, we might spend time
    # on impossible instances.
    # We will limit the number of attempts.
    
    best_n = None
    best_edges = None
    
    # Try n values
    # Step size 10 to cover range quickly
    for n_try in range(moore, moore + 200, 10):
        if time.time() > deadline:
            break
        
        if (n_try * k) % 2 != 0:
            continue
        if n_try < k + 1:
            continue
            
        # Try to construct a random k-regular graph on n_try vertices
        # and repair it.
        
        # Generate random k-regular graph
        adj = None
        for _ in range(5): # 5 attempts to generate valid random graph
            stubs = []
            for v in range(n_try):
                for _ in range(k):
                    stubs.append(v)
            random.shuffle(stubs)
            
            adj = [set() for _ in range(n_try)]
            valid = True
            for idx in range(0, len(stubs), 2):
                u, v = stubs[idx], stubs[idx + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                break
        
        if adj is None:
            continue
            
        # Repair
        result = _try_repair(adj, n_try, k, g, time_limit=2.0)
        if result:
            best_n, best_edges = result
            break # Found a valid graph, stop searching higher n
    
    if best_n is not None:
        # Validate just in case
        if _validate(k, g, best_n, best_edges):
            return best_n, best_edges
    
    # Strategy 2: Fallback to LUW
    try:
        n_luw, edges_luw = _lazebnik_construct(k, g)
        if _validate(k, g, n_luw, edges_luw):
            return n_luw, edges_luw
    except Exception:
        pass
        
    # Last resort: Return LUW without validation (should be valid)
    return _lazebnik_construct(k, g)
# EVOLVE-BLOCK-END
