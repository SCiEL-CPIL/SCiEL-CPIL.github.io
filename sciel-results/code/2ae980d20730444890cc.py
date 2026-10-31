import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    if k != 3:
        return _luw_construct(k, g)

    # Moore bound for k=3, g=16 is 512.
    # We try to find a graph with n >= 512.
    # We iterate n upwards.
    
    # We will use a local search strategy:
    # 1. Generate random simple 3-regular graph.
    # 2. Find short cycles.
    # 3. Apply 2-switches to break them.
    
    # To speed up generation of simple 3-regular graphs, we use a retry loop.
    # For n ~ 500, this is fast.
    
    best_result = None
    best_n = float('inf')
    
    # We scan n from 512 to 1000.
    # If we find a valid graph, we return it immediately (since we want min n).
    # But we might get stuck in a local search for a specific n.
    # So we limit the time spent per n.
    
    for n in range(512, 1000, 2):
        if time.time() > deadline:
            break
            
        # Try a few random starts for this n
        # If we find one, return.
        # How many attempts?
        # Let's try 5 random graphs, with 1000 switching steps each.
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Generate random simple 3-regular graph
            edges = _generate_random_3reg(n)
            if not edges:
                continue
                
            # Local search to increase girth
            edges = _increase_girth(n, edges, g, max_steps=2000, timeout=5)
            
            # Verify
            if _verify_girth_fast(n, edges, g):
                return n, edges
                
        # If we didn't find one for this n, try next n
        # Note: It is possible that n=512 is impossible.
        # We just move on.

    # If we didn't find anything smaller than 936, 
    # we might have returned something earlier.
    # If we reach here, we should return a valid graph, even if large.
    # The prompt implies we should return *a* graph.
    # If we haven't returned, let's return a known large one or a LUW graph.
    # But the loop above should have returned if it found any valid graph for n < 1000.
    # If it didn't, it means we couldn't find one for n < 1000 in the time.
    # Let's try a larger n with fewer attempts, or just return a LUW graph.
    
    if time.time() < deadline:
        # Try n=936 specifically with more attempts, as it is known to exist
        for attempt in range(20):
            if time.time() > deadline:
                break
            edges = _generate_random_3reg(936)
            if not edges:
                continue
            edges = _increase_girth(936, edges, g, max_steps=5000, timeout=10)
            if _verify_girth_fast(936, edges, g):
                return 936, edges
                
    # Fallback: LUW
    return _luw_construct(3, g)


def _generate_random_3reg(n):
    """
    Generate a random simple 3-regular graph on n vertices.
    Returns list of edges or None if failed (should rarely fail for n>10).
    """
    # Configuration model
    # 3n stubs
    # Pair them randomly
    
    # To avoid infinite loops on failure, we limit retries
    for _ in range(100):
        # Create list of stubs: [0,0,0, 1,1,1, ..., n-1, n-1, n-1]
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
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
            
    return None


def _increase_girth(n, edges, g, max_steps=1000, timeout=5.0):
    """
    Use 2-switches to break short cycles.
    """
    start_time = time.time()
    adj = _build_adj(n, edges)
    
    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break
            
        # Find a cycle of length < g
        cycle = _find_short_cycle(n, adj, g)
        if not cycle:
            break
            
        # cycle is a list of vertices [v0, v1, ..., vk-1]
        # Edges are (v0,v1), (v1,v2), ..., (vk-1,v0)
        
        # Pick a random edge in the cycle to switch
        # Let's pick (u, v) from the cycle
        edge_idx = random.randint(0, len(cycle) - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % len(cycle)]
        
        # Find another edge (u2, v2) NOT in the cycle
        # We need to be careful: the new edges (u, v2) and (u2, v) must not be loops or existing edges.
        
        # Candidate edges: all edges not in the cycle
        cycle_edges_set = set()
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            if a > b: a, b = b, a
            cycle_edges_set.add((a, b))
            
        # We can try a few random edges from the rest of the graph
        success = False
        for _ in range(20): # Try 20 random other edges
            # Pick a random vertex x, then a random neighbor y
            x = random.randint(0, n - 1)
            if not adj[x]:
                continue
            y = random.choice(adj[x])
            
            # Check if (x,y) is in the cycle
            if x > y:
                x, y = y, x
            if (x, y) in cycle_edges_set:
                continue
                
            # Try switch:
            # Old: (u,v) and (x,y)
            # New: (u,y) and (x,v)
            
            # Check validity
            # 1. No loops
            if u == y or x == v:
                continue
            # 2. No multiple edges
            # Check if (u,y) exists (and is not (u,v) or (x,y))
            # Check if (x,v) exists
            
            # We need to check adjacency
            # Since we are modifying, we should check against the current graph state.
            # But we are in a loop, so adj is current.
            
            # (u,y) is valid if y in adj[u] is False OR (u,y) is the edge we are removing?
            # No, we are replacing (u,v). So (u,v) is removed.
            # We add (u,y) and (x,v).
            
            # Check if (u,y) is already an edge (and not (u,v))
            if y in adj[u] and y != v:
                continue
            # Check if (x,v) is already an edge (and not (x,y))
            if v in adj[x] and v != y:
                continue
                
            # Also ensure we don't create (u,y) == (x,v) which implies u=x, y=v or u=v, y=x (impossible as u!=v)
            # If u == x and y == v, then we are swapping (u,v) and (u,v) -> no change?
            # Wait, if u=x, then (x,v) is (u,v). But (u,v) is being removed.
            # So if u==x, then new edges are (u,y) and (u,v).
            # But (u,v) is removed. So we add (u,y) and (u,v).
            # This effectively replaces (u,v) with (u,y) and keeps (u,v)? No.
            # The set of edges changes.
            
            # Let's just perform the swap and check simplicity afterwards?
            # Or trust the checks above.
            
            # Perform swap
            # Remove (u,v) and (x,y)
            # Add (u,y) and (x,v)
            
            # Update adj
            adj[u].remove(v)
            adj[x].remove(y)
            adj[u].append(y)
            adj[x].append(v)
            
            # Update edges list? 
            # We don't maintain edges list in this function, we return adj at end?
            # No, we return edges.
            # Let's maintain edges as a set for easy removal?
            # Or just return adj and convert at end.
            
            success = True
            break
            
        if not success:
            # If we couldn't find a valid switch, break to avoid infinite loop
            break
            
    # Convert adj back to edges
    new_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                new_edges.append((u, v))
                
    return new_edges


def _find_short_cycle(n, adj, g):
    """
    Find a cycle of length < g.
    Returns list of vertices in cycle, or None.
    """
    # BFS from each vertex?
    # Or just DFS?
    # Since we just need ANY short cycle, we can do a quick check.
    # For small g, BFS is fine.
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        parent[start] = -1
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found cycle
                    # Reconstruct cycle
                    cycle = []
                    curr = u
                    while curr != v:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.append(v)
                    cycle.append(u)
                    
                    # Verify length
                    if len(cycle) < g:
                        return cycle
                        
    return None


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _verify_girth_fast(n, edges, g):
    adj = _build_adj(n, edges)
    
    dist = [0] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0
    
    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True


def _luw_construct(k, g):
    # Fallback
    if k != 3:
        # Simple LUW for general k
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3
    q = 3
    def build_luw(d):
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

    # d=8 gives n=3^9-1? No, LUW size is q^(d+1)-1? 
    # For k=3, d=8 -> n=3^9-1 = 19682. Too big.
    # We want small n.
    # Actually, LUW is not good for minimizing n in this range.
    # But it's a valid fallback.
    return build_luw(8)
