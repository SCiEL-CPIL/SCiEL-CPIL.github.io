import time
import random
import sys
from collections import deque

def _is_valid_4_regular(adj, n):
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    # Check simplicity (no loops, no multi-edges)
    # Since we generate via swaps, we mostly check the new edges
    return True

def _count_short_cycles(adj, n, max_girth):
    """
    Counts the number of cycles of length < max_girth.
    Returns 0 if girth >= max_girth.
    """
    count = 0
    # We only care about cycles up to max_girth - 1
    # BFS depth needed: max_girth // 2
    depth_limit = max_girth // 2
    
    for start in range(n):
        # BFS
        visited = [-1] * n
        parent = [-1] * n
        visited[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            depth = visited[u]
            if depth >= depth_limit:
                continue
                
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Cycle length is depth[u] + visited[v] + 1
                    # But this might count the same cycle multiple times or longer ones
                    # We just want to know if ANY short cycle exists.
                    # To get a "score" for annealing, we can count the number of such backtracking-free hits.
                    # A hit at depth d and d' implies a cycle of length d + d' + 1.
                    # If this length < max_girth, it's a "bad" cycle.
                    cycle_len = depth + visited[v] + 1
                    if cycle_len < max_girth:
                        count += 1
    return count

def _generate_random_4_regular(n):
    """
    Generates a random 4-regular simple graph on n vertices.
    Uses configuration model and fixes issues.
    """
    if n < 5:
        return None
        
    while True:
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        valid = True
        
        # Pair stubs
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
                
            adj[u].append(v)
            adj[v].append(u)
            
        if valid:
            return adj
            
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    """
    if k != 4 or g != 10:
        # Fallback to seed logic for other parameters
        # (Copy of seed logic)
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

    # Specialized search for k=4, g=10
    deadline = time.time() + 110  # Leave 10s for return and safety
    
    # Try sizes from 250 to 320
    # Step 2 to allow odd/even, but 4-regular usually works on even N for bipartite-like cages
    # However, we can try any N >= 5.
    
    best_result = None
    best_n = float('inf')
    
    # We will try to find a solution.
    # Start with 320 (known to exist). Then try smaller.
    # Or start with smaller and go up?
    # If we find a valid one at 250, that's better than 320.
    # So iterate 250, 252, ..., 320.
    
    for n in range(250, 321, 2):
        if time.time() > deadline:
            break
            
        # Allocate time for this N
        # If we have 100s left, and 30 values to try, ~3s each?
        # Better: Try 320 first to ensure we have a valid answer.
        
    # Actually, let's try 320 first.
    # If we find it, we have a valid graph.
    # Then we can try to find smaller ones if time permits.
    
    # Phase 1: Find any valid graph (Start with 320)
    n_target = 320
    found = False
    adj = None
    
    # Try to find a valid graph at n=320
    start_time = time.time()
    while time.time() - start_time < 20: # 20 seconds for 320
        adj = _generate_random_4_regular(n_target)
        if adj is None:
            continue
            
        # Simulated Annealing
        T = 1.0
        current_cost = _count_short_cycles(adj, n_target, g)
        
        if current_cost == 0:
            found = True
            break
            
        while time.time() - start_time < 20:
            # Pick random edges
            # Get list of edges
            # To speed up, maintain edge list?
            # For now, pick random u, v from adj[u]
            
            u = random.randint(0, n_target - 1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            # Pick another edge (x, y) disjoint from (u, v)
            # Retry loop
            x = random.randint(0, n_target - 1)
            y = -1
            attempts = 0
            while attempts < 10:
                x = random.randint(0, n_target - 1)
                if x == u or x == v:
                    attempts += 1
                    continue
                if not adj[x]:
                    attempts += 1
                    continue
                y = random.choice(adj[x])
                if y == u or y == v:
                    attempts += 1
                    continue
                if x == y: # Loop? No, simple graph
                    attempts += 1
                    continue
                # Check if (x,y) is the same as (u,v) or (v,u)
                if (x == u and y == v) or (x == v and y == u):
                    attempts += 1
                    continue
                break
            else:
                continue
                
            # Swap (u,v) and (x,y) to (u,x) and (v,y)
            # Remove old edges
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add new edges
            # Check for multi-edges
            if x in adj[u] or y in adj[v]:
                # Revert
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                continue
                
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_cost = _count_short_cycles(adj, n_target, g)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    found = True
                    break
            else:
                # Annealing
                delta = new_cost - current_cost
                if random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)
                    
            T = max(0.1, T * 0.99)
            
    if found:
        # Convert adj to edges
        edges = []
        for i in range(n_target):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n_target, edges
        
    # Phase 2: Try smaller N if time permits
    # We have ~90s left.
    # Try 250 to 318.
    
    for n in range(250, 320, 2):
        if time.time() > deadline - 5:
            break
            
        # Give each N a small time slice, e.g., 5 seconds
        start_time = time.time()
        
        adj = _generate_random_4_regular(n)
        if adj is None:
            continue
            
        T = 1.0
        current_cost = _count_short_cycles(adj, n, g)
        
        if current_cost == 0:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
        while time.time() - start_time < 5:
            # Same SA loop as above
            u = random.randint(0, n - 1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            x = random.randint(0, n - 1)
            y = -1
            attempts = 0
            while attempts < 10:
                x = random.randint(0, n - 1)
                if x == u or x == v:
                    attempts += 1
                    continue
                if not adj[x]:
                    attempts += 1
                    continue
                y = random.choice(adj[x])
                if y == u or y == v:
                    attempts += 1
                    continue
                if (x == u and y == v) or (x == v and y == u):
                    attempts += 1
                    continue
                break
            else:
                continue
                
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            if x in adj[u] or y in adj[v]:
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                continue
                
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_cost = _count_short_cycles(adj, n, g)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
            else:
                delta = new_cost - current_cost
                if random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                else:
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)
                    
            T = max(0.1, T * 0.99)
            
    # Fallback: Seed construction
    q = 4
    d = 5 # g-5 = 5
    # ... (Copy seed logic) ...
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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
