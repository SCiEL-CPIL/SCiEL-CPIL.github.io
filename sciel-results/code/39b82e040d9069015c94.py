import random
import time
from collections import deque

def construct(k, g):
    t_start = time.time()
    DEADLINE = t_start + 135  # Leave 15s safety margin for the 150s limit
    
    # Moore bound: for k-regular girth g (even), n >= 2 * (k-1)^(g/2 - 1)
    # For k=4, g=10: 2 * 3^4 = 162... wait, let me recalculate
    # Actually for even girth g = 2r: n >= 2 * (k-1)^(r-1) = 2 * 3^4 = 162? 
    # No, the Moore bound for (k, g) with g even is: n >= 2 * sum_{i=0}^{r-1} (k-1)^i where r = g/2
    # For k=4, g=10, r=5: n >= 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242
    # For g=9 (odd): n >= (k-1) * sum_{i=0}^{r-1} (k-1)^i where r = (g-1)/2
    # But we're targeting girth 10, so even case applies.
    # Moore bound = 242 (or 243 as stated in the problem)
    
    def seed_construct():
        """Algebraic construction for the cage graph."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    def verify_girth(adj, n, required_girth):
        """Check that the graph has girth >= required_girth using BFS."""
        # For girth g, we need no cycles shorter than g.
        # BFS from each vertex up to depth (g-1)//2.
        # If we encounter a back-edge creating a cycle of length < g, fail.
        half_depth = (required_girth - 1) // 2
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != i and dist[v] <= dist[u]:
                        # Cycle found: length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def random_4reg(n):
        """Generate a random 4-regular simple graph using swaps from a circulant."""
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in [1, 2, n-1, n-2]:
                v = (i + j) % n
                if i != v and v not in adj[i]:
                    adj[i].add(v)
                    adj[v].add(i)
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        return adj

    def swap_edges(adj, n, rng):
        """Perform a random 2-swap on the edge set."""
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        if len(edges) < 2:
            return False
        
        e1 = edges[rng.randrange(len(edges))]
        e2 = edges[rng.randrange(len(edges))]
        if e1 == e2:
            return False
        
        u, v = e1
        x, y = e2
        
        # Try swap: (u,v) and (x,y) -> (u,x) and (v,y)
        adj[u].discard(v)
        adj[v].discard(u)
        adj[x].discard(y)
        adj[y].discard(x)
        
        success = False
        if x not in adj[u] and y not in adj[v] and u != x and v != y:
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            success = True
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            adj[x].add(y)
            adj[y].add(x)
            # Try other swap: (u,v) and (x,y) -> (u,y) and (v,x)
            adj[u].discard(v)
            adj[v].discard(u)
            adj[x].discard(y)
            adj[y].discard(x)
            if y not in adj[u] and x not in adj[v] and u != y and v != x:
                adj[u].add(y)
                adj[y].add(u)
                adj[v].add(x)
                adj[x].add(v)
                success = True
            else:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                adj[x].add(y)
                adj[y].add(x)
        
        return success

    def count_short_cycles(adj, n, max_len):
        """Count cycles of length < max_len using BFS."""
        count = 0
        half_depth = (max_len - 1) // 2
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != i and dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < max_len:
                            count += 1
                            break
                if count > 0:
                    return count
        return count

    # Start with the seed construction
    best_n, best_edges = seed_construct()
    best_adj = build_adj(best_n, best_edges)
    best_girth_ok = verify_girth(best_adj, best_n, g)
    
    # Try to find a smaller graph
    # Moore bound for k=4, g=10 is 242
    moore_bound = 2 * sum((k - 1)**i for i in range(g // 2))
    
    rng = random.Random(42)
    
    # Try sizes from just above Moore bound up to best_n
    # Focus on sizes where improvement is likely
    target_sizes = []
    for n_try in range(moore_bound, best_n + 1, 1):
        if n_try % 2 == 0:  # 4-regular graphs need even number of vertices
            target_sizes.append(n_try)
    
    # Sort by closeness to Moore bound (smallest first)
    target_sizes.sort()
    
    # Limit the number of sizes we try to stay within time budget
    # Each size gets a few seconds
    time_per_size = 5
    max_sizes = int(120 / time_per_size)
    target_sizes = target_sizes[:max_sizes]
    
    for n_try in target_sizes:
        if time.time() > DEADLINE:
            break
        
        # Generate random 4-regular graph
        adj = random_4reg(n_try)
        if adj is None:
            continue
        
        # Check if it already has good girth
        if verify_girth(adj, n_try, g):
            # Convert to edge list
            edges = []
            for i in range(n_try):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            best_n = n_try
            best_edges = edges
            best_adj = adj
            best_girth_ok = True
            break
        
        # Local search: try to increase girth by reducing short cycles
        # Use simulated annealing-like approach
        t_local = time.time()
        local_deadline = t_local + time_per_size
        
        current_adj = [set(s) for s in adj]
        
        # Count initial short cycles
        short_count = count_short_cycles(current_adj, n_try, g)
        
        # If already zero short cycles, we're done (but verify_girth should have caught this)
        
        best_local_short = short_count
        best_local_adj = [set(s) for s in current_adj]
        
        for iteration in range(500):
            if time.time() > local_deadline or time.time() > DEADLINE:
                break
            
            # Pick random edge pair for 2-swap
            all_edges = []
            for i in range(n_try):
                for j in current_adj[i]:
                    if i < j:
                        all_edges.append((i, j))
            
            if len(all_edges) < 2:
                break
            
            e1 = all_edges[rng.randrange(len(all_edges))]
            e2 = all_edges[rng.randrange(len(all_edges))]
            if e1 == e2:
                continue
            
            u, v = e1
            x, y = e2
            
            # Attempt swap
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            current_adj[x].discard(y)
            current_adj[y].discard(x)
            
            swapped = False
            if x not in current_adj[u] and y not in current_adj[v] and u != x and v != y:
                current_adj[u].add(x)
                current_adj[x].add(u)
                current_adj[v].add(y)
                current_adj[y].add(v)
                swapped = True
            else:
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_adj[x].add(y)
                current_adj[y].add(x)
                if y not in current_adj[u] and x not in current_adj[v] and u != y and v != x:
                    current_adj[u].add(y)
                    current_adj[y].add(u)
                    current_adj[v].add(x)
                    current_adj[x].add(v)
                    swapped = True
                else:
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_adj[x].add(y)
                    current_adj[y].add(x)
            
            if not swapped:
                continue
            
            # Check if degrees are still 4
            valid = all(len(current_adj[i]) == 4 for i in range(n_try))
            if not valid:
                # Revert
                current_adj[u].discard(x if swapped else v)
                # This is getting complicated; just regenerate
                current_adj = [set(s) for s in adj]
                break
            
            # Count short cycles (approximate - just check if we improved)
            new_short = count_short_cycles(current_adj, n_try, g)
            
            if new_short <= best_local_short:
                best_local_short = new_short
                best_local_adj = [set(s) for s in current_adj]
            else:
                # Revert
                current_adj = [set(s) for s in best_local_adj]
            
            if best_local_short == 0:
                break
        
        # Check if local search found a valid graph
        if best_local_short == 0:
            if verify_girth(best_local_adj, n_try, g):
                edges = []
                for i in range(n_try):
                    for j in best_local_adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n_try
                best_edges = edges
                best_adj = best_local_adj
                best_girth_ok = True
                break
    
    # If we still haven't improved, try a few more random restarts with the seed size
    if not best_girth_ok:
        # The seed construction should be valid, but let's double-check
        best_girth_ok = verify_girth(best_adj, best_n, g)
    
    return best_n, best_edges
