import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Helper: Check Girth
    def check_girth(adjacency, n, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper: Verify Graph
    def verify_graph(n, edges, k, g):
        if n < 1:
            return False
        adjacency = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u >= n or v >= n:
                return False
            if (min(u, v), max(u, v)) in edge_set:
                return False
            edge_set.add((min(u, v), max(u, v)))
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        for i in range(n):
            if len(adjacency[i]) != k:
                return False
        
        if not check_girth(adjacency, n, g):
            return False
            
        return True

    # Petersen Graph (3,5)
    def petersen():
        n = 10
        edges = []
        # Outer pentagon
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner star
        for i in range(5):
            edges.append((5 + i, 5 + (i + 2) % 5))
        # Spokes
        for i in range(5):
            edges.append((i, 5 + i))
        return n, edges

    # LUW Construction
    def luw_construct(k, d):
        q = k
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

    # Strategy
    if k == 3 and g <= 5:
        n, edges = petersen()
        if verify_graph(n, edges, k, g):
            return n, edges

    # Edge-switching algorithm for k=5, g=9
    def random_regular_graph(n, k, seed=42):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        
        rng = random.Random(seed)
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        
        # Pair stubs
        edges = []
        edge_set = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None  # Self-loop, reject
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None  # Multi-edge, reject
            edge_set.add(key)
            edges.append((u, v))
        
        # Verify degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return None
        
        return n, edges

    def check_girth_fast(adjacency, n, g):
        """Check girth using BFS from each vertex, early exit. Returns (valid, shortest_cycle)."""
        shortest = g  # Start with target girth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < shortest:
                                shortest = cycle_len
                                if shortest < g:
                                    return False, shortest
        return True, shortest

    def find_short_cycles(adjacency, n, g, max_cycles=50):
        """Find up to max_cycles short cycles (length < g)."""
        cycles = []
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue and len(cycles) < max_cycles:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                cycle = [u]
                                cur = u
                                while cur != v:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                cycles.append(cycle)
                                if len(cycles) >= max_cycles:
                                    break
        return cycles

    def edge_switching_improve(n, edges, k, g, max_time=30.0):
        """
        Improve girth via 2-switches (edge swaps).
        A 2-switch: if (a,b) and (c,d) are edges, replace with (a,c) and (b,d)
        if neither new edge exists and no self-loops are created.
        Accept if it doesn't create shorter cycles.
        """
        deadline = time.time() + max_time
        
        # Build adjacency and edge set
        adjacency = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
        
        # Initial girth check
        valid, shortest = check_girth_fast(adjacency, n, g)
        if valid:
            return edges, True
        
        # Try to improve
        best_edges = edges[:]
        best_shortest = shortest
        
        rng = random.Random(12345)
        
        # Precompute all edges for quick lookup
        edge_list = [(u, v) for u, v in edge_set]
        
        for iteration in range(100000):
            if time.time() > deadline:
                break
            
            # Pick two random edges
            if len(edge_list) < 2:
                break
            idx1 = rng.randrange(len(edge_list))
            idx2 = rng.randrange(len(edge_list))
            if idx1 == idx2:
                continue
            
            e1 = edge_list[idx1]
            e2 = edge_list[idx2]
            
            # Skip if edges share a vertex (would create self-loop or multi-edge)
            if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
                continue
            
            # Try swap: (a,b) and (c,d) -> (a,c) and (b,d)
            a, b = e1
            c, d = e2
            
            # Check no self-loops
            if a == c or a == d or b == c or b == d:
                continue
            
            # Check no multi-edges
            new_e1 = (min(a, c), max(a, c))
            new_e2 = (min(b, d), max(b, d))
            
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
            
            # Perform the swap
            # Remove old edges
            adjacency[a].remove(c) if c in adjacency[a] else None
            adjacency[a].remove(d) if d in adjacency[a] else None
            adjacency[b].remove(c) if c in adjacency[b] else None
            adjacency[b].remove(d) if d in adjacency[b] else None
            
            # Actually, let's do this more carefully
            # Remove e1 = (a,b) and e2 = (c,d)
            adjacency[a].remove(b)
            adjacency[b].remove(a)
            adjacency[c].remove(d)
            adjacency[d].remove(c)
            edge_set.remove(e1)
            edge_set.remove(e2)
            
            # Add new edges (a,c) and (b,d)
            adjacency[a].append(c)
            adjacency[c].append(a)
            adjacency[b].append(d)
            adjacency[d].append(b)
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            # Update edge_list
            edge_list[idx1] = new_e1
            edge_list[idx2] = new_e2
            
            # Check girth
            valid, new_shortest = check_girth_fast(adjacency, n, g)
            
            if new_shortest >= best_shortest:
                # Accept the switch
                best_shortest = new_shortest
                if valid:
                    # Reconstruct edges from edge_set
                    final_edges = list(edge_set)
                    return final_edges, True
            else:
                # Reject: undo the swap
                adjacency[a].remove(c)
                adjacency[c].remove(a)
                adjacency[b].remove(d)
                adjacency[d].remove(b)
                edge_set.remove(new_e1)
                edge_set.remove(new_e2)
                
                adjacency[a].append(b)
                adjacency[b].append(a)
                adjacency[c].append(d)
                adjacency[d].append(c)
                edge_set.add(e1)
                edge_set.add(e2)
                
                edge_list[idx1] = e1
                edge_list[idx2] = e2
        
        # Return best found
        final_edges = list(edge_set)
        valid, _ = check_girth_fast(adjacency, n, g)
        return final_edges, valid

    # Try randomized search for small graphs with edge switching
    if k == 5 and g == 9:
        # Moore bound for (5,9): 428
        moore_bound = 428
        
        best_n = None
        best_edges = None
        
        # Try increasing n from Moore bound upward
        for n_try in range(moore_bound, moore_bound + 100):
            if time.time() > deadline:
                break
            
            # Try a few random seeds
            for seed in range(30):
                if time.time() > deadline:
                    break
                
                result = random_regular_graph(n_try, k, seed=seed + n_try * 1000)
                if result is None:
                    continue
                
                n_val, edges_val = result
                
                # First quick girth check
                adjacency = [[] for _ in range(n_val)]
                for u, v in edges_val:
                    adjacency[u].append(v)
                    adjacency[v].append(u)
                
                valid, shortest = check_girth_fast(adjacency, n_val, g)
                if valid:
                    best_n = n_val
                    best_edges = edges_val
                    break
                
                # If not valid, try edge switching to improve
                if shortest >= 6:  # Only worth trying if girth is already decent
                    switch_deadline = min(deadline, time.time() + 5.0)
                    improved_edges, improved_valid = edge_switching_improve(
                        n_val, edges_val, k, g, max_time=min(5.0, deadline - time.time())
                    )
                    if improved_valid:
                        best_n = n_val
                        best_edges = improved_edges
                        break
            
            if best_n is not None:
                break
        
        if best_n is not None:
            return best_n, best_edges

    # Calculate minimal depth for LUW
    # Girth of LUW(k, d) is 2d + 2
    # We need 2d + 2 >= g
    # d >= (g - 2) / 2
    d = (g - 1) // 2
    
    # Ensure d is at least 2 for LUW to be non-trivial
    if d < 2:
        d = 2

    luw_n, luw_edges = luw_construct(k, d)
    
    if verify_graph(luw_n, luw_edges, k, g):
        return luw_n, luw_edges
        
    # Fallback: try larger d
    for d_try in range(d + 1, d + 5):
        if time.time() > deadline:
            break
        luw_n, luw_edges = luw_construct(k, d_try)
        if verify_graph(luw_n, luw_edges, k, g):
            return luw_n, luw_edges

    return 0, []
