import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Optimized Girth Check
        max_depth = g // 2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
                
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        # Ignore tree edges (parent-child)
                        if v == parent[curr]:
                            continue
                        # If v is a child (dist[v] == dist[curr] + 1) and not parent, 
                        # it was discovered by another node in the same level.
                        # This forms a cycle of length dist[curr] + 1 + dist[v].
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        # Original LUW construction
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def generate_random_k_regular(n, k):
        """
        Generate a random k-regular graph on n vertices using the configuration model.
        Returns adjacency list or None if failed.
        """
        if n * k % 2 != 0:
            return None
        
        # Create stubs: each vertex has k stubs
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        
        # Try multiple random permutations
        for attempt in range(200):
            random.shuffle(stubs)
            
            # Pair up stubs to form edges
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                # Verify all vertices have degree exactly k
                for v in range(n):
                    if len(adj[v]) != k:
                        valid = False
                        break
                if valid:
                    return adj
        
        return None

    def find_short_cycles(adj, n, g):
        """
        Find all cycles shorter than g in the graph.
        Returns a list of cycles, where each cycle is a list of vertices.
        """
        max_depth = g // 2
        short_cycles = []
        
        for u in range(n):
            # BFS from u
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        # Ignore tree edges (parent-child)
                        if v == parent[curr]:
                            continue
                        # Check for cycle
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            # Reconstruct cycle
                            # Path from u to curr
                            path1 = []
                            node = curr
                            while node != -1:
                                path1.append(node)
                                node = parent[node]
                            path1.reverse()
                            
                            # Path from u to v
                            path2 = []
                            node = v
                            while node != -1:
                                path2.append(node)
                                node = parent[node]
                            path2.reverse()
                            
                            # Cycle is path1 + path2[1:] (excluding u which is in both)
                            cycle = path1 + path2[1:]
                            if len(cycle) == cycle_len:
                                short_cycles.append(cycle)
        
        return short_cycles

    def apply_2switch(adj, n, k, e1, e2):
        """
        Apply a 2-switch: remove edges e1=(u1,v1) and e2=(u2,v2), add edges (u1,u2) and (v1,v2)
        or (u1,v2) and (v1,u2). Returns True if successful.
        """
        u1, v1 = e1
        u2, v2 = e2
        
        # Check if the new edges would create loops or multi-edges
        # Option 1: (u1,u2) and (v1,v2)
        if u1 != u2 and v1 != v2 and u2 not in adj[u1] and v2 not in adj[v1]:
            # Try this option
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            return True
        
        # Option 2: (u1,v2) and (v1,u2)
        if u1 != v2 and v1 != u2 and v2 not in adj[u1] and u2 not in adj[v1]:
            # Try this option
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(v2)
            adj[v2].add(u1)
            adj[v1].add(u2)
            adj[u2].add(v1)
            return True
        
        return False

    def random_walk_construction(n, k, g, time_budget):
        """
        Try to construct a k-regular graph on n vertices with girth >= g
        using random generation and 2-switches.
        """
        deadline_local = time.time() + time_budget
        
        # Generate initial random k-regular graph
        adj = generate_random_k_regular(n, k)
        if adj is None:
            return None
        
        # Iteratively find and eliminate short cycles
        for iteration in range(10000):
            if time.time() > deadline_local:
                break
            
            # Find short cycles
            short_cycles = find_short_cycles(adj, n, g)
            if not short_cycles:
                # No short cycles found - success!
                return adj
            
            # Pick a random short cycle
            cycle = random.choice(short_cycles)
            
            # Try to break the cycle using 2-switches
            # Find two non-adjacent edges in the cycle
            cycle_edges = []
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                cycle_edges.append((u, v))
            
            # Try random pairs of non-adjacent edges
            success = False
            for attempt in range(50):
                if time.time() > deadline_local:
                    break
                
                # Pick two random edges from the cycle
                i1, i2 = random.sample(range(len(cycle_edges)), 2)
                e1, e2 = cycle_edges[i1], cycle_edges[i2]
                
                # Check if edges are non-adjacent in the cycle
                if abs(i1 - i2) == 1 or abs(i1 - i2) == len(cycle_edges) - 1:
                    continue
                
                # Save state for rollback
                adj_copy = [set(s) for s in adj]
                
                if apply_2switch(adj, n, k, e1, e2):
                    # Check if this helped (reduces number of short cycles)
                    new_cycles = find_short_cycles(adj, n, g)
                    if len(new_cycles) < len(short_cycles):
                        success = True
                        break
                    else:
                        # Rollback
                        adj = adj_copy
            
            if not success:
                # Try a completely random 2-switch
                all_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            all_edges.append((u, v))
                
                success = False
                for attempt in range(100):
                    if time.time() > deadline_local:
                        break
                    
                    if len(all_edges) < 2:
                        break
                    
                    e1, e2 = random.sample(all_edges, 2)
                    
                    # Check if edges are non-adjacent
                    if e1[0] in e2 or e1[1] in e2:
                        continue
                    
                    adj_copy = [set(s) for s in adj]
                    
                    if apply_2switch(adj, n, k, e1, e2):
                        new_cycles = find_short_cycles(adj, n, g)
                        if len(new_cycles) < len(short_cycles):
                            success = True
                            break
                        else:
                            adj = adj_copy
                
                if not success:
                    # Stuck - restart with a new random graph
                    adj = generate_random_k_regular(n, k)
                    if adj is None:
                        return None
                    continue
        
        # Final check
        if time.time() < deadline_local:
            final_cycles = find_short_cycles(adj, n, g)
            if not final_cycles:
                return adj
        
        return None

    # 1. Try LUW family with optimized d
    q = k
    
    # Refinement: Use the theoretical lower bound for d to minimize vertices.
    # Girth >= 2d + 2 implies d >= (g - 2) / 2.
    # We use integer division.
    min_d = max(3, (g - 2) // 2)
    
    # Try a range of d values starting from the minimum required
    for d_val in range(min_d, min_d + 15):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            # Skip if graph is too large
            if n > 50000:
                continue
            if n < best_n:
                # Verify before accepting
                t_limit = 5.0 if n > 10000 else 20.0
                if verify_graph(n, edges, k, g, time_limit=t_limit):
                    best_n = n
                    best_edges = edges
                    # Since n increases with d, the first valid graph is the smallest.
                    break
        except Exception:
            pass

    # Fallback: If no valid graph was found via optimized d, try original logic
    if best_edges is None and time.time() < deadline - 5:
        # Original logic used g-5
        orig_min_d = max(3, g - 5)
        for d_val in range(orig_min_d, orig_min_d + 5):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = build_luw_graph(q, d_val)
                if n <= 50000:
                    if verify_graph(n, edges, k, g, time_limit=5.0):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                pass

    # 2. Try probabilistic random-walk construction
    if time.time() < deadline - 10:
        # For k=4, g=10, the cage has 320 vertices
        # Try to construct graphs with n vertices starting from a reasonable lower bound
        # Moore bound for k=4, g=10: 2 * sum_{i=0}^{4} 3^i = 2 * (1+3+9+27+81) = 242
        # So try n from 320 upward
        
        if k == 4 and g == 10:
            target_n = 320
            time_remaining = deadline - time.time() - 5
            if time_remaining > 10:
                adj_result = random_walk_construction(target_n, k, g, time_remaining)
                if adj_result is not None:
                    # Convert adjacency list to edge list
                    edges = []
                    for u in range(target_n):
                        for v in adj_result[u]:
                            if u < v:
                                edges.append((u, v))
                    
                    # Verify
                    if verify_graph(target_n, edges, k, g, time_limit=5.0):
                        if target_n < best_n:
                            best_n = target_n
                            best_edges = edges
        else:
            # For other k, g, try a few sizes
            # Use Moore bound as starting point
            moore_bound = 2 * sum((k-1)**i for i in range(g//2))
            for trial_n in [moore_bound + 2, moore_bound + 10, moore_bound + 50]:
                if time.time() > deadline - 10:
                    break
                if trial_n * k % 2 != 0:
                    continue
                time_remaining = deadline - time.time() - 5
                if time_remaining < 10:
                    break
                adj_result = random_walk_construction(trial_n, k, g, time_remaining)
                if adj_result is not None:
                    edges = []
                    for u in range(trial_n):
                        for v in adj_result[u]:
                            if u < v:
                                edges.append((u, v))
                    
                    if verify_graph(trial_n, edges, k, g, time_limit=5.0):
                        if trial_n < best_n:
                            best_n = trial_n
                            best_edges = edges
                        break

    if best_edges is None:
        return 0, []

    return best_n, best_edges
