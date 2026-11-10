import random
import time
import sys

def construct(k, g):
    # Target: 3-regular graph with girth >= g
    # k is expected to be 3
    if k != 3:
        # Fallback for other k if needed, but task specifies k=3
        # For now, return a simple cycle or error
        # But the contract says evaluator passes k=3
        pass
        
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Lower bound for n(3, 16) is 512
    # We will try to construct graphs starting from n=512 upwards
    
    def generate_tree_with_degrees(n, degrees):
        """
        Generates a random tree with n vertices and given degrees.
        degrees: list of length n, sum must be 2*(n-1).
        Returns: list of edges (u, v)
        """
        if sum(degrees) != 2 * (n - 1):
            return None
        
        # Use a union-find to ensure we form a tree
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Create a list of vertices with available slots
        # We will randomly pick two vertices with slots > 0 and connect them
        # To ensure it's a tree, we must connect two different components.
        # However, the simple "pick two slots" method might create cycles if not careful.
        # A safer way:
        # 1. Start with n isolated vertices.
        # 2. We need to add n-1 edges.
        # 3. We maintain a list of vertices with remaining degree > 0.
        # 4. Randomly pick two vertices u, v from this list.
        # 5. If they are in different components, add edge (u,v), reduce degrees.
        # 6. If they are in the same component, retry (or pick different).
        # This can be slow if components merge early.
        
        # Alternative: Prüfer Code based generation is hard to control degrees exactly.
        # Let's use the "Random Pairing" with Union-Find, but with a twist:
        # We only pick pairs that are in different components.
        
        slots = degrees[:]
        active = [i for i in range(n) if slots[i] > 0]
        edges = []
        
        # To speed up, we can group vertices by component?
        # For n=512, n^2 is 260k. n-1 is 511.
        # We just loop until we have n-1 edges.
        
        for _ in range(n - 1):
            if not active:
                return None # Should not happen if sum is correct
            
            # Pick two random active vertices
            # To avoid bias, we can pick from a weighted list?
            # Simple random pick from active list.
            u_idx = random.randint(0, len(active) - 1)
            v_idx = random.randint(0, len(active) - 1)
            
            # Ensure u != v
            if u_idx == v_idx and len(active) == 1:
                return None # Only one vertex left, but we need to connect? No, if only one active, it must have slot 0 or we are done.
            
            u = active[u_idx]
            v = active[v_idx]
            
            if u == v:
                continue
            
            if union(u, v):
                edges.append((u, v))
                slots[u] -= 1
                slots[v] -= 1
                
                # Remove from active if slot becomes 0
                if slots[u] == 0:
                    active.remove(u)
                if slots[v] == 0:
                    active.remove(v)
            else:
                # Same component, retry
                # To prevent infinite loop, we might need to restart or be smarter
                # But for n=512, this should work.
                # If it fails too often, we restart the whole tree generation.
                pass
        
        return edges

    def check_girth(edges, n, g):
        # BFS from each vertex
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            # BFS up to depth g//2 - 1
            # If we find a neighbor that is already visited at a smaller depth, we have a cycle.
            # Actually, standard BFS girth check:
            # dist[v] = shortest distance from start to v.
            # If we encounter an edge (u,v) such that dist[u] + 1 == dist[v] and parent[u] != v, etc.
            # Simpler: 
            # dist = [-1]*n
            # parent = [-1]*n
            # queue = deque([(start, 0)])
            # dist[start] = 0
            # while queue:
            #   u, d = queue.popleft()
            #   if d >= g//2: continue
            #   for v in adj[u]:
            #     if dist[v] == -1:
            #       dist[v] = d+1
            #       parent[v] = u
            #       queue.append((v, d+1))
            #     elif v != parent[u] and dist[v] <= d:
            #       # Found a cycle
            #       # Cycle length = dist[u] + dist[v] + 1
            #       # If this length < g, return False
            #       pass
            # This is O(n * g * degree).
            # For n=512, g=16, this is fast.
            
            from collections import deque
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            dist[start] = 0
            
            while queue:
                u, d = queue.popleft()
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        queue.append((v, d + 1))
                    elif v != parent[u]:
                        # Found a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # But we need to be careful: this is the shortest cycle passing through u-v?
                        # Actually, if we find a back edge, the cycle length is dist[u] + dist[v] + 1.
                        # If this is < g, we fail.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            # If we finish BFS without finding short cycle, this start is OK.
            # But we need to check for ALL starts?
            # Yes, a cycle might not be found from a specific start if it's "far"?
            # No, BFS from any vertex in the cycle will find it.
            # Actually, if we BFS from *every* vertex, we are sure.
            
        return True

    def try_construct(n):
        # Degrees:
        # We want a tree with n vertices.
        # Let L be number of leaves (deg 1), I be internal (deg 3), R be root (deg 2).
        # L + I + R = n
        # L + 3I + 2R = 2(n-1)
        # From first: I = n - L - R
        # L + 3(n - L - R) + 2R = 2n - 2
        # L + 3n - 3L - 3R + 2R = 2n - 2
        # 3n - 2L - R = 2n - 2
        # n + 2 = 2L + R
        # We need integer solutions for L, R >= 0.
        # Let R = 1 (one root with deg 2).
        # n + 1 = 2L => L = (n+1)/2.
        # So n must be odd for R=1.
        # If n is even, let R=2?
        # n + 2 = 2L + 2 => 2L = n => L = n/2.
        # So if n is even, we can have 2 roots (deg 2) and n/2 leaves.
        # If n is odd, we can have 1 root (deg 2) and (n+1)/2 leaves.
        
        if n % 2 == 0:
            L = n // 2
            R = 2
            I = n - L - R
            degrees = [1] * L + [3] * I + [2] * R
        else:
            L = (n + 1) // 2
            R = 1
            I = n - L - R
            degrees = [1] * L + [3] * I + [2] * R
            
        random.shuffle(degrees)
        
        # Generate tree
        tree_edges = generate_tree_with_degrees(n, degrees)
        if not tree_edges:
            return None
            
        # Identify free ends (vertices with degree < 3 in the tree)
        # In our construction, leaves have deg 1, roots have deg 2.
        # We need to add edges to make them deg 3.
        # Deficit of leaves: 2. Deficit of roots: 1.
        # Wait, in the tree, leaf deg is 1. We need deg 3. Deficit 2.
        # Root deg is 2. We need deg 3. Deficit 1.
        # Total deficit = 2*L + 1*R.
        # For even n: 2*(n/2) + 2*1 = n+2.
        # Number of new edges = (n+2)/2.
        # We need to pair up the "deficit units".
        # Each leaf contributes 2 units. Each root contributes 1 unit.
        # We can model this as matching "half-edges".
        # Total half-edges = n+2.
        # We need to pair them up.
        
        # List of (vertex, slot_index)
        # Slot 0 and 1 for leaves. Slot 0 for roots.
        half_edges = []
        for i in range(n):
            if degrees[i] == 1:
                half_edges.append((i, 0))
                half_edges.append((i, 1))
            elif degrees[i] == 2:
                half_edges.append((i, 0))
                
        # Compute distances between all vertices in the tree
        # We need distance between any two vertices that might be connected.
        # We will compute all-pairs shortest paths? n=512, Floyd-Warshall O(n^3) = 134M. Too slow in Python.
        # BFS from each vertex: n * (n + E) = 512 * 1024 ~ 500k. Fast.
        
        dist_matrix = [[-1]*n for _ in range(n)]
        for start in range(n):
            # BFS
            dist = [-1]*n
            dist[start] = 0
            from collections import deque
            q = deque([start])
            while q:
                u = q.popleft()
                for v in tree_adj[u]: # Need adj list
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            dist_matrix[start] = dist
        
        # Greedy Matching
        # We want to pair half_edges such that distance >= g-1
        # To maximize success, we can sort half_edges by "difficulty" (degree of freedom)?
        # Or just random order.
        
        random.shuffle(half_edges)
        used = set()
        new_edges = []
        
        for i in range(len(half_edges)):
            if i in used:
                continue
            u_he = half_edges[i]
            u = u_he[0]
            
            # Find best partner
            best_j = -1
            best_dist = -1
            for j in range(i+1, len(half_edges)):
                if j in used:
                    continue
                v_he = half_edges[j]
                v = v_he[0]
                if u == v:
                    continue # No loops
                d = dist_matrix[u][v]
                if d >= g - 1 and d > best_dist:
                    best_dist = d
                    best_j = j
            
            if best_j == -1:
                return None # Failed to match
            
            used.add(i)
            used.add(best_j)
            v = half_edges[best_j][0]
            new_edges.append((u, v))
            
        # Combine edges
        all_edges = tree_edges + new_edges
        
        # Verify
        if check_girth(all_edges, n, g):
            return n, all_edges
        else:
            return None

    # Build adjacency list for tree for distance calc
    # We need to do this inside try_construct or pass it.
    # Let's refactor try_construct to handle adj.
    
    def try_construct_refactored(n):
        if n % 2 == 0:
            L = n // 2
            R = 2
            I = n - L - R
            degrees = [1] * L + [3] * I + [2] * R
        else:
            L = (n + 1) // 2
            R = 1
            I = n - L - R
            degrees = [1] * L + [3] * I + [2] * R
            
        random.shuffle(degrees)
        
        tree_edges = generate_tree_with_degrees(n, degrees)
        if not tree_edges:
            return None
            
        # Build adj
        tree_adj = [[] for _ in range(n)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            
        # Compute distances
        dist_matrix = [[-1]*n for _ in range(n)]
        from collections import deque
        for start in range(n):
            dist = [-1]*n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in tree_adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            dist_matrix[start] = dist
            
        half_edges = []
        for i in range(n):
            if degrees[i] == 1:
                half_edges.append((i, 0))
                half_edges.append((i, 1))
            elif degrees[i] == 2:
                half_edges.append((i, 0))
                
        random.shuffle(half_edges)
        used = set()
        new_edges = []
        
        for i in range(len(half_edges)):
            if i in used:
                continue
            u = half_edges[i][0]
            best_j = -1
            best_dist = -1
            for j in range(i+1, len(half_edges)):
                if j in used:
                    continue
                v = half_edges[j][0]
                if u == v:
                    continue
                d = dist_matrix[u][v]
                if d >= g - 1 and d > best_dist:
                    best_dist = d
                    best_j = j
            
            if best_j == -1:
                return None
            
            used.add(i)
            used.add(best_j)
            v = half_edges[best_j][0]
            new_edges.append((u, v))
            
        all_edges = tree_edges + new_edges
        
        if check_girth(all_edges, n, g):
            return n, all_edges
        else:
            return None

    # Main loop
    best_result = None
    # Try n from 512 to 900
    # We will try a few iterations for each n
    
    for n in range(512, 950, 4):
        if time.time() > deadline:
            break
            
        # Try up to 50 times for this n
        for attempt in range(50):
            if time.time() > deadline:
                break
            res = try_construct_refactored(n)
            if res:
                best_result = res
                # If we found one, we can try to find smaller?
                # No, we are iterating n upwards. So the first one found is the smallest.
                # But maybe a smaller n failed due to bad luck.
                # We should continue to smaller n? No, we started at 512.
                # So if we find one at n=512, that's the best possible (Moore bound).
                # If we find one at n=516, we can return it.
                return res
                
    # If we failed to find any graph in the range, fallback to L-U-W or similar
    # But L-U-W might be slow or large.
    # Let's return a known valid graph if we can construct one quickly.
    # For now, return the best found, or None.
    # But we MUST return a valid graph.
    # If we didn't find one, we have a problem.
    # We can try a larger n with more attempts?
    
    if best_result is None:
        # Fallback: Use the L-U-W logic from the seed?
        # Or just return a simple cycle? No, must be 3-regular.
        # Let's try n=1000 with more attempts.
        for n in range(950, 1200, 10):
            if time.time() > deadline:
                break
            for attempt in range(20):
                if time.time() > deadline:
                    break
                res = try_construct_refactored(n)
                if res:
                    return res
                    
    # If still nothing, return the seed program's result?
    # We can't easily call the seed program's logic without copying it.
    # But we can copy the L-U-W code here as a last resort.
    
    # L-U-W Fallback
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

