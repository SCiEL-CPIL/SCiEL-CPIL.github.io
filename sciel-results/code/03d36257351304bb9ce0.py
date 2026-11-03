import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Helper: check if graph is valid (simple, k-regular, girth >= g)
    # Returns (is_valid, girth_found)
    def check_validity(adj, n):
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False, 0
            if len(set(adj[i])) != k:
                return False, 0
        # Check no self-loops or multi-edges
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u == v:
                    return False, 0
                if (min(u, v), max(u, v)) in edge_set:
                    return False, 0
                edge_set.add((min(u, v), max(u, v)))
        
        # Girth check via BFS from each vertex up to depth g//2
        # If we find a cycle of length < g, return False
        for u in range(n):
            depth = [-1] * n
            parent = [-1] * n
            queue = deque([u])
            depth[u] = 0
            while queue:
                v = queue.popleft()
                if depth[v] >= g // 2:
                    continue
                for w in adj[v]:
                    if depth[w] == -1:
                        depth[w] = depth[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v]:
                        cycle_len = depth[v] + depth[w] + 1
                        if cycle_len < g:
                            return False, cycle_len
        return True, g

    # LUW construction for fallback
    def get_luw():
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

    luw_n, luw_edges = get_luw()
    best_n = luw_n
    best_edges = luw_edges

    # Try to find smaller graphs using random regular graph + local search
    # Strategy: Generate random k-regular graphs and check girth
    # For k=5, g=9, try n from 450 to min(1100, luw_n - 1)
    
    def generate_random_regular(n, k, seed=None):
        """Generate a random k-regular graph using configuration model."""
        if seed is not None:
            random.seed(seed)
        
        # Create stubs
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        
        # Pair up stubs
        edges = set()
        adj = [set() for _ in range(n)]
        i = 0
        attempts = 0
        while i < len(stubs) and attempts < 1000:
            u, v = stubs[i], stubs[i + 1]
            if u != v and v not in adj[u]:
                edges.add((min(u, v), max(u, v)))
                adj[u].add(v)
                adj[v].add(u)
                i += 2
            else:
                # Swap with a later stub
                j = random.randint(i + 2, len(stubs) - 1)
                stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                attempts += 1
        
        # Check if we got a valid k-regular graph
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return adj, list(edges)

    def local_search(adj, n, k, g, time_limit):
        """Improve girth by swapping edges."""
        limit_time = time.time() + time_limit
        adj = [list(s) for s in adj]
        
        # Check current girth
        valid, girth = check_validity(adj, n)
        if valid:
            return adj, True
        
        # Try edge swaps to increase girth
        for _ in range(10000):
            if time.time() > limit_time:
                break
            
            # Pick two random edges
            u1 = random.randint(0, n - 1)
            v1 = random.choice(adj[u1])
            u2 = random.randint(0, n - 1)
            v2 = random.choice(adj[u2])
            
            # Ensure we're swapping different edges
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            
            # Remove edges (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
            # Try both options
            for swap_type in range(2):
                new_edges = set()
                # Check if swap is valid (no loops, no multi-edges)
                if swap_type == 0:
                    new_pairs = [(u1, u2), (v1, v2)]
                else:
                    new_pairs = [(u1, v2), (v1, u2)]
                
                valid_swap = True
                for a, b in new_pairs:
                    if a == b:
                        valid_swap = False
                        break
                    if b in adj[a]:
                        valid_swap = False
                        break
                
                if not valid_swap:
                    continue
                
                # Perform swap
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                for a, b in new_pairs:
                    adj[a].append(b)
                    adj[b].append(a)
                
                # Check if girth improved
                valid, girth = check_validity(adj, n)
                if valid:
                    return adj, True
                
                # Undo swap
                for a, b in new_pairs:
                    adj[a].remove(b)
                    adj[b].remove(a)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
        
        return adj, False

    # Try different n values
    # For k=5, g=9, lower bound is 428
    # Try n from 450 to 1100
    if k == 5 and g == 9:
        n_values = list(range(450, 1100, 50))
        n_values.reverse()  # Start with larger n (easier to find valid graphs)
        
        for n in n_values:
            if time.time() > deadline - 10:
                break
            if n >= best_n:
                continue
            
            # Try multiple random seeds
            for seed in range(20):
                if time.time() > deadline - 10:
                    break
                
                result = generate_random_regular(n, k, seed)
                if result is None:
                    continue
                
                adj, edges = result
                valid, girth = check_validity(adj, n)
                
                if valid:
                    best_n = n
                    best_edges = edges
                    break
                else:
                    # Try local search
                    adj, valid = local_search(adj, n, k, g, time_limit=2.0)
                    if valid:
                        # Reconstruct edges from adj
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_n = n
                        best_edges = edges
                        break
    
    # Final check: ensure best_edges is valid
    if best_n < luw_n:
        # Verify the found graph
        adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        valid, _ = check_validity(adj, best_n)
        if not valid:
            # Fall back to LUW
            best_n = luw_n
            best_edges = luw_edges
    
    return best_n, best_edges
