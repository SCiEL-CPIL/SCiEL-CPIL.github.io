import time
import random
import math
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # If k=4 and g=10, try to find the 320-vertex cage via search
    if k == 4 and g == 10:
        deadline = time.time() + 115  # Safety margin
        
        def random_4_regular(n):
            # Configuration model
            if n % 2 != 0:
                return None
            
            def random_perfect_matching(n):
                if n % 2 != 0: return None
                items = list(range(n))
                random.shuffle(items)
                match = {}
                for i in range(0, n, 2):
                    u, v = items[i], items[i+1]
                    match[u] = v
                    match[v] = u
                return match

            m1 = random_perfect_matching(n)
            if m1 is None: return None
            m2 = random_perfect_matching(n)
            if m2 is None: return None
            
            adj = [[] for _ in range(n)]
            for u in range(n):
                v1 = m1[u]
                v2 = m2[u]
                if v1 == v2:
                    return None # Collision, retry
                adj[u].append(v1)
                adj[v1].append(u)
                adj[u].append(v2)
                adj[v2].append(u)
                
            # Check for multi-edges
            for u in range(n):
                if len(set(adj[u])) < 4:
                    return None
            return adj

        def get_girth_fast(adj, max_check=None):
            """Compute girth with early termination. Returns (girth, found_cycle)"""
            n = len(adj)
            best = float('inf')
            if max_check is not None:
                best = max_check
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] * 2 >= best:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif dist[v] < dist[u]:
                            # v is ancestor of u in BFS tree
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < best:
                                best = cycle_len
                                # Reconstruct cycle
                                path = []
                                cur = u
                                while cur != v:
                                    path.append(cur)
                                    cur = parent[cur]
                                path.append(v)
                                return best, path
            return best if best != float('inf') else 0, None

        # Start search
        n_target = 320
        adj = None
        for _ in range(200): # Initial random tries
            adj_try = random_4_regular(n_target)
            if adj_try:
                adj = adj_try
                break
        
        if not adj:
            # Fallback to LUW
            return luw_construct(k, g)

        current_girth, _ = get_girth_fast(adj)
        
        while time.time() < deadline:
            if current_girth >= g:
                # Found valid graph
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
            
            # Find a short cycle to break
            _, cycle = get_girth_fast(adj, current_girth)
            if not cycle:
                # Re-evaluate girth, might be stale or error
                current_girth, _ = get_girth_fast(adj)
                continue
            
            # Try to break the cycle by swapping edges
            # Cycle nodes: c0-c1-...-ck-c0
            # Pick an edge in the cycle and try to reroute it
            cycle_len = len(cycle)
            if cycle_len > g:
                continue
                
            # For each edge in the cycle, try to find an alternative
            swapped = False
            for i in range(cycle_len):
                u = cycle[i]
                v = cycle[(i + 1) % cycle_len]
                
                # Try to find a different neighbor for u and v
                # that doesn't create a shorter cycle
                for new_u in adj[u]:
                    if new_u == v:
                        continue
                    for new_v in adj[v]:
                        if new_v == u or new_v == new_u:
                            continue
                        
                        # Check if this swap would create a shorter cycle
                        # Simple heuristic: just try it
                        # Remove u-v, add u-new_u, v-new_v, and connect new_u-new_v? No.
                        # Actually, we need to maintain 4-regularity.
                        # Swap: remove (u,v), add (u,new_v) and (v,new_u) if new_u != new_v
                        if new_u == new_v:
                            continue
                            
                        # Check for multi-edges
                        if new_v in adj[u] or new_u in adj[v]:
                            continue
                        
                        # Perform swap
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[u].append(new_v)
                        adj[new_v].append(u)
                        adj[v].append(new_u)
                        adj[new_u].append(v)
                        
                        new_girth, _ = get_girth_fast(adj, current_girth)
                        if new_girth >= current_girth:
                            current_girth = new_girth
                            swapped = True
                            break
                        else:
                            # Revert
                            adj[u].remove(new_v)
                            adj[new_v].remove(u)
                            adj[v].remove(new_u)
                            adj[new_u].remove(v)
                            adj[u].append(v)
                            adj[v].append(u)
                        
                        if swapped:
                            break
                
                if swapped:
                    break
            
            # If no swap worked, do a random 2-opt swap
            if not swapped:
                a = random.randint(0, n_target - 1)
                b = random.choice(adj[a])
                c = random.randint(0, n_target - 1)
                d = random.choice(adj[c])
                
                if b == a or c == a or b == c or d == a or b == d or c == d:
                    continue
                if a == c or a == d or b == c or b == d:
                    continue
                
                # Check for multi-edges
                if c in adj[a] or d in adj[b]:
                    continue
                
                # Swap
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                
                adj[a].append(c)
                adj[c].append(a)
                adj[b].append(d)
                adj[d].append(b)
                
                new_girth, _ = get_girth_fast(adj, current_girth)
                if new_girth >= current_girth:
                    current_girth = new_girth
                else:
                    # Revert
                    adj[a].remove(c)
                    adj[c].remove(a)
                    adj[b].remove(d)
                    adj[d].remove(b)
                    
                    adj[a].append(b)
                    adj[b].append(a)
                    adj[c].append(d)
                    adj[d].append(c)

        # If search fails, return LUW
        return luw_construct(k, g)

    else:
        return luw_construct(k, g)
