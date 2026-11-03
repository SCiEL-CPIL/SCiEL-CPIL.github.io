import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Target n = 320 (Known best)
        n_target = 320
        
        # 1. Try Cayley graph on Z_320 (Fast, unlikely to succeed for cage)
        result = _try_cayley_z320(n_target, deadline)
        if result is not None:
            return result
            
        # 2. Try Cayley graph on Z_16 x Z_20
        result = _try_cayley_16_20(deadline)
        if result is not None:
            return result

        # 3. Optimized Local Search for n=320
        result = _local_search_4_reg(n_target, g, deadline)
        if result is not None:
            return result

    # General fallback: LUW Construction
    return _luw_construct(k, g)

def _try_cayley_z320(n, deadline):
    def check_girth(b):
        visited = [-1] * n
        visited[0] = 0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            depth = visited[u]
            if depth >= 4:
                pass
            for g in [1, n-1, b, (n-b)%n]:
                v = (u + g) % n
                if visited[v] == -1:
                    visited[v] = depth + 1
                    queue.append(v)
                else:
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < 10:
                        return False
        return True

    for b in range(2, n//2 + 1):
        if time.time() > deadline:
            return None
        if b in [1, n-1, n//2]: continue
        if (n-b) % n == b: continue
        if check_girth(b):
            edges = []
            for u in range(n):
                for g in [1, b]:
                    v = (u + g) % n
                    if u < v:
                        edges.append((u, v))
            return n, edges
    return None

def _try_cayley_16_20(deadline):
    m, n = 16, 20
    total = m * n
    
    def check_girth(x, y):
        visited = [-1] * total
        visited[0] = 0
        queue = deque([0])
        gens = [(1, 0), (m-1, 0), (x, y), ((-x)%m, (-y)%n)]
        while queue:
            u = queue.popleft()
            depth = visited[u]
            ui, uj = divmod(u, n)
            for g1, g2 in gens:
                nu = (ui + g1) % m
                nv = (uj + g2) % n
                v = nu * n + nv
                if visited[v] == -1:
                    visited[v] = depth + 1
                    queue.append(v)
                else:
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < 10:
                        return False
        return True

    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            if (x == 0 and y == 0) or (x == 1 and y == 0) or (x == m-1 and y == 0):
                continue
            if (2*x) % m == 0 and (2*y) % n == 0:
                continue
            if check_girth(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        uid = u * n + v
                        nu = (u + 1) % m
                        vid = nu * n + v
                        if uid < vid: edges.append((uid, vid))
                        nu = (u + x) % m
                        nv = (v + y) % n
                        vid = nu * n + nv
                        if uid < vid: edges.append((uid, vid))
                return total, edges
    return None

def _local_search_4_reg(n, g, deadline):
    # 1. Generate random 4-regular graph
    def generate_random_4reg(n):
        stubs = [i for i in range(n) for _ in range(4)]
        random.shuffle(stubs)
        edges = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if (u, v) in edges or (v, u) in edges:
                return None
            edges.add((min(u,v), max(u,v)))
        return edges

    # 2. Fast Girth Check (Boolean: is girth >= g?)
    def check_girth_fast(edges_set, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length < g
        # We use BFS from each vertex, but we stop if we find a short cycle.
        # Optimization: Only check up to depth g//2 - 1?
        # If a cycle of length L < g exists, then from any vertex on it,
        # the "other side" is at distance < L/2.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= g // 2: 
                    # If depth is large, any cycle formed by backtracking to a 
                    # node at similar depth would be long.
                    # But we must be careful.
                    pass
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        # If v is parent of u, it's just the tree edge, ignore.
                        if v == parent[u]:
                            continue
                        # Calculate cycle length
                        # We need LCA of u and v.
                        # Since we only care about L < g, and du, dv are small,
                        # we can trace back.
                        # But tracing back is O(N).
                        # Alternative: 
                        # If dist[u] + dist[v] + 1 < g, it's a candidate.
                        # But we need exact length.
                        # For speed, if dist[u] + dist[v] + 1 >= g, 
                        # the cycle is definitely >= g?
                        # No, LCA could be deep.
                        # But if dist[u] + dist[v] + 1 < g, it's a short cycle candidate.
                        # If dist[u] + dist[v] + 1 >= g, it might still be short if LCA is deep?
                        # No, LCA depth <= min(dist[u], dist[v]).
                        # Cycle len = dist[u] + dist[v] + 1 - 2*dist[lca].
                        # Max len = dist[u] + dist[v] + 1 (if lca is root).
                        # Min len = ...
                        # If dist[u] + dist[v] + 1 >= g, it's possible the cycle is < g?
                        # Yes, if lca is close to u/v.
                        # Example: dist[u]=4, dist[v]=4, g=10. Sum=9.
                        # If lca is at depth 3, len = 4+4+1-6 = 3.
                        # So we MUST check.
                        
                        # To speed up, we only check if dist[u] + dist[v] + 1 < 2*g?
                        # Actually, if dist[u] + dist[v] + 1 >= g, 
                        # and we know that for a cycle of length < g, 
                        # the sum of distances from start to the two ends of the chord 
                        # must be < g?
                        # No.
                        
                        # Let's just do the LCA trace. It's short.
                        # Trace u up to root, store path.
                        # Trace v up to root, find intersection.
                        path_u = set()
                        cur = u
                        while cur != -1:
                            path_u.add(cur)
                            cur = parent[cur]
                        
                        cur = v
                        dv = dist[v]
                        # We need dist of LCA.
                        # Walk v up until we hit path_u.
                        while cur not in path_u:
                            cur = parent[cur]
                            if cur == -1: break # Should not happen in connected comp
                        lca = cur
                        dlca = dist[lca]
                        
                        cycle_len = dist[u] + dist[v] + 1 - 2 * dlca
                        if cycle_len < g:
                            return False
        return True

    # 3. Search Loop
    best_edges = None
    best_girth = 0 # We don't track exact girth, just boolean success
    
    # Try a few random restarts
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        edges = generate_random_4reg(n)
        if not edges:
            continue
        
        # Initial check
        if check_girth_fast(edges, n, g):
            return n, list(edges)
            
        # Local Search
        stuck_count = 0
        while time.time() < deadline:
            if stuck_count > 200:
                break # Restart
            
            # Find a short cycle to break
            # We can reuse the BFS logic to find a specific short cycle.
            # Or just pick a random edge and swap?
            # Picking random edges is slow.
            # Let's find a short cycle.
            
            short_cycle_edges = None
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Find any cycle < g
            found = False
            for start in range(n):
                if time.time() > deadline: break
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue and not found:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if v == parent[u]: continue
                            # Check cycle len
                            # Simplified: if dist[u]+dist[v]+1 < g, assume short?
                            # No, must check.
                            # For speed in search, let's just pick a random edge 
                            # and swap, and check girth afterwards.
                            # This is "random hill climbing".
                            pass
                # If we found a short cycle, we'd break here.
                # But implementing the finder is complex.
                # Let's just do random swaps.
                break # Exit start loop, just do random swaps
            
            # Perform a random swap
            edge_list = list(edges)
            e1 = random.choice(edge_list)
            e2 = random.choice(edge_list)
            if e1 == e2: continue
            
            u1, v1 = e1
            u2, v2 = e2
            
            new_e1 = (min(u1, u2), max(u1, u2))
            new_e2 = (min(v1, v2), max(v1, v2))
            
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                continue
            if new_e1 in edges or new_e2 in edges:
                continue
            if new_e1 == new_e2:
                continue
            
            edges.remove(e1)
            edges.remove(e2)
            edges.add(new_e1)
            edges.add(new_e2)
            
            # Check girth
            if check_girth_fast(edges, n, g):
                return n, list(edges)
            
            stuck_count += 1
            
    return None

def _luw_construct(k, g):
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
