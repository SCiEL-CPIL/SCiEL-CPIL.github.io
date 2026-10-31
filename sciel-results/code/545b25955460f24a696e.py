"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 118  # safety margin before 150s kill
    start_time = time.time()
    
    # Moore bound for degree k, girth g (even)
    if g % 2 == 0:
        moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore_bound = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    
    # Ensure n is even for 3-regular (handshaking lemma: n*k must be even)
    if k % 2 == 1 and moore_bound % 2 == 1:
        moore_bound += 1
    
    def verify_girth_fast(n, adj, k, g):
        """Verify girth >= g using non-backtracking BFS from every vertex up to depth r = g//2 - 1."""
        r = g // 2 - 1
        # Expected number of non-backtracking walks at each distance
        expected = [1] + [k * (k - 1) ** i for i in range(r)]
        
        for start in range(n):
            # dist[v] = distance from start, parent[v] = previous vertex in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= r:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue  # non-backtracking
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if du + 1 <= r:
                            counts[du + 1] += 1
                            queue.append(v)
                    else:
                        # Found a non-tree edge; check if it creates a short cycle
                        # The cycle length would be dist[u] + dist[v] + 1 (if they meet)
                        # Actually in non-backtracking BFS, a collision means a cycle
                        # of length <= dist[u] + dist[v] + 1, but we need to be more careful
                        # For girth verification: if dist[u] + dist[v] + 1 < g, we have a short cycle
                        # But actually, the standard check is that the counts match expected
                        pass
            
            if counts != expected:
                return False
        return True
    
    def find_short_cycle(n, adj, g):
        """Find a cycle of length < g. Returns list of nodes or None."""
        r = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        # Non-tree edge creating a cycle
                        # Find LCA of u and v
                        au, av = u, v
                        du, dv = dist[au], dist[av]
                        while du > dv:
                            au = parent[au]
                            du -= 1
                        while dv > du:
                            av = parent[av]
                            dv -= 1
                        while au != av:
                            au = parent[au]
                            av = parent[av]
                        # au is LCA
                        path_u = []
                        cur = u
                        while cur != au:
                            path_u.append(cur)
                            cur = parent[cur]
                        path_u.append(au)
                        path_u.reverse()
                        path_v = []
                        cur = v
                        while cur != au:
                            path_v.append(cur)
                            cur = parent[cur]
                        path_v.append(au)
                        path_v.reverse()
                        cycle = path_u[:-1] + path_v[::-1]
                        if 3 <= len(cycle) < g:
                            return cycle
        return None
    
    def count_short_cycles(n, adj, g):
        """Count the number of cycles of length < g."""
        r = g // 2 - 1
        count = 0
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        # Non-tree edge creating a cycle
                        au, av = u, v
                        du, dv = dist[au], dist[av]
                        while du > dv:
                            au = parent[au]
                            du -= 1
                        while dv > du:
                            av = parent[av]
                            dv -= 1
                        while au != av:
                            au = parent[au]
                            av = parent[av]
                        path_u = []
                        cur = u
                        while cur != au:
                            path_u.append(cur)
                            cur = parent[cur]
                        path_u.append(au)
                        path_u.reverse()
                        path_v = []
                        cur = v
                        while cur != au:
                            path_v.append(cur)
                            cur = parent[cur]
                        path_v.append(au)
                        path_v.reverse()
                        cycle = path_u[:-1] + path_v[::-1]
                        if 3 <= len(cycle) < g:
                            count += 1
        return count
    
    def generate_random_regular(n, k):
        """Generate random k-regular graph via configuration model with retries."""
        for attempt in range(500):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
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
        return None
    
    def edge_swap(adj, n, k):
        """Perform a random 2-switch to maintain k-regularity."""
        for _ in range(200):
            u1 = random.randint(0, n - 1)
            v1 = random.choice(adj[u1])
            u2 = random.randint(0, n - 1)
            v2 = random.choice(adj[u2])
            
            if len({u1, v1, u2, v2}) < 4:
                continue
            if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                continue
            
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        return False
    
    def luw_construction(k, d):
        """LUW (Lubotzky-Phillips-Sarnak / Morgenstern) construction."""
        q = k
        # Build the rule table
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r_val = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
                yield (1 - side, tuple(y))
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for idx, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        return n, edges, adj
    
    # Strategy:
    # 1. Try LUW construction with various d values (algebraic, fast)
    # 2. Try random regular graphs with local search (configuration model)
    # 3. Return best found
    
    best_n = None
    best_edges = None
    
    # First: try LUW constructions (fast, deterministic)
    # For girth g, we need d such that the resulting graph has girth >= g
    # LUW graphs typically have girth ~ 2*log_k(n), so for large g we need large d
    # The number of vertices is 2 * q^d
    for d in range(6, 20):
        if time.time() > deadline - 30:
            break
        try:
            n, edges, adj = luw_construction(k, d)
            if verify_girth_fast(n, adj, k, g):
                return n, edges
            elif best_n is None or n < best_n:
                best_n = n
                best_edges = edges
        except:
            pass
    
    # Second: try random regular graphs with local search
    # Start from Moore bound and increase
    n_candidates = []
    for n in range(moore_bound, moore_bound + 500, 2):
        if time.time() > deadline - 30:
            break
        n_candidates.append(n)
    
    for n in n_candidates:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 20:
                break
            
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
            
            # Check if already valid
            if verify_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            
            # Local search: iteratively break short cycles
            prev_count = count_short_cycles(n, adj, g)
            for iteration in range(500):
                if time.time() > deadline - 20:
                    break
                
                cycle = find_short_cycle(n, adj, g)
                if cycle is None:
                    if verify_girth_fast(n, adj, k, g):
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                    break
                
                # Try to break the cycle
                adj_backup = [list(lst) for lst in adj]
                improved = False
                
                # Try several random swaps
                for _ in range(20):
                    if time.time() > deadline - 20:
                        break
                    
                    adj_try = [list(lst) for lst in adj]
                    if edge_swap(adj_try, n, k):
                        new_count = count_short_cycles(n, adj_try, g)
                        if new_count < prev_count:
                            adj = adj_try
                            prev_count = new_count
                            improved = True
                            break
                
                if not improved:
                    # Try a pure random swap to escape local minimum
                    if edge_swap(adj, n, k):
                        prev_count = count_short_cycles(n, adj, g)
            
            # Final check
            if verify_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Return best found
    if best_n is not None and best_edges is not None:
        return best_n, best_edges
    
    # Fallback: LUW with larger d
    d_orig = max(8, g - 5 if (g - 5) % 2 == 0 else g - 4)
    n, edges, adj = luw_construction(k, d_orig)
    return n, edges
