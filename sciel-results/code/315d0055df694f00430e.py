import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 145  # Safety margin before 150s limit

    def luw_construct(k, g):
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

    def check_girth_fast(adj, n, g):
        """Check if graph has girth >= g using BFS from each vertex up to depth (g-1)//2."""
        max_depth = (g - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle; compute its length
                        # Walk back from u and v to find LCA
                        du = dist[u]
                        dv = dist[v]
                        uu, vv = u, v
                        while uu != vv:
                            if du > dv:
                                uu = parent[uu]
                                du -= 1
                            elif dv > du:
                                vv = parent[vv]
                                dv -= 1
                            else:
                                uu = parent[uu]
                                vv = parent[vv]
                                du -= 1
                                dv -= 1
                        cycle_len = dist[u] + dist[v] - 2 * du + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_random_regular_graph(n, k):
        """Generate a random k-regular graph using two random permutations (for k=4)."""
        if k != 4:
            return None
        for attempt in range(50):
            p1 = list(range(n))
            random.shuffle(p1)
            p2 = list(range(n))
            random.shuffle(p2)
            
            adj = {i: [] for i in range(n)}
            valid = True
            for i in range(n):
                u = p1[i]
                v1 = p1[(i + 1) % n]
                v2 = p1[(i - 1) % n]
                w1 = p2[i]
                w2 = p2[(i + 1) % n]
                w3 = p2[(i - 1) % n]
                
                # Edge from p1 cycle
                if v1 not in adj[u]:
                    adj[u].append(v1)
                    adj[v1].append(u)
                if v2 not in adj[u]:
                    adj[u].append(v2)
                    adj[v2].append(u)
                
                # Edge from p2 cycle
                if w2 not in adj[u]:
                    adj[u].append(w2)
                    adj[w2].append(u)
                if w3 not in adj[u]:
                    adj[u].append(w3)
                    adj[w3].append(u)
            
            # Validate
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
                if len(set(adj[i])) != k:
                    valid = False
                    break
                for j in adj[i]:
                    if i not in adj[j]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check for multi-edges
            for i in range(n):
                for j in adj[i]:
                    if adj[j].count(i) > 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            return adj
        return None

    def local_search(k, g, n_target, time_limit=130):
        start_time = time.time()
        adj = generate_random_regular_graph(n_target, k)
        if adj is None:
            return None, None
        
        if check_girth_fast(adj, n_target, g):
            edges = [(i, j) for i in range(n_target) for j in adj[i] if i < j]
            return n_target, edges
        
        # 2-switch local search
        for step in range(50000):
            if time.time() - start_time > time_limit:
                break
            
            # Pick two random edges and swap
            u = random.randint(0, n_target - 1)
            v = random.choice(adj[u])
            w = random.randint(0, n_target - 1)
            x = random.choice(adj[w])
            
            if u == w or u == x or v == w or v == x:
                continue
            if w in adj[u] or x in adj[v]:
                continue
            if u in adj[w] or v in adj[x]:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            if check_girth_fast(adj, n_target, g):
                edges = [(i, j) for i in range(n_target) for j in adj[i] if i < j]
                return n_target, edges
            
            # Undo swap (always undo for now; could use simulated annealing later)
            adj[u].remove(w)
            adj[w].remove(u)
            adj[v].remove(x)
            adj[x].remove(v)
            adj[u].append(v)
            adj[v].append(u)
            adj[w].append(x)
            adj[x].append(w)
        
        return None, None

    # Try local search first
    n_ls, edges_ls = local_search(k, g, 320, 130)
    if n_ls is not None:
        return n_ls, edges_ls
    
    # Fallback to LUW
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
