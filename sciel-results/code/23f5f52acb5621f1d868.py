import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    if k != 3:
        # Fallback for other k, though task is k=3
        return seed_fallback(k, g)

    # For k=3, g=16, the Moore bound is 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254.
    # However, the smallest known is 936.
    # We will attempt to construct a graph of size 936 with girth >= 16.
    # If we fail, we fall back to the LUW construction (size 1280).

    n_target = 936

    def generate_random_3_regular(n):
        # Configuration model with careful pairing to avoid multi-edges and self-loops
        # We use a "stub" list and a set of used pairs to ensure simplicity.
        
        # A more robust method: start with a random permutation and fix conflicts,
        # or use the "switching" method to generate a simple graph.
        # For n=936, the configuration model often produces multi-edges.
        
        # Let's use a method that ensures simplicity:
        # 1. Create stubs.
        # 2. Shuffle.
        # 3. Pair them.
        # 4. If multi-edge or self-loop, find a conflicting pair and swap.
        
        # Actually, a simpler approach for moderate n:
        # Try to generate a simple 3-regular graph using a heuristic.
        # Or just retry the configuration model until it's simple.
        # Probability of simplicity for n=936 is decent.
        
        for _ in range(100):
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            
            adj = [set() for _ in range(n)]
            valid = True
            
            # Pair stubs
            for i in range(0, 3 * n, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if not valid:
                continue
            
            return adj
        return None

    def has_short_cycle(adj, n, girth_limit):
        # Returns True if there is a cycle of length < girth_limit
        # Use BFS from each node, limited depth
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= girth_limit // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    elif v != adj[u].pop() if False else True: # Placeholder, logic below
                        pass
            
            # Proper BFS check for back-edges
            # We need to track parent to avoid going back
            # Let's rewrite this function properly.
            pass
        
        # Correct implementation:
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= girth_limit // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_limit:
                            return True
        return False

    def find_short_cycle(adj, n, girth_limit):
        # Returns the cycle (list of vertices) or None
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= girth_limit // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_limit:
                            # Reconstruct cycle
                            path_u = [u]
                            while path_u[-1] != s:
                                path_u.append(parent[path_u[-1]])
                            path_u.reverse()
                            
                            path_v = [v]
                            while path_v[-1] != s:
                                path_v.append(parent[path_v[-1]])
                            path_v.reverse()
                            
                            # Combine paths
                            # s ... u and s ... v
                            # Cycle is u ... s ... v
                            # The vertices are path_u + path_v[1:]
                            # But we need to be careful with the order.
                            # path_u is s, ..., u
                            # path_v is s, ..., v
                            # The cycle is u, ..., s, ..., v, u? No.
                            # The edge (u,v) closes the cycle.
                            # The path from u to v via s is path_u[::-1] + path_v[1:]
                            cycle = path_u[::-1] + path_v[1:]
                            return cycle
        return None

    def local_search(adj, n, target_girth, time_limit):
        # Iterative 2-switches to increase girth
        current_adj = [set(lst) for lst in adj]
        
        # Check initial state
        if not has_short_cycle(current_adj, n, target_girth):
            return current_adj

        max_steps = 20000
        for step in range(max_steps):
            if time.time() > time_limit:
                return None
            
            # Find a short cycle
            cycle = find_short_cycle(current_adj, n, target_girth)
            if not cycle:
                return current_adj
            
            # Pick a random edge from the cycle
            # Cycle is a list of vertices [v0, v1, ..., v_{L-1}] where (v_{L-1}, v0) is an edge
            L = len(cycle)
            edge_idx = random.randint(0, L - 1)
            u1 = cycle[edge_idx]
            v1 = cycle[(edge_idx + 1) % L]
            
            # Find a random edge (u2, v2) not incident to u1 or v1
            # And not the same edge
            all_edges = []
            for i in range(n):
                for j in current_adj[i]:
                    if i < j:
                        all_edges.append((i, j))
            
            valid_edges = []
            for u2, v2 in all_edges:
                if u2 == u1 or u2 == v1 or v2 == u1 or v2 == v1:
                    continue
                valid_edges.append((u2, v2))
            
            if not valid_edges:
                continue
                
            u2, v2 = random.choice(valid_edges)
            
            # Check for multi-edges after switch
            # New edges: (u1, u2) and (v1, v2)
            if u2 in current_adj[u1] or v2 in current_adj[v1]:
                continue
            # Also check if (u1, v2) or (v1, u2) exist? No, those would be different edges.
            # We only care if the *new* edges are already present.
            
            # Perform 2-switch
            current_adj[u1].remove(v1)
            current_adj[v1].remove(u1)
            current_adj[u2].remove(v2)
            current_adj[v2].remove(u2)
            
            current_adj[u1].add(u2)
            current_adj[u2].add(u1)
            current_adj[v1].add(v2)
            current_adj[v2].add(v1)
            
            # Check if girth improved or stayed same
            # We accept if no *shorter* cycle is created.
            # Since we broke a cycle, the new girth is likely >= old girth.
            # But we might create a new short cycle elsewhere? Unlikely with 2-switch.
            # Let's just check if the specific broken cycle is gone and no new short cycle exists.
            # Actually, just checking if has_short_cycle is False is expensive.
            # Let's assume the move is good if it doesn't create a self-loop/multi-edge.
            # We will verify at the end.
            
        # Final verification
        if not has_short_cycle(current_adj, n, target_girth):
            return current_adj
        return None

    # Main logic
    if time.time() < deadline - 10:
        adj = generate_random_3_regular(n_target)
        if adj:
            res_adj = local_search(adj, n_target, g, deadline - 5)
            if res_adj:
                edges = []
                for i in range(n_target):
                    for j in res_adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_target, edges

    # Fallback: LUW
    return seed_fallback(k, g)

def seed_fallback(k, g):
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
