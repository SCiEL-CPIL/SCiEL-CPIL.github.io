import time
import random
import math
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # Determine d based on g
        # For g=10, d=5?
        # The original code uses d = max(3, g-5 if (g-5)%2 else g-4)
        # For g=10, g-5=5 (odd) -> d=5.
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

    if k == 4 and g == 10:
        deadline = time.time() + 115
        n_target = 320
        
        def random_4_regular(n):
            if n % 2 != 0:
                return None
            def random_perfect_matching(n):
                items = list(range(n))
                random.shuffle(items)
                match = {}
                for i in range(0, n, 2):
                    u, v = items[i], items[i+1]
                    match[u] = v
                    match[v] = u
                return match

            m1 = random_perfect_matching(n)
            m2 = random_perfect_matching(n)
            adj = [[] for _ in range(n)]
            for u in range(n):
                v1 = m1[u]
                v2 = m2[u]
                if v1 == v2:
                    return None
                adj[u].append(v1)
                adj[v1].append(u)
                adj[u].append(v2)
                adj[v2].append(u)
            
            for u in range(n):
                if len(set(adj[u])) < 4:
                    return None
            return adj

        # Optimized Girth Checker
        def get_girth(adj, best_known=11):
            """
            Returns the girth. If girth >= best_known, returns best_known (or higher).
            best_known is used for early termination.
            """
            n = len(adj)
            best = best_known
            # BFS from each node
            # We only need to check up to depth (best-1)//2
            max_depth = (best - 1) // 2
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                found = False
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du > max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            q.append(v)
                        elif dist[v] < du:
                            # v is ancestor
                            cycle_len = du + dist[v] + 1
                            if cycle_len < best:
                                best = cycle_len
                                max_depth = (best - 1) // 2
                                # If best becomes 4, we can stop everything? 
                                # No, we just update max_depth.
                                # If best is 4, max_depth is 1.
                                if best <= 4:
                                    return best
            return best

        def find_short_cycle(adj, target_len):
            """Find a cycle of length <= target_len. Returns list of nodes or None."""
            n = len(adj)
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du * 2 >= target_len:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            q.append(v)
                        elif dist[v] < du:
                            cycle_len = du + dist[v] + 1
                            if cycle_len <= target_len:
                                # Reconstruct
                                path = []
                                cur = u
                                while cur != v:
                                    path.append(cur)
                                    cur = parent[cur]
                                path.append(v)
                                return path
            return None

        # Initialize
        adj = None
        for _ in range(50):
            adj_try = random_4_regular(n_target)
            if adj_try:
                adj = adj_try
                break
        
        if not adj:
            return luw_construct(k, g)

        current_girth = get_girth(adj, 11)
        best_girth = current_girth
        best_adj = [lst[:] for lst in adj]

        # Search Loop
        # We use a simple hill climbing with random restarts if stuck
        stuck_counter = 0
        T = 1.0 # Temperature for simulated annealing (if we implement it)
        
        while time.time() < deadline:
            if current_girth >= g:
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges

            # Find a short cycle to guide the move
            short_cycle = find_short_cycle(adj, min(current_girth, 9))
            
            if not short_cycle:
                # Girth might have increased, re-evaluate
                current_girth = get_girth(adj, best_girth)
                if current_girth > best_girth:
                    best_girth = current_girth
                    best_adj = [lst[:] for lst in adj]
                continue

            # Generate candidate moves
            # Try to break the short cycle
            candidates = []
            
            # 1. Cycle breaking swaps
            for i in range(len(short_cycle)):
                u = short_cycle[i]
                v = short_cycle[(i + 1) % len(short_cycle)]
                # Try to swap (u,v) with (x,y)
                # x is neighbor of u (not v)
                # y is neighbor of v (not u)
                # And x,y are not connected?
                for x in adj[u]:
                    if x == v: continue
                    for y in adj[v]:
                        if y == u or y == x: continue
                        # Check if x-y is an edge? If so, swapping (u,v) and (x,y) 
                        # is a valid 2-opt only if we remove (x,y) too.
                        # But we only want to remove (u,v).
                        # A 2-opt swap removes (u,v) and (x,y) and adds (u,x) and (v,y).
                        # This requires (x,y) to be an edge.
                        if y in adj[x]:
                            # Valid 2-opt
                            # Check for multi-edges in new graph
                            # u-x new edge: u and x are not adjacent (x is neighbor of u? No, x is neighbor of u in original, so u-x is an edge? 
                            # Wait. x is in adj[u]. So (u,x) is already an edge.
                            # If we add (u,x) again, it's a multi-edge.
                            # So x must NOT be a neighbor of u.
                            # But I picked x from adj[u]. So this logic is flawed.
                            # We need x to be a neighbor of u, and y to be a neighbor of v.
                            # And we remove (u,v) and (x,y).
                            # And we add (u,x) and (v,y).
                            # For (u,x) to be a new edge, x must not be a neighbor of u.
                            # But x IS a neighbor of u.
                            # So this swap creates a multi-edge.
                            # Correct 2-opt:
                            # Remove (u,v) and (w,z).
                            # Add (u,w) and (v,z).
                            # u,w must not be adjacent.
                            # v,z must not be adjacent.
                            # w,z must be adjacent (to remove).
                            # So w is neighbor of u? No.
                            # Let's pick two edges (u,v) and (w,z).
                            # u,w not adjacent. v,z not adjacent.
                            # Remove (u,v), (w,z). Add (u,w), (v,z).
                            
                            pass # This path is complex to validate quickly.
            
            # Simpler Move: Random 2-Opt
            # Pick random edge (u,v). Pick random edge (w,z).
            # If u,w not adj and v,z not adj:
            # Remove (u,v), (w,z). Add (u,w), (v,z).
            
            moved = False
            for _ in range(10): # Try 10 random swaps
                u = random.randint(0, n_target - 1)
                v = random.choice(adj[u])
                w = random.randint(0, n_target - 1)
                z = random.choice(adj[w])
                
                if u == w or u == z or v == w or v == z:
                    continue
                if w in adj[u] or z in adj[v]:
                    continue
                # Also check z in adj[u] and w in adj[v] for the other new edges?
                # New edges: (u,w) and (v,z).
                # We checked w in adj[u] and z in adj[v].
                # We also need to ensure we don't create multi-edges with the *other* new edge?
                # No, (u,w) and (v,z) are the only new ones.
                # But we must ensure (u,w) is not already an edge (checked).
                # And (v,z) is not already an edge (checked).
                
                # Perform Swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(z)
                adj[z].remove(w)
                
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(z)
                adj[z].append(v)
                
                new_girth = get_girth(adj, current_girth)
                
                if new_girth > current_girth:
                    current_girth = new_girth
                    if current_girth > best_girth:
                        best_girth = current_girth
                        best_adj = [lst[:] for lst in adj]
                    moved = True
                    break
                elif new_girth == current_girth:
                    # Accept with prob 0.5 to explore
                    if random.random() < 0.5:
                        moved = True
                        break
                    else:
                        # Revert
                        adj[u].remove(w)
                        adj[w].remove(u)
                        adj[v].remove(z)
                        adj[z].remove(v)
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[w].append(z)
                        adj[z].append(w)
                else:
                    # Revert
                    adj[u].remove(w)
                    adj[w].remove(u)
                    adj[v].remove(z)
                    adj[z].remove(v)
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[w].append(z)
                    adj[z].append(w)
            
            if not moved:
                stuck_counter += 1
                if stuck_counter > 50:
                    # Restart from best
                    adj = [lst[:] for lst in best_adj]
                    current_girth = best_girth
                    stuck_counter = 0

        # Return best found
        if best_girth >= g:
            edges = []
            for u in range(n_target):
                for v in best_adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_target, edges
        else:
            return luw_construct(k, g)
    else:
        return luw_construct(k, g)
