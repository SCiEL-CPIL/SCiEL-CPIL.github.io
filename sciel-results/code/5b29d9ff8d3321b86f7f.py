import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    def moore_bound(k, g):
        if g % 2 == 0:
            # even girth
            # n >= 2 * k * (k-1)^(g/2 - 1)
            return 2 * k * ((k-1) ** (g//2 - 1))
        else:
            # odd girth
            # n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
            s = 0
            for i in range((g-3)//2 + 1):
                s += (k-1)**i
            return k * s

    lb = moore_bound(k, g)
    if lb % 2 != 0:
        lb += 1
        
    def generate_random_regular(n, k):
        # Returns adj list and edge set
        # Use configuration model
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
                edges.add((min(u,v), max(u,v)))
            if valid:
                return adj, edges

    def find_shortest_cycle(adj, n):
        # Returns (length, cycle_edges) or (inf, None)
        min_len = float('inf')
        best_cycle = None
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            parent[s] = -1
            q = deque([s])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and parent[v] != -1 and dist[v] < dist[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But we need to ensure we don't count the same cycle multiple times
                        # and that it's the shortest.
                        # Actually, BFS from s finds the shortest cycle containing s?
                        # No, it finds the shortest cycle in the graph if we take min over all s?
                        # Yes.
                        cyc_len = dist[u] + dist[v] + 1
                        if cyc_len < min_len:
                            # Reconstruct cycle
                            # Path s -> ... -> u
                            # Path s -> ... -> v
                            # Edge u-v
                            # Cycle: u -> ... -> s -> ... -> v -> u
                            # We need the list of edges.
                            # Trace back from u to s
                            path_u = []
                            cur = u
                            while cur != s:
                                path_u.append((cur, parent[cur]))
                                cur = parent[cur]
                            path_u.reverse() # Now s -> ... -> u
                            
                            # Trace back from v to s
                            path_v = []
                            cur = v
                            while cur != s:
                                path_v.append((cur, parent[cur]))
                                cur = parent[cur]
                            path_v.reverse() # Now s -> ... -> v
                            
                            # Combine: u -> ... -> s (reverse of path_u) + s -> ... -> v (path_v) + v-u
                            # Wait, path_u is s->...->u. Reverse is u->...->s.
                            # path_v is s->...->v.
                            # Cycle edges: reverse(path_u) + path_v + [(v,u)]?
                            # Let's just store the nodes.
                            nodes_u = []
                            cur = u
                            while cur != s:
                                nodes_u.append(cur)
                                cur = parent[cur]
                            nodes_u.append(s)
                            nodes_u.reverse() # u, ..., s
                            
                            nodes_v = []
                            cur = v
                            while cur != s:
                                nodes_v.append(cur)
                                cur = parent[cur]
                            nodes_v.append(s)
                            nodes_v.reverse() # v, ..., s
                            
                            # Cycle: u -> ... -> s -> ... -> v -> u
                            # Nodes: nodes_u (ends with s) + nodes_v[1:] (starts with s, skip s) + [u]
                            # Wait, nodes_u is u...s. nodes_v is v...s.
                            # We want u...s...v...u.
                            # So nodes_u + nodes_v[1:] (which is ...v) ? No.
                            # nodes_v is v, parent(v), ..., s.
                            # So path s->...->v is reverse(nodes_v).
                            # Let's just return the length and the set of edges in the cycle.
                            # For breaking, we just need two edges.
                            # We can just pick random edges from the union of edges in path_u and path_v and (u,v).
                            
                            # To be efficient, we just need to know that a short cycle exists.
                            # For the swap, we need to pick edges.
                            # Let's store the edges.
                            cyc_edges = set()
                            # Add u-v
                            cyc_edges.add((min(u,v), max(u,v)))
                            # Add path_u edges
                            for i in range(len(nodes_u)-1):
                                a, b = nodes_u[i], nodes_u[i+1]
                                cyc_edges.add((min(a,b), max(a,b)))
                            # Add path_v edges
                            for i in range(len(nodes_v)-1):
                                a, b = nodes_v[i], nodes_v[i+1]
                                cyc_edges.add((min(a,b), max(a,b)))
                            
                            min_len = cyc_len
                            best_cycle = cyc_edges
                            
            if min_len < g: # Early exit if we found something shorter than target
                 # Actually we want the absolute shortest to break it most effectively
                 # But if min_len < g, we can break it.
                 # We should break the loop if we found the global minimum?
                 # BFS from s finds the shortest cycle *containing s*.
                 # The global shortest cycle is the min over all s.
                 # We can stop if min_len == 3 (triangle) etc.
                 pass
                
        return min_len, best_cycle

    def try_find_graph(n, k, g, time_limit):
        start_t = time.time()
        max_restarts = 50
        for _ in range(max_restarts):
            if time.time() - start_t > time_limit:
                return None
            
            adj, edges = generate_random_regular(n, k)
            
            # Local search
            for step in range(10000):
                if time.time() - start_t > time_limit:
                    return None
                
                min_len, cycle_edges = find_shortest_cycle(adj, n)
                
                if min_len >= g:
                    return n, list(edges)
                
                if not cycle_edges:
                    break # Should not happen if min_len < inf
                
                # Pick two edges to swap
                # Ensure they are in the cycle
                edge_list = list(cycle_edges)
                if len(edge_list) < 2:
                    break
                e1, e2 = random.sample(edge_list, 2)
                
                u1, v1 = e1
                u2, v2 = e2
                
                # Try swap 1: (u1, u2) and (v1, v2)
                # Check validity
                # u1 != u2, v1 != v2, etc.
                # Also check if new edges exist in adj
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                
                # Check if (u1, u2) or (v1, v2) exist
                if u2 in adj[u1] or v2 in adj[v1]:
                    # Try other pairing
                    if u2 in adj[v1] or v2 in adj[u1]:
                        continue
                    # Swap: remove (u1,v1), (u2,v2). Add (u1,v2), (v1,u2)
                    # Wait, if (u1,u2) exists, we can't add it.
                    # So we must add (u1,v2) and (v1,u2).
                    # Check if (u1,v2) and (v1,u2) exist
                    if v2 in adj[u1] or u2 in adj[v1]:
                        continue
                    
                    # Apply
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    adj[u1].append(v2)
                    adj[v2].append(u1)
                    adj[v1].append(u2)
                    adj[u2].append(v1)
                    
                    edges.discard(e1)
                    edges.discard(e2)
                    edges.add((min(u1,v2), max(u1,v2)))
                    edges.add((min(v1,u2), max(v1,u2)))
                
                else:
                    # (u1,u2) and (v1,v2) don't exist
                    if u2 in adj[v1] or v2 in adj[u1]:
                        # This case is covered by "Try other pairing" logic?
                        # If (u1,u2) doesn't exist, we can add it.
                        # If (v1,v2) doesn't exist, we can add it.
                        # But we need to remove (u1,v1) and (u2,v2).
                        # So we add (u1,u2) and (v1,v2).
                        # Check if (u1,u2) and (v1,v2) are valid (they are, by check above)
                        # Wait, the check above was `if u2 in adj[u1]`.
                        # So if we are in `else`, `u2 not in adj[u1]` and `v2 not in adj[v1]`.
                        # So we can add them.
                        
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        
                        adj[u1].append(u2)
                        adj[u2].append(u1)
                        adj[v1].append(v2)
                        adj[v2].append(v1)
                        
                        edges.discard(e1)
                        edges.discard(e2)
                        edges.add((min(u1,u2), max(u1,u2)))
                        edges.add((min(v1,v2), max(v1,v2)))

        return None

    best_n = None
    best_edges = None
    
    # Try increasing n
    # Start from lower bound
    # If we find one, we can return it.
    # To be safe, we might want to try a few sizes.
    # But returning the first found (smallest n) is optimal for the score.
    
    n = lb
    while n < 1500:
        if time.time() > deadline:
            break
            
        # Allocate more time for smaller n?
        # Or just fixed time per n.
        # If n is small, search is faster?
        # Let's give 2 seconds per n.
        res = try_find_graph(n, k, g, 2.0)
        if res:
            return res
        
        n += 2
        
    # Fallback: LUW
    # Re-implement LUW from seed
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

    return luw_construct(k, g)
