import random
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for k=3, g=16 (even)
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # g=16, r=8. sum = 1+2+4+8+16+32+64+128 = 255. n >= 510.
    # n must be even.
    
    for n in range(512, 1000, 2):
        if time.time() > deadline - 5:
            break
            
        # Try a few restarts for each n
        for restart in range(5):
            if time.time() > deadline - 2:
                break
                
            # Generate random 3-regular graph
            adj = [set() for _ in range(n)]
            # Configuration model
            half_edges = n * k // 2
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
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
                
            # Check girth
            girth = calc_girth(adj)
            if girth >= g:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            # Local Search
            current_girth = girth
            no_improve_count = 0
            max_iters = 5000
            iter_limit_time = time.time() + 10 # 10s per restart
            
            for _ in range(max_iters):
                if time.time() > iter_limit_time:
                    break
                if time.time() > deadline - 2:
                    break
                    
                # Find shortest cycle
                cycle = find_shortest_cycle(adj, current_girth)
                if not cycle:
                    # Should not happen if girth calc is correct
                    break
                if len(cycle) >= g:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
                # Pick a random pair of non-adjacent edges in the cycle
                L = len(cycle)
                if L < 4:
                    break
                # Cycle nodes: cycle[0], cycle[1], ..., cycle[L-1]
                # Edges: (cycle[i], cycle[(i+1)%L])
                
                # Try to swap
                swapped = False
                # Pick random start index
                start = random.randint(0, L-1)
                # Pick gap d (distance between edges)
                # Edges are (u,v) and (w,z).
                # u=cycle[start], v=cycle[(start+1)%L]
                # We need another edge (w,z) not adjacent to (u,v).
                # Adjacent edges share a vertex.
                # Non-adjacent means w,z not in {u,v} and no shared vertices.
                
                possible_swaps = []
                for d in range(2, L//2 + 1):
                    # Edge 1: cycle[start] - cycle[(start+1)%L]
                    u = cycle[start]
                    v = cycle[(start+1)%L]
                    
                    # Edge 2: cycle[(start+d)%L] - cycle[(start+d+1)%L]
                    w = cycle[(start+d)%L]
                    z = cycle[(start+d+1)%L]
                    
                    # Check if (u,w) and (v,z) are valid new edges
                    if w not in adj[u] and z not in adj[v]:
                        # Also check (u,z) and (v,w) for the other pairing
                        pass
                        
                    # Try pairing 1: (u,w) and (v,z)
                    if w not in adj[u] and z not in adj[v]:
                        possible_swaps.append((u, v, w, z, 1))
                    # Try pairing 2: (u,z) and (v,w)
                    if z not in adj[u] and w not in adj[v]:
                        possible_swaps.append((u, v, w, z, 2))

                if not possible_swaps:
                    no_improve_count += 1
                    if no_improve_count > 100:
                        break
                    continue
                    
                u, v, w, z, pairing = random.choice(possible_swaps)
                
                # Apply swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(z)
                adj[z].remove(w)
                
                if pairing == 1:
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(z)
                    adj[z].add(v)
                else:
                    adj[u].add(z)
                    adj[z].add(u)
                    adj[v].add(w)
                    adj[w].add(v)
                    
                # Check girth
                new_girth = calc_girth(adj)
                if new_girth > current_girth:
                    current_girth = new_girth
                    no_improve_count = 0
                elif new_girth == current_girth:
                    no_improve_count += 1
                    if no_improve_count > 200:
                        # Revert to avoid getting stuck?
                        # Just keep going, random restarts help
                        pass
                else:
                    # Revert
                    if pairing == 1:
                        adj[u].remove(w)
                        adj[w].remove(u)
                        adj[v].remove(z)
                        adj[z].remove(v)
                    else:
                        adj[u].remove(z)
                        adj[z].remove(u)
                        adj[v].remove(w)
                        adj[w].remove(v)
                        
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[w].add(z)
                    adj[z].add(w)
                    
                    no_improve_count += 1
                    if no_improve_count > 500:
                        break

    # Fallback to LUW if local search failed
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

def calc_girth(adj):
    n = len(adj)
    min_girth = float('inf')
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        q = deque()
        q.append(start)
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    return min_girth

def find_shortest_cycle(adj, current_girth):
    n = len(adj)
    best_cycle = None
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        q = deque()
        q.append(start)
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < current_girth:
                        # Reconstruct cycle
                        path1 = []
                        cur = u
                        while cur != -1:
                            path1.append(cur)
                            cur = parent[cur]
                        path2 = []
                        cur = v
                        while cur != -1:
                            path2.append(cur)
                            cur = parent[cur]
                        
                        # Find LCA
                        # Simple method: set of path1
                        set1 = set(path1)
                        lca = -1
                        for node in path2:
                            if node in set1:
                                lca = node
                                break
                                
                        if lca != -1:
                            cycle = []
                            cur = u
                            while cur != lca:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.append(lca)
                            cur = v
                            while cur != lca:
                                cycle.append(cur)
                                cur = parent[cur]
                            
                            if best_cycle is None or len(cycle) < len(best_cycle):
                                best_cycle = cycle
                                if len(best_cycle) == 4: # Min possible for simple 3-reg
                                    return best_cycle
    return best_cycle
