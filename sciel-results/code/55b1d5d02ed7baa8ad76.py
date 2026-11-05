import random
import time
from collections import deque

def _generate_random_regular_graph(n, k):
    """Generate a random simple k-regular graph on n vertices."""
    for _ in range(1000):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edge_set:
                valid = False
                break
            edge_set.add(edge)
            adj[u].append(v)
            adj[v].append(u)
        if valid:
            return adj, edge_set
    return None, None

def _find_shortest_cycle(adj):
    """Find the length and edges of the shortest cycle."""
    n = len(adj)
    min_len = float('inf')
    min_edges = []
    
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # BFS from u to v avoiding edge (u,v)
                dist = [-1] * n
                parent = [-1] * n
                dist[u] = 0
                q = deque([u])
                found = False
                while q:
                    cur = q.popleft()
                    if dist[cur] >= min_len:
                        continue
                    for nb in adj[cur]:
                        if nb == v and cur == u:
                            continue
                        if dist[nb] == -1:
                            dist[nb] = dist[cur] + 1
                            parent[nb] = cur
                            q.append(nb)
                            if nb == v:
                                found = True
                                break
                    if found:
                        break
                
                if found:
                    cycle_len = dist[v] + 1
                    if cycle_len < min_len:
                        min_len = cycle_len
                        # Reconstruct cycle
                        cycle = [v]
                        cur = v
                        while cur != u:
                            cur = parent[cur]
                            cycle.append(cur)
                        # cycle is v, ..., u
                        # Edges are (u, cycle[-2]), (cycle[-2], cycle[-3]), ..., (cycle[1], v)
                        edges_in_cycle = []
                        for i in range(len(cycle)-1):
                            a, b = cycle[i], cycle[i+1]
                            edges_in_cycle.append((min(a,b), max(a,b)))
                        min_edges = edges_in_cycle
    return min_len, min_edges

def construct(k, g):
    start_time = time.time()
    timeout = 115
    
    # Try increasing n
    for n in range(512, 1000, 2):
        if time.time() - start_time > timeout:
            break
        
        # Try a few random graphs for this n
        for attempt in range(20):
            if time.time() - start_time > timeout:
                break
                
            adj, edge_set = _generate_random_regular_graph(n, k)
            if adj is None:
                continue
            
            current_girth, _ = _find_shortest_cycle(adj)
            
            # Local search
            for step in range(2000):
                if time.time() - start_time > timeout:
                    break
                
                if current_girth >= g:
                    return n, list(edge_set)
                
                _, cycle_edges = _find_shortest_cycle(adj)
                if not cycle_edges:
                    break
                
                # Pick random edge from cycle
                u, v = random.choice(cycle_edges)
                
                # Find candidate edge (x, y) to swap with
                # Must not be incident to u or v
                candidates = [e for e in edge_set if u not in e and v not in e]
                if not candidates:
                    continue
                    
                x, y = random.choice(candidates)
                
                # Try swap 1: (u,v), (x,y) -> (u,x), (v,y)
                # Check validity
                if x != u and y != v:
                    new_e1 = (min(u, x), max(u, x))
                    new_e2 = (min(v, y), max(v, y))
                    # Check if new edges exist
                    if new_e1 not in edge_set and new_e2 not in edge_set:
                        # Perform swap
                        edge_set.remove((u, v))
                        edge_set.remove((x, y))
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        edge_set.add(new_e1)
                        edge_set.add(new_e2)
                        adj[u].append(x)
                        adj[x].append(u)
                        adj[v].append(y)
                        adj[y].append(v)
                        
                        new_girth, _ = _find_shortest_cycle(adj)
                        if new_girth < current_girth:
                            # Revert
                            edge_set.remove(new_e1)
                            edge_set.remove(new_e2)
                            adj[u].remove(x)
                            adj[x].remove(u)
                            adj[v].remove(y)
                            adj[y].remove(v)
                            
                            edge_set.add((u, v))
                            edge_set.add((x, y))
                            adj[u].append(v)
                            adj[v].append(u)
                            adj[x].append(y)
                            adj[y].append(x)
                        else:
                            current_girth = new_girth

    # Fallback
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
