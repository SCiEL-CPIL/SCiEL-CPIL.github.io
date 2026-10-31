"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 130  # 20s safety margin before 150s kill
    import random
    
    def check_girth_fast(n, adj, k, g):
        """
        Check if graph has girth >= g using BFS from every vertex.
        For girth >= g (even), the BFS tree from any vertex up to depth r = g//2 - 1
        must be a perfect tree with no early collisions.
        """
        r = g // 2 - 1
        expected_counts = [1] + [k * (k - 1) ** i for i in range(r)]
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
            
            if counts != expected_counts:
                return False
        return True

    def find_short_cycle(n, adj, g):
        """Find a cycle of length < g if one exists. Returns (cycle_nodes, cycle_edges) or None."""
        r = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
                    elif dist[v] <= dist[u]:
                        # Found a back edge or cross edge creating a cycle
                        # Reconstruct cycle
                        if v == start:
                            continue  # skip self-loop equivalent
                        # Walk back from u to find LCA with v
                        # Actually, for BFS, if dist[v] == dist[u], it's a cross edge
                        # The cycle length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g and cycle_len >= 3:
                            # Reconstruct the cycle
                            # Path from start to u
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                cur = parent[cur]
                            path_u.reverse()
                            # Path from start to v
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = parent[cur]
                            path_v.reverse()
                            # LCA is start for both, so cycle is path_u + reversed(path_v[1:]) + edge u-v
                            # Actually the cycle goes: start -> ... -> u -> v -> ... -> start
                            # But we need the actual cycle nodes
                            cycle_nodes = path_u + list(reversed(path_v[1:]))
                            return cycle_nodes
                        elif dist[v] == dist[u] - 1 and v != parent[u]:
                            # This is a back edge to a non-parent at level dist[u]-1
                            # Cycle length = dist[u] - dist[v] + 1 = 2
                            # That's just the edge itself, skip
                            pass
                        elif dist[v] == dist[u] + 1 and u != parent[v]:
                            # Forward edge, already counted
                            pass
                        elif dist[v] == dist[u]:
                            # Cross edge, cycle length = 2*dist[u] + 1
                            cycle_len = 2 * dist[u] + 1
                            if cycle_len < g and cycle_len >= 3:
                                path_u = []
                                cur = u
                                while cur != -1:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_u.reverse()
                                path_v = []
                                cur = v
                                while cur != -1:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                path_v.reverse()
                                cycle_nodes = path_u + list(reversed(path_v[1:]))
                                return cycle_nodes
            if counts != expected_counts:
                # There's a collision but we didn't find the exact cycle above
                # Do a more careful search
                # For now, just return a dummy to signal failure
                # Let's do a proper cycle extraction
                dist2 = [-1] * n
                parent2 = [-1] * n
                dist2[start] = 0
                queue2 = deque([start])
                while queue2:
                    u = queue2.popleft()
                    if dist2[u] >= r:
                        continue
                    for v in adj[u]:
                        if dist2[v] == -1:
                            dist2[v] = dist2[u] + 1
                            parent2[v] = u
                            queue2.append(v)
                        else:
                            # Non-tree edge
                            if v == parent2[u]:
                                continue
                            if v == start:
                                continue
                            # Find LCA
                            au = u
                            av = v
                            du = dist2[au]
                            dv = dist2[av]
                            while du > dv:
                                au = parent2[au]
                                du -= 1
                            while dv > du:
                                av = parent2[av]
                                dv -= 1
                            while au != av:
                                au = parent2[au]
                                av = parent2[av]
                            # au is LCA
                            path_u = []
                            cur = u
                            while cur != au:
                                path_u.append(cur)
                                cur = parent2[cur]
                            path_u.append(au)
                            path_u.reverse()
                            path_v = []
                            cur = v
                            while cur != au:
                                path_v.append(cur)
                                cur = parent2[cur]
                            path_v.append(au)
                            path_v.reverse()
                            cycle_nodes = path_u[:-1] + path_v[::-1]
                            if len(cycle_nodes) < g:
                                return cycle_nodes
        return None

    def two_switch(adj, cycle1, cycle2):
        """
        Perform a 2-switch: remove two edges from cycle1 and two from cycle2,
        reconnect to break the cycles.
        cycle1, cycle2 are lists of nodes forming cycles.
        Returns True if successful.
        """
        n = len(adj)
        # Find edges in each cycle
        def get_cycle_edges(cycle):
            edges = set()
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                if u in adj[v]:
                    edges.add((min(u,v), max(u,v)))
            return edges
        
        e1 = get_cycle_edges(cycle1)
        e2 = get_cycle_edges(cycle2)
        
        if not e1 or not e2:
            return False
        
        # Pick two non-adjacent edges from each cycle
        list1 = list(e1)
        list2 = list(e2)
        
        if len(list1) < 2 or len(list2) < 2:
            return False
        
        # Try to find valid pairs
        for i in range(len(list1)):
            for j in range(i+1, len(list1)):
                (a1, b1), (a2, b2) = list1[i], list1[j]
                # Check they're not adjacent in the cycle
                # For simplicity, just try
                for p in range(len(list2)):
                    for q in range(p+1, len(list2)):
                        (c1, d1), (c2, d2) = list2[p], list2[q]
                        
                        # Remove edges
                        adj[a1].remove(b1)
                        adj[b1].remove(a1)
                        adj[a2].remove(b2)
                        adj[b2].remove(a2)
                        adj[c1].remove(d1)
                        adj[d1].remove(c1)
                        adj[c2].remove(d2)
                        adj[d2].remove(c2)
                        
                        # Add new edges: a1-c1, b1-d1, a2-c2, b2-d2 (one pairing)
                        # We need to maintain degree 3, so we add 4 edges
                        # But that changes the structure. Actually a 2-switch removes 2 edges
                        # from each cycle and adds 2 new edges.
                        
                        # Let me redo: a 2-switch on two cycles:
                        # Remove edge (a1,b1) from cycle1 and edge (c1,d1) from cycle2
                        # Add edges (a1,c1) and (b1,d1)
                        # This maintains regularity.
                        
                        # Undo the above
                        adj[a1].append(b1)
                        adj[b1].append(a1)
                        adj[a2].append(b2)
                        adj[b2].append(a2)
                        adj[c1].append(d1)
                        adj[d1].append(c1)
                        adj[c2].append(d2)
                        adj[d2].append(c2)
                        
                        # Now do a proper 2-switch: remove (a1,b1) and (c1,d1), add (a1,c1) and (b1,d1)
                        if c1 in adj[a1] or d1 in adj[a1]:
                            continue
                        if d1 in adj[b1] or c1 in adj[b1]:
                            continue
                        
                        adj[a1].remove(b1)
                        adj[b1].remove(a1)
                        adj[c1].remove(d1)
                        adj[d1].remove(c1)
                        
                        adj[a1].append(c1)
                        adj[c1].append(a1)
                        adj[b1].append(d1)
                        adj[d1].append(b1)
                        
                        return True
        return False

    def dodecahedron_base():
        """
        Return the dodecahedral graph as adjacency list (20 vertices, 3-regular, girth 5).
        """
        # Dodecahedron: 20 vertices, 30 edges
        # Standard labeling
        adj = [[] for _ in range(20)]
        edges = [
            (0,1),(0,2),(0,10),
            (1,3),(1,11),
            (2,4),(2,12),
            (3,5),(3,13),
            (4,6),(4,14),
            (5,7),(5,15),
            (6,8),(6,16),
            (7,9),(7,17),
            (8,10),(8,18),
            (9,11),(9,19),
            (10,18),(10,19),
            (11,12),(11,13),
            (12,14),(12,15),
            (13,15),(13,16),
            (14,16),(14,17),
            (15,17),(15,18),
            (16,18),(16,19),
            (17,19),
        ]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def voltage_lift(base_adj, voltages, m):
        """
        Given a base graph and voltages (each edge has a voltage in Z_m),
        construct the voltage graph (lift).
        voltages: dict mapping (u,v) to voltage value (for u<v)
        Returns adjacency list for the lifted graph on 20*m vertices.
        """
        n_base = len(base_adj)
        n = n_base * m
        adj = [[] for _ in range(n)]
        
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v:
                    vol = voltages.get((u,v), 0)
                    # u_i connects to v_{(i+vol)%m}
                    for i in range(m):
                        ui = u * m + i
                        vj = v * m + (i + vol) % m
                        adj[ui].append(vj)
        
        return adj

    def random_voltages(base_adj, m, seed=None):
        """Assign random voltages to edges of the base graph."""
        if seed is not None:
            random.seed(seed)
        voltages = {}
        for u in range(len(base_adj)):
            for v in base_adj[u]:
                if u < v:
                    voltages[(u,v)] = random.randint(0, m-1)
        return voltages

    def luw_construction(k, d):
        """
        LUW type construction for 3-regular graphs.
        Returns (n, edges, adj).
        """
        q = k
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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
        
        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        return n, edges, adj

    def random_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Moore bound for (3, 16): 2 + 3*(1+2+2^2+...+2^6) = 2 + 3*(2^7-1) = 2+3*127 = 383
    # But for bipartite: 2*(2^7-1) = 254... wait, let me recalculate.
    # Moore bound for degree d, girth g (even): 2 * (1 + (d-1) + (d-1)^2 + ... + (d-1)^(g/2-1))
    # = 2 * sum_{i=0}^{g/2-1} (d-1)^i
    # For d=3, g=16: 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 2 * 255 = 510
    # So n >= 510. Best known is 936.
    
    moore_bound = 2 * sum((k-1)**i for i in range(g//2))
    
    # Strategy:
    # 1. Try voltage lifts over dodecahedron with m from ~26 upward
    #    n = 20*m, so m >= ceil(510/20) = 26
    # 2. For each m, try random voltage assignments and check girth
    # 3. Use local search (2-switches) to fix short cycles
    # 4. Fallback to LUW
    
    base_adj = dodecahedron_base()
    base_edges = [(u, v) for u in range(20) for v in base_adj[u] if u < v]
    
    # Try voltage lifts
    best_result = None
    
    for m in range(26, 60):
        if time.time() > deadline - 40:
            break
        n = 20 * m
        if n < moore_bound:
            continue
        
        # Try several random voltage assignments
        for attempt in range(30):
            if time.time() > deadline - 40:
                break
            voltages = random_voltages(base_adj, m)
            adj = voltage_lift(base_adj, voltages, m)
            
            # Verify regularity and simplicity
            valid = True
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
                if len(set(adj[i])) != len(adj[i]):
                    valid = False
                    break
            if not valid:
                continue
            
            if check_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Try random regular graphs with local search
    for n in range(moore_bound, moore_bound + 200, 2):
        if time.time() > deadline - 30:
            break
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            
            # Local search: find short cycles and try to break them
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                cycle = find_short_cycle(n, adj, g)
                if cycle is None:
                    # Check girth
                    if check_girth_fast(n, adj, k, g):
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                    break
                # Try to break the cycle with a 2-switch
                # Find another cycle to switch with
                # For simplicity, just try random edge swaps
                # Pick a random edge from the cycle and try to swap
                if len(cycle) >= 4:
                    # Pick two non-adjacent edges from the cycle
                    idx1 = 0
                    idx2 = len(cycle) // 2
                    u1, v1 = cycle[idx1], cycle[(idx1+1) % len(cycle)]
                    u2, v2 = cycle[idx2], cycle[(idx2+1) % len(cycle)]
                    
                    # Find alternative neighbors
                    # This is complex; skip for now
                    pass
            
            # If local search didn't work, just check
            if check_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback: LUW with d=7 (girth >= 16, n ~ 13122)
    for d in [7, 8]:
        if time.time() > deadline - 10:
            break
        try:
            n, edges, adj = luw_construction(k, d)
            if check_girth_fast(n, adj, k, g):
                return n, edges
        except:
            pass

    # Final fallback: LUW with larger d
    d_orig = max(7, g - 5 if (g - 5) % 2 else g - 4)
    n, edges, adj = luw_construction(k, d_orig)
    return n, edges

# EVOLVE-BLOCK-END
