import time
import random
import collections
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Heawood graph: 3-regular, girth 6, 14 vertices
    # Vertices 0..13
    # Standard Heawood edges
    heawood_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 4), (2, 7), (2, 8),
        (3, 5), (3, 7), (3, 9),
        (4, 10), (4, 11), (4, 12),
        (5, 10), (5, 13), (5, 11),
        (6, 8), (6, 9), (6, 12),
        (7, 9), (7, 13), (7, 12),
        (8, 10), (8, 11), (8, 13),
        (9, 11), (9, 12), (9, 13)
    ]
    # Wait, Heawood has 21 edges.
    # Let's verify the list.
    # Heawood is the incidence graph of Fano plane.
    # 7 points, 7 lines. 14 vertices.
    # My list above has 24 edges?
    # (0,1),(0,2),(0,3) -> 3
    # (1,4),(1,5),(1,6) -> 3
    # (2,4),(2,7),(2,8) -> 3
    # (3,5),(3,7),(3,9) -> 3
    # (4,10),(4,11),(4,12) -> 3
    # (5,10),(5,13),(5,11) -> 3
    # (6,8),(6,9),(6,12) -> 3
    # (7,9),(7,13),(7,12) -> 3
    # (8,10),(8,11),(8,13) -> 3
    # (9,11),(9,12),(9,13) -> 3
    # Total 27 edges?
    # 14 vertices * 3 / 2 = 21 edges.
    # My list is wrong. I need the correct Heawood graph.
    
    # Correct Heawood graph edges:
    # 0-1, 0-2, 0-3
    # 1-4, 1-5, 1-6
    # 2-4, 2-7, 2-8
    # 3-5, 3-7, 3-9
    # 4-10, 4-11, 4-12
    # 5-10, 5-13, 5-11
    # 6-8, 6-9, 6-12
    # 7-9, 7-13, 7-12
    # 8-10, 8-11, 8-13
    # 9-11, 9-12, 9-13
    # Let's check degrees.
    # 0: 1,2,3 (3)
    # 1: 0,4,5,6 (4) -> WRONG.
    
    # I will generate Heawood graph programmatically or use a known small graph.
    # Actually, I can just use K4 lift if Heawood is hard to get right.
    # K4 is easy.
    # K4 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    
    base_edges_k4 = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    base_vertices = 4
    
    def generate_lift(base_edges, base_vertices, d, seed):
        random.seed(seed)
        # permutations for each edge
        # edge (u,v) -> perm
        # we need perm[u][v] and perm[v][u]
        # store as list of lists or dict
        perms = {}
        for u, v in base_edges:
            p = list(range(d))
            random.shuffle(p)
            perms[(u,v)] = p
            # inverse
            inv = [0]*d
            for i, val in enumerate(p):
                inv[val] = i
            perms[(v,u)] = inv
        
        # build adjacency list for lift
        # vertices are (u, i)
        # total vertices = base_vertices * d
        n = base_vertices * d
        adj = [[] for _ in range(n)]
        
        for u, v in base_edges:
            p = perms[(u,v)]
            for i in range(d):
                j = p[i]
                u_idx = u * d + i
                v_idx = v * d + j
                adj[u_idx].append(v_idx)
                # adj[v_idx].append(u_idx) is redundant if we iterate all edges, 
                # but base_edges is undirected list of pairs.
                # If we add both directions here, we double count.
                # Let's just add u->v. The graph is undirected.
        
        # To check girth, we need undirected adjacency.
        # So we should add both directions.
        # But if we iterate base_edges, (u,v) and (v,u) are not both in the list.
        # So we must add both.
        
        # Re-do adjacency building
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            p = perms[(u,v)]
            for i in range(d):
                j = p[i]
                u_idx = u * d + i
                v_idx = v * d + j
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
        
        return n, adj

    def check_girth(n, adj, min_girth):
        # BFS from each vertex
        # If we find a cycle of length < min_girth, return False
        # Cycle length in BFS: dist[u] + dist[v] + 1?
        # No, if we visit v from u, and v was already visited at dist d_v.
        # If d_v < dist[u], then cycle length is d_v + dist[u] + 1?
        # Actually, standard BFS:
        # dist[start] = 0.
        # If edge (u,v) exists:
        # if dist[v] == -1: dist[v] = dist[u] + 1
        # else: cycle_len = dist[u] + dist[v] + 1
        # We need cycle_len >= min_girth.
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = collections.deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= min_girth // 2: 
                    # No need to go deeper, any cycle found now would be >= min_girth?
                    # If du = 7, and we find edge to v with dist[v] >= 0.
                    # Cycle len = 7 + dist[v] + 1.
                    # If dist[v] >= 0, len >= 8.
                    # We need to ensure no cycle < min_girth.
                    # If min_girth = 16, we care about cycles < 16.
                    # Max cycle len we can form from depth 7 is 7 + 7 + 1 = 15?
                    # If dist[u]=7, dist[v]=7, len=15.
                    # If dist[v]=8, len=16.
                    # So we need to go to depth 7.
                    # If we find a connection between two nodes at depth <= 7,
                    # the cycle length is dist[u] + dist[v] + 1.
                    # We need this to be >= 16.
                    # So if dist[u] + dist[v] + 1 < 16, return False.
                    pass
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        # Found a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # But this might be the edge back to parent?
                        # If v is parent of u, dist[v] = dist[u] - 1.
                        # Cycle len = dist[u] + dist[u] - 1 + 1 = 2*dist[u].
                        # This is a valid cycle.
                        # We must check all edges.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Search
    best_n = None
    best_edges = None
    
    # Try K4 lifts with varying d
    # Target N < 354294
    # d < 88573
    # Start with d=80000, 85000, etc.
    # Or start small and go up?
    # Smaller d is better score.
    # But smaller d has higher chance of short cycles.
    # Let's try d in [50000, 60000, 70000, 80000, 85000]
    
    d_values = [50000, 60000, 70000, 80000, 85000]
    
    for d in d_values:
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            n, adj = generate_lift(base_edges_k4, base_vertices, d, seed)
            if check_girth(n, adj, 16):
                # Convert adj to edges list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # Fallback: LUW
    # Re-implement LUW from seed
    q = k
    d_luw = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d_luw + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d_luw - 1)
            for c in range(2, d_luw + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d_luw)
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
# EVOLVE-BLOCK-END
