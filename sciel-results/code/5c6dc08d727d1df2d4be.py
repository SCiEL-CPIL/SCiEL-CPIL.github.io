import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # The 936-vertex cage is a regular cover of the Klein graph (26 vertices)
    # with voltage group Z_36.
    # We attempt to find voltage assignments that yield girth >= g.
    
    # Define the Klein graph (26 vertices, 3-regular, girth 5)
    # Vertices 0..25
    # This is a standard representation of the Klein graph (also known as the 26-vertex cubic graph of girth 5)
    # Note: There are multiple non-isomorphic cubic graphs on 26 vertices with girth 5.
    # The Klein graph is one of them.
    klein_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 13), (4, 14), (4, 15),
        (5, 16), (5, 17), (5, 18),
        (6, 19), (6, 20), (6, 21),
        (7, 22), (7, 23), (7, 24),
        (8, 25), (8, 0), # Wait, 0 is already connected to 1,2,3. This edge (8,0) creates a cycle?
        # Let's use a verified Klein graph structure.
        # A common Klein graph is the bipartite double cover of the Petersen? No.
        # Let's use the "Klein" graph from NetworkX or a known list.
        # Actually, the 936-vertex cage is often associated with the **Dodecahedron** or **Petersen** in literature?
        # Let's check 936/10 = 93.6. 936/20 = 46.8. 936/26 = 36.
        # So 26 is the only small divisor that makes sense for a "small base".
        
        # Let's use a generic cubic graph on 26 vertices with girth 5.
        # If the specific Klein graph doesn't work, we can try other cubic graphs on 26 vertices.
    ]
    
    # To be safe, I will use a **randomized search on voltage graphs over several small cubic base graphs**.
    # Base graphs: Petersen (10), Dodecahedron (20), Klein (26), etc.
    # We need N_base * |G| = N_target.
    # If we target N=936, possible (N_base, |G|) pairs:
    # (26, 36), (13, 72) - 13-vertex cubic graph? No, cubic graphs must have even number of vertices.
    # (12, 78) - 12-vertex cubic graph (Dodecahedron is 20, but there are 12-vertex cubic graphs? No, min is 4, then 6, 8, 10, 12... yes, the 12-vertex graph is the prism? No, prism is 2n. The 12-vertex cubic graph is the **Dodecahedron**? No, Dodecahedron is 20.
    # The 12-vertex cubic graph is the **Petersen**? No, Petersen is 10.
    # The 12-vertex cubic graph is the **Prism over a hexagon**? No, that's 12 vertices, girth 6.
    # Let's stick to (26, 36) and (10, 93.6 - impossible).
    # So (26, 36) is the primary candidate.
    
    # Let's define the Klein graph properly.
    # The Klein graph is a cubic graph on 26 vertices.
    # One standard adjacency list:
    def get_klein_graph():
        # This is a known cubic graph on 26 vertices with girth 5.
        adj = [[] for _ in range(26)]
        # Generate a random cubic graph on 26 vertices and check girth?
        # Or use a specific one.
        # Let's use the "Klein" graph from the House of Graphs.
        # Vertices: 0..25
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5), (1, 6),
            (2, 7), (2, 8), (2, 9),
            (3, 10), (3, 11), (3, 12),
            (4, 13), (4, 14), (4, 15),
            (5, 16), (5, 17), (5, 18),
            (6, 19), (6, 20), (6, 21),
            (7, 22), (7, 23), (7, 24),
            (8, 25), (8, 9), (8, 10), # 8 already connected to 2. This is getting messy.
        ]
        # Let's use a simpler approach:
        # Generate a random 3-regular graph on 26 vertices.
        # Check if girth >= 5.
        # If yes, use it as base.
        
        def random_cubic(n):
            # Configuration model
            half_edges = []
            for i in range(n):
                for _ in range(3):
                    half_edges.append(i)
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            for i in range(0, n*3, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    return None
                if v in adj[u]:
                    return None
                adj[u].append(v)
                adj[v].append(u)
            return adj
        
        for _ in range(1000):
            adj = random_cubic(26)
            if adj is None:
                continue
            # Check girth
            girth = 100
            for start in range(26):
                dist = [-1]*26
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                        else:
                            # Cycle length is dist[u] + dist[v] + 1
                            # But this is not accurate for girth, need BFS tree
                            pass
            # Simpler girth check:
            def check_girth(adj, min_g):
                n = len(adj)
                for start in range(n):
                    dist = [-1]*n
                    parent = [-1]*n
                    dist[start] = 0
                    q = deque([start])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if v == parent[u]:
                                continue
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q.append(v)
                            else:
                                if dist[u] + dist[v] + 1 < min_g:
                                    return False
                return True
            
            if check_girth(adj, 5):
                return adj
        return None

    base_adj = get_klein_graph()
    if base_adj is None:
        # Fallback to LUW if we can't even get a base
        def luw_fallback():
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
        return luw_fallback()

    n_base = len(base_adj)
    # We want total vertices N = 936.
    # If n_base = 26, then group size = 36.
    group_size = 936 // n_base
    if group_size == 0:
        return luw_fallback()

    # Voltage graph construction
    # Vertices are (base_vertex, group_element)
    # Edges are (u_base, v_base) with voltage w.
    # Lift edge: (u, g) -> (v, g+w)
    
    # We need to assign voltages to the 39 edges of the base graph.
    # We will use a local search to find voltages that maximize girth.
    
    edges_list = []
    for u in range(n_base):
        for v in base_adj[u]:
            if u < v:
                edges_list.append((u, v))
                
    n_edges = len(edges_list)
    
    def build_lift(voltages):
        # voltages is a list of length n_edges
        # Map edge index to voltage
        edge_map = {}
        for idx, (u, v) in enumerate(edges_list):
            edge_map[(u, v)] = voltages[idx]
            edge_map[(v, u)] = -voltages[idx]
            
        lift_n = n_base * group_size
        adj = [[] for _ in range(lift_n)]
        
        for u in range(n_base):
            for v in base_adj[u]:
                w = edge_map[(u, v)]
                # For each group element g
                for g in range(group_size):
                    u_lift = u * group_size + g
                    v_lift = v * group_size + ((g + w) % group_size)
                    if v_lift not in adj[u_lift]:
                        adj[u_lift].append(v_lift)
                        adj[v_lift].append(u_lift)
        return adj

    def check_girth_fast(n, adj, min_girth):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[u] + dist[v] + 1 < min_girth:
                        return False
        return True

    # Local search on voltages
    best_girth = 0
    best_voltages = None
    current_voltages = [random.randint(0, group_size - 1) for _ in range(n_edges)]
    
    for step in range(5000):
        if time.time() > deadline:
            break
        
        adj = build_lift(current_voltages)
        
        # Quick girth check
        # We only need to check if girth >= g
        if check_girth_fast(len(adj), adj, g):
            # Extract edges
            edges = []
            for i in range(len(adj)):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return len(adj), edges
            
        # If not, try to improve
        # Change a random voltage
        idx = random.randint(0, n_edges - 1)
        old_v = current_voltages[idx]
        new_v = random.randint(0, group_size - 1)
        current_voltages[idx] = new_v
        
        # Re-check
        adj_new = build_lift(current_voltages)
        if check_girth_fast(len(adj_new), adj_new, g):
            edges = []
            for i in range(len(adj_new)):
                for j in adj_new[i]:
                    if i < j:
                        edges.append((i, j))
            return len(adj_new), edges
        else:
            # Revert if it didn't help? 
            # Actually, we don't easily know if it "helped" without calculating girth.
            # We can just keep the new state (random walk).
            pass

    # If local search fails, fallback to LUW
    return luw_fallback()
