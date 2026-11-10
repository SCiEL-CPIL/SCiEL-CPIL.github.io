import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, safety margin
    
    # Helper: Edge count from adj list
    def edge_count_from_adj(adj):
        count = 0
        for u in range(len(adj)):
            count += bin(adj[u]).count('1')
        return count // 2

    # Helper: Check if graph is 3-colorable
    def is_3_colorable(adj, n):
        # colors: 0=uncolored, 1,2,3
        colors = [0] * n
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine forbidden colors
            forbidden = 0
            for u in range(n):
                if (adj[v] >> u) & 1:
                    if colors[u] != 0:
                        forbidden |= (1 << colors[u])
                        
            for c in range(1, 4):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = 0
            return False
            
        return backtrack(0)

    # Strategy 1: Grötzsch Blow-up
    def get_groetzsch_graph():
        groetzsch_edges = []
        cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        groetzsch_edges.extend(cycle_edges)
        
        neighbors = {
            0: [1, 4],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3, 0]
        }
        
        for v in range(5):
            v_prime = 5 + v
            for u in neighbors[v]:
                groetzsch_edges.append((v_prime, u))
                
        for v in range(5):
            groetzsch_edges.append((10, 5 + v))
            
        def generate_compositions(n, k):
            if k == 1:
                yield [n]
            else:
                for i in range(1, n - k + 2):
                    for tail in generate_compositions(n - i, k - 1):
                        yield [i] + tail
                        
        best_edges_count = -1
        best_sizes = None
        
        for comp in generate_compositions(16, 11):
            comp_sorted = sorted(comp, reverse=True)
            s = [0] * 11
            s[10] = comp_sorted[0]
            for i in range(5):
                s[5+i] = comp_sorted[1+i]
            for i in range(5):
                s[i] = comp_sorted[6+i]
                
            edge_count = 0
            for u, v in groetzsch_edges:
                edge_count += s[u] * s[v]
                
            if edge_count > best_edges_count:
                best_edges_count = edge_count
                best_sizes = s[:]
                
        adj = [0] * 16
        offsets = [0] * 12
        for i in range(11):
            offsets[i+1] = offsets[i] + best_sizes[i]
            
        for u, v in groetzsch_edges:
            for i in range(best_sizes[u]):
                u_node = offsets[u] + i
                for j in range(best_sizes[v]):
                    v_node = offsets[v] + j
                    adj[u_node] |= (1 << v_node)
                    adj[v_node] |= (1 << u_node)
                    
        # Greedy edge addition
        def can_add_edge(u, v, adj):
            return (adj[u] & adj[v]) == 0
            
        best_adj = [x for x in adj]
        best_count = edge_count_from_adj(best_adj)
        
        import random
        rng = random.Random(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            work_adj = [x for x in best_adj]
            changed = True
            while changed and time.time() < deadline:
                changed = False
                pairs = [(u, v) for u in range(16) for v in range(u+1, 16) if not (work_adj[u] >> v) & 1]
                rng.shuffle(pairs)
                for u, v in pairs:
                    if time.time() > deadline:
                        break
                    if not (work_adj[u] >> v) & 1:
                        if can_add_edge(u, v, work_adj):
                            work_adj[u] |= (1 << v)
                            work_adj[v] |= (1 << u)
                            changed = True
                            
            count = edge_count_from_adj(work_adj)
            if count > best_count:
                best_count = count
                best_adj = work_adj[:]
                
        edges = []
        for u in range(16):
            for v in range(u+1, 16):
                if (best_adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Strategy 2: Mycielski of Clebsch
    def get_mycielski_clebsch_graph():
        # Clebsch graph edges (16 vertices)
        # Vertices 0..15
        # Hardcoded edges for 5-regular Clebsch
        clebsch_edges = [
            (0, 5), (0, 6), (0, 9), (0, 10), (0, 15),
            (1, 4), (1, 7), (1, 8), (1, 11), (1, 14),
            (2, 3), (2, 8), (2, 11), (2, 12), (2, 13),
            (3, 2), (3, 9), (3, 10), (3, 13), (3, 14),
            (4, 1), (4, 6), (4, 13), (4, 14), (4, 15),
            (5, 0), (5, 7), (5, 8), (5, 11), (5, 12),
            (6, 0), (6, 4), (6, 9), (6, 10), (6, 15),
            (7, 1), (7, 5), (7, 12), (7, 13), (7, 14),
            (8, 1), (8, 2), (8, 5), (8, 10), (8, 15),
            (9, 0), (9, 3), (9, 6), (9, 8), (9, 12),
            (10, 0), (10, 3), (10, 6), (10, 8), (10, 11),
            (11, 1), (11, 2), (11, 5), (11, 10), (11, 12),
            (12, 2), (12, 5), (12, 7), (12, 9), (12, 13),
            (13, 2), (13, 3), (13, 4), (13, 7), (13, 12),
            (14, 1), (14, 3), (14, 4), (14, 7), (14, 15),
            (15, 0), (15, 4), (15, 6), (15, 8), (15, 14)
        ]
        
        # Construct Mycielski
        # 16 original, 16 copies (16..31), 1 root (32)
        n_orig = 16
        n_total = 33
        adj = [0] * n_total
        
        for u, v in clebsch_edges:
            # Original edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Copy edges: v' connected to u
            v_prime = n_orig + v
            adj[v_prime] |= (1 << u)
            adj[u] |= (1 << v_prime)
            
            u_prime = n_orig + u
            adj[u_prime] |= (1 << v)
            adj[v] |= (1 << u_prime)
            
        # Root connected to all copies
        root = 32
        for v in range(n_orig):
            v_prime = n_orig + v
            adj[root] |= (1 << v_prime)
            adj[v_prime] |= (1 << root)
            
        # Greedy deletion to find 16-vertex subgraph
        # We want to keep 16 vertices.
        # Start with all 33.
        current_vertices = set(range(33))
        
        while len(current_vertices) > 16:
            if time.time() > deadline:
                break
            min_deg = float('inf')
            min_v = None
            for v in current_vertices:
                # Count neighbors in current_vertices
                deg = 0
                mask = adj[v]
                # Iterate bits
                m = mask
                while m:
                    lsb = m & -m
                    neighbor = lsb.bit_length() - 1
                    if neighbor in current_vertices:
                        deg += 1
                    m ^= lsb
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            current_vertices.remove(min_v)
            
        # Build subgraph adj
        sub_adj = [0] * 16
        vertex_map = {v: i for i, v in enumerate(sorted(current_vertices))}
        
        for u in current_vertices:
            for v in current_vertices:
                if u < v and (adj[u] >> v) & 1:
                    i = vertex_map[u]
                    j = vertex_map[v]
                    sub_adj[i] |= (1 << j)
                    sub_adj[j] |= (1 << i)
                    
        # Check 3-colorability
        if is_3_colorable(sub_adj, 16):
            return [] # Invalid
            
        edges = []
        for u in range(16):
            for v in range(u+1, 16):
                if (sub_adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Execute strategies
    best_edges = []
    best_count = 0
    
    # Try Grötzsch
    g_edges = get_groetzsch_graph()
    if g_edges:
        best_edges = g_edges
        best_count = len(g_edges)
        
    # Try Mycielski Clebsch
    if time.time() < deadline:
        m_edges = get_mycielski_clebsch_graph()
        if m_edges and len(m_edges) > best_count:
            best_edges = m_edges
            best_count = len(m_edges)
            
    return best_edges
