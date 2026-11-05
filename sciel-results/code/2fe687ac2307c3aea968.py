import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    best_score = 18
    best_edges = []
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def find_k4s(adj):
        k4s = []
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if (adj[u][v] and adj[u][w] and adj[u][x] and
                adj[v][w] and adj[v][x] and
                adj[w][x]):
                k4s.append(quad)
        return k4s

    def remove_k4s(adj):
        adj = [row[:] for row in adj]
        while True:
            k4s = find_k4s(adj)
            if not k4s:
                break
            edge_counts = {}
            for quad in k4s:
                edges_in_quad = list(itertools.combinations(quad, 2))
                for e in edges_in_quad:
                    edge_counts[e] = edge_counts.get(e, 0) + 1
            if not edge_counts:
                break
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            e = random.choice(candidates)
            u, v = e
            adj[u][v] = 0
            adj[v][u] = 0
        return adj

    def get_triangles(adj):
        triangles = []
        for tri in itertools.combinations(range(n), 3):
            u, v, w = tri
            if adj[u][v] and adj[v][w] and adj[u][w]:
                triangles.append(tri)
        return triangles

    def greedy_vertex_cover(triangles):
        if not triangles:
            return 0
        # Count vertex frequencies in triangles
        freq = [0] * n
        for t in triangles:
            for v in t:
                freq[v] += 1
        
        cover = set()
        remaining_tris = list(triangles)
        
        while remaining_tris:
            # Pick vertex with max freq in remaining_tris
            # Recalculate freq? Or just use global freq?
            # Global freq is faster but less accurate.
            # Let's use global freq for speed.
            # Actually, we should pick from remaining_tris.
            # For speed, just pick max global freq among vertices not in cover?
            # No, we need to hit all remaining_tris.
            # Let's just pick a random triangle and pick a vertex from it with max global freq.
            t = random.choice(remaining_tris)
            # Pick vertex in t with max freq
            v = max(t, key=lambda x: freq[x])
            cover.add(v)
            # Remove all triangles containing v
            remaining_tris = [tt for tt in remaining_tris if v not in tt]
            
        return len(cover)

    def max_triangle_free_size_exact(adj):
        # Check from largest k down to 1
        # Use bitmask representation for speed?
        # For n=18, simple loops are okay if pruned.
        # But we are only calling this if heuristic suggests it's good.
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                # Check if triangle-free
                is_tf = True
                verts = list(subset)
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        if adj[u][v]:
                            for w in verts[j+1:]:
                                if adj[u][w] and adj[v][w]:
                                    is_tf = False
                                    break
                    if not is_tf:
                        break
                if is_tf:
                    return k
        return 0

    # Main loop
    for p in [0.4, 0.5, 0.6]:
        while time.time() < deadline:
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            adj_clean = remove_k4s(adj)
            
            triangles = get_triangles(adj_clean)
            
            # Heuristic: Greedy vertex cover size
            c_greedy = greedy_vertex_cover(triangles)
            
            # If c_greedy is small, the max triangle-free size is large.
            # We want small max triangle-free size, so we want large c_min.
            # c_min <= c_greedy.
            # So max_tf_size >= n - c_greedy.
            # If n - c_greedy >= best_score, then this graph cannot beat best_score.
            if n - c_greedy >= best_score:
                continue
            
            # Exact check
            score = max_triangle_free_size_exact(adj_clean)
            
            if score < best_score:
                best_score = score
                best_edges = get_edges_from_adj(adj_clean)

    return best_edges
