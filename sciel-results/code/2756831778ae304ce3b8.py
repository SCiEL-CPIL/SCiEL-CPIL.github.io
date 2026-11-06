import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # 110s limit
    
    n = 16
    num_colors = 15
    
    # Precompute edge indices
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    # Function to check validity
    def is_valid(colors):
        # colors is a list of 120 ints
        # Check all K4s
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        edges = [
                            (a, b), (a, c), (a, d),
                            (b, c), (b, d),
                            (c, d)
                        ]
                        col_set = set()
                        for e in edges:
                            u, v = e
                            if u > v: u, v = v, u
                            col_set.add(colors[edge_to_idx[(u, v)]])
                        if len(col_set) < 5:
                            return False
        return True

    # Generate standard 1-factorization of K_16
    # Vertices 0..14, and 15 as infinity
    colors = [0] * 120
    for k in range(num_colors):
        # Edge (15, k)
        u, v = 15, k
        if u > v: u, v = v, u
        colors[edge_to_idx[(u, v)]] = k
        
        # Edges (k+i, k-i) mod 15 for i=1..7
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u > v: u, v = v, u
            colors[edge_to_idx[(u, v)]] = k
            
    if is_valid(colors):
        return colors

    # If deterministic fails, try random search with time limit
    # (Keep existing logic as fallback but optimized)
    def generate_random_1_factorization():
        colors = [0] * 120
        remaining_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
        
        for color in range(num_colors):
            matching = find_random_matching(remaining_edges, list(range(n)))
            if matching is None:
                return None
            
            for (u, v) in matching:
                if u > v: u, v = v, u
                colors[edge_to_idx[(u, v)]] = color
                remaining_edges.discard((u, v))
                
        return colors

    def find_random_matching(edges, vertices):
        if not vertices:
            return []
        v = vertices[0]
        neighbors = []
        for (x, y) in edges:
            if x == v:
                neighbors.append(y)
            elif y == v:
                neighbors.append(x)
        
        if not neighbors:
            return None
            
        random.shuffle(neighbors)
        
        for u in neighbors:
            new_edges = set(edges)
            new_edges.discard((v, u))
            new_edges.discard((u, v))
            new_vertices = [x for x in vertices if x != v and x != u]
            m = find_random_matching(new_edges, new_vertices)
            if m is not None:
                return [(v, u)] + m
        return None

    while time.time() < deadline:
        colors = generate_random_1_factorization()
        if colors is None:
            continue
        if is_valid(colors):
            return colors

    # Fallback to seed
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
