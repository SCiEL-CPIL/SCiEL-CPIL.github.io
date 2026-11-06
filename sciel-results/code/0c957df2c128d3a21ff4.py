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

    # Function to generate random 1-factorization
    def generate_random_1_factorization():
        # Returns a list of 120 colors
        colors = [0] * 120
        remaining_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
        
        # We need to pick 15 perfect matchings
        for color in range(num_colors):
            # Find a random perfect matching in remaining_edges
            # Use recursive backtracking
            matching = find_random_matching(remaining_edges, list(range(n)))
            if matching is None:
                return None # Should not happen
            
            for (u, v) in matching:
                if u > v: u, v = v, u
                colors[edge_to_idx[(u, v)]] = color
                remaining_edges.discard((u, v))
                
        return colors

    # Recursive helper
    def find_random_matching(edges, vertices):
        if not vertices:
            return []
        v = vertices[0]
        # Find neighbors of v in edges
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
            # Try to match v with u
            new_edges = set(edges)
            new_edges.discard((v, u))
            new_edges.discard((u, v))
            
            # Check if u is in vertices (it should be)
            new_vertices = [x for x in vertices if x != v and x != u]
            
            # Recurse
            m = find_random_matching(new_edges, new_vertices)
            if m is not None:
                return [(v, u)] + m
                
        return None

    best_colors = None
    best_score = 1000
    
    # Try to find a valid 15-coloring
    while time.time() < deadline:
        colors = generate_random_1_factorization()
        if colors is None:
            continue
        if is_valid(colors):
            return colors

    # If no 15-coloring found, return seed (17 colors)
    # Seed: (i+j) % 17
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
