import random
import itertools

def construct():
    n = 16
    # Multiplicative structure mod 17
    # Primitive root 3
    # g^k mod 17
    # We assign vertex i to exponent i.
    # Edge (i,j) color is g^(i+j) mod 17.
    # Map non-zero residues to 0..15
    
    # Generate powers of 3 mod 17
    pow3 = [1] * 16
    for i in range(1, 16):
        pow3[i] = (pow3[i-1] * 3) % 17
        
    # Map residue to color index 0..15
    # Residues are 1..16
    # pow3[k] is the residue for exponent k
    # We want color of edge (i,j) to be based on i+j
    # Let's just use the residue itself, mapped to 0..15
    # Residue 1 -> 0, 2 -> 1, ... 16 -> 15?
    # Or just use the exponent sum mod 16?
    # The previous analysis showed exponents mod 16 are the colors.
    # So color(i,j) = (i+j) % 16.
    # This uses 16 colors: 0..15.
    
    # Initialize colors
    colors = [(i + j) % 16 for i in range(n) for j in range(i + 1, n)]
    
    # Check validity
    def is_valid(colors):
        # Reconstruct adjacency or just use list
        # colors is flat list of 120 edges
        # Edge (i,j) index:
        # i=0: 0..14 (15 edges)
        # i=1: 15..28 (14 edges)
        # ...
        # Index formula: i*n - i*(i+1)//2 + (j-i-1)
        # Or just iterate
        
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            edges = [
                (u, v), (u, w), (u, x),
                (v, w), (v, x),
                (w, x)
            ]
            edge_colors = set()
            for (a, b) in edges:
                # Get color
                # Calculate index
                idx = a * n - a * (a + 1) // 2 + (b - a - 1)
                edge_colors.add(colors[idx])
            if len(edge_colors) < 5:
                return False
        return True

    if is_valid(colors):
        return colors

    # Local Search
    random.seed(42)
    current_colors = colors[:]
    
    for _ in range(10000): # Limit iterations
        if is_valid(current_colors):
            return current_colors
            
        # Find an invalid K4
        # To save time, just pick random K4?
        # Better: pick random edge, check K4s containing it?
        # For simplicity, pick random K4
        quad = random.sample(range(n), 4)
        u, v, w, x = quad
        edges = [
            (u, v), (u, w), (u, x),
            (v, w), (v, x),
            (w, x)
        ]
        
        # Check if this K4 is invalid
        edge_colors = set()
        indices = []
        for (a, b) in edges:
            idx = a * n - a * (a + 1) // 2 + (b - a - 1)
            indices.append(idx)
            edge_colors.add(current_colors[idx])
            
        if len(edge_colors) >= 5:
            continue # This K4 is fine, try another
            
        # Pick a random edge in this K4 and change its color
        edge_idx = random.choice(indices)
        old_color = current_colors[edge_idx]
        new_color = random.randint(0, 15)
        current_colors[edge_idx] = new_color
        
    return current_colors
