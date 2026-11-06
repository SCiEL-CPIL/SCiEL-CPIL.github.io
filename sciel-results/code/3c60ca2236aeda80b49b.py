"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def edge_index(n, i, j):
        """Index for edge (i,j) with i<j in lexicographic order."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """Check if coloring has no monochromatic C4."""
        for c in range(5):
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = edge_index(n, i, j)
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
            for u in range(n):
                for v in range(u + 1, n):
                    if len(adj[u].intersection(adj[v])) >= 2:
                        return False
        return True

    def greedy_construct(n, seed=42):
        """
        Greedy construction: build coloring vertex by vertex.
        When adding vertex v, for each u < v, choose color for edge (u,v)
        such that no C4 is created.
        
        A C4 would be created if for some color c, u and v already have
        >= 2 common neighbors in color c (excluding each other).
        """
        rng = random.Random(seed)
        colors = [None] * (n * (n - 1) // 2)
        
        # Track neighbors for each color: adj[c][v] = set of neighbors of v in color c
        adj = [set() for _ in range(5)]
        for c in range(5):
            adj[c] = [set() for _ in range(n)]
        
        # Process vertices in order 0, 1, ..., n-1
        # When processing vertex v, all edges (u, v) for u < v need colors
        for v in range(n):
            for u in range(v):
                idx = edge_index(n, u, v)
                # Try each color, pick one that doesn't create a C4
                valid_colors = []
                for c in range(5):
                    # Check if adding edge (u,v) with color c creates a C4
                    # This happens if u and v have >= 2 common neighbors in color c
                    common = adj[c][u].intersection(adj[c][v])
                    if len(common) < 2:
                        valid_colors.append(c)
                
                if not valid_colors:
                    return None  # Dead end
                
                # Pick a random valid color
                c = rng.choice(valid_colors)
                colors[idx] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
        
        # Verify
        if is_valid(n, colors):
            return colors
        return None

    # Try to construct for n = 26, 25, 24, ...
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline:
            break
        
        # Try multiple seeds for robustness
        for seed in range(50):
            if time.time() > deadline:
                break
            
            colors = greedy_construct(n, seed)
            if colors is not None:
                return {'n': n, 'colors': colors}
    
    # Fallback: return a small valid coloring
    n = 6
    # Simple valid 5-coloring of K_6: just use distinct patterns
    # Actually, let's construct one that's definitely valid
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use (j-i) mod 5 as color - need to verify this is C4-free
            colors.append((j - i) % 5)
    
    # Verify fallback
    if not is_valid(n, colors):
        # Use an even simpler fallback
        n = 5
        colors = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
