"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # 1. Generate the known valid n=32 coloring (Greenwood-Gleason)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    best_n = 32
    best_colors = base_colors

    # 2. Try to find a larger coloring using Local Search
    # We aim for n=50. If we can't find it, we stick with 32.
    # We can also try n=33, 34... but 50 is the target.
    
    def count_triangles(colors, n):
        # colors is a list of length n*(n-1)//2
        # This is slow. We need a faster check or incremental update.
        # For local search, we just need to find *one* triangle to break.
        pass

    def find_triangle(colors, n):
        # colors is a flat list. index = i*n + j - (i+1)//2 ? 
        # Standard lexicographic index:
        # idx(i, j) for i<j: i*n - i*(i+1)//2 + (j-i-1)
        # Let's use a 2D array for speed in search, then flatten at the end.
        pass

    # Using numpy for speed in local search
    target_n = 50
    if time.time() < deadline - 5:
        # Try to solve for n=50
        # Initialize random coloring
        num_edges = target_n * (target_n - 1) // 2
        # 4 colors: 0,1,2,3
        # We will use a 2D array adj[i][j]
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * target_n - i * (i + 1) // 2 + (j - i - 1)

        # Precompute neighbors for triangle checking? 
        # A triangle (i,j,k) is monochromatic if c(i,j)==c(i,k)==c(j,k).
        
        # Local Search: Min-Conflicts
        # 1. Random init
        # 2. Find all triangles
        # 3. Pick a random triangle, flip one of its edges
        # 4. Repeat
        
        # Optimization: Only check triangles involving changed edges?
        # Simpler: Just run a few iterations.
        
        max_iter = 5000
        for restart in range(5):
            if time.time() > deadline - 5: break
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.uint8)
            # Convert to 2D for easier access? 
            # c[i][j]
            c = np.zeros((target_n, target_n), dtype=np.uint8)
            for i in range(target_n):
                for j in range(i+1, target_n):
                    c[i, j] = colors[get_idx(i, j)]
                    c[j, i] = colors[get_idx(i, j)]

            # Check validity initially? Unlikely.
            # Run local search
            for it in range(max_iter):
                if time.time() > deadline - 5: break
                # Find a random triangle
                # Randomly pick 3 vertices
                while True:
                    i, j, k = np.random.randint(0, target_n, 3)
                    if i != j and j != k and i != k:
                        break
                c_ij = c[i, j]
                c_ik = c[i, k]
                c_jk = c[j, k]
                if c_ij == c_ik == c_jk:
                    # Found a monochromatic triangle
                    # Change one of the edges
                    # Pick random edge
                    edge_idx = np.random.randint(0, 3)
                    new_color = (c_ij + 1) % 4 # or random
                    # Actually, just pick a random different color
                    new_color = random.choice([0,1,2,3])
                    while new_color == c_ij:
                        new_color = random.choice([0,1,2,3])
                    
                    if edge_idx == 0:
                        c[i, j] = new_color
                        c[j, i] = new_color
                        colors[get_idx(i, j)] = new_color
                    elif edge_idx == 1:
                        c[i, k] = new_color
                        c[k, i] = new_color
                        colors[get_idx(i, k)] = new_color
                    else:
                        c[j, k] = new_color
                        c[k, j] = new_color
                        colors[get_idx(j, k)] = new_color
                else:
                    # No triangle found in this sample?
                    # We need to be sure. 
                    # For a robust check, we should verify the whole graph occasionally.
                    if it % 100 == 0:
                        # Full check
                        is_valid = True
                        for a in range(target_n):
                            for b in range(a+1, target_n):
                                ca = c[a, b]
                                # Check all k
                                # This is O(N^3) = 125,000 ops. Fast enough in numpy?
                                # Let's do it efficiently.
                                # c[a, :] == ca and c[b, :] == ca
                                # Intersection of neighbors
                                # If there is any k such that c[a,k]==ca and c[b,k]==ca, then triangle.
                                # mask_a = (c[a, :] == ca)
                                # mask_b = (c[b, :] == ca)
                                # if np.any(mask_a & mask_b): triangle
                                # Exclude a and b themselves
                                mask_a = (c[a, :] == ca)
                                mask_a[a] = False
                                mask_b = (c[b, :] == ca)
                                mask_b[b] = False
                                if np.any(mask_a & mask_b):
                                    is_valid = False
                                    break
                            if not is_valid: break
                        if is_valid:
                            # Found valid coloring for n=50
                            best_n = target_n
                            best_colors = colors.flatten().tolist()
                            # Flatten in lexicographic order
                            final_colors = []
                            for i in range(target_n):
                                for j in range(i+1, target_n):
                                    final_colors.append(int(c[i, j]))
                            return {'n': target_n, 'colors': final_colors}
            # End local search for this restart

    # If we didn't find n=50, try to extend n=32 to n=33, 34... using backtracking?
    # Or just return n=32.
    # Given time, returning n=32 is safe.
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
