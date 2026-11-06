import time
import random
import itertools

def construct():
    # Target n
    max_n = 51
    deadline = time.time() + 100  # 100 seconds limit
    
    # Helper to check if a coloring is valid
    # colors is a list of length n*(n-1)//2
    # Index for (i, j) with i < j: i*(n-1) - i*(i+1)//2 + (j-i-1)
    # Actually, standard lexicographic order:
    # (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2, n-1)
    # Index formula: idx(i, j) = i*n - i*(i+1)//2 + (j-i-1) ?
    # Let's derive:
    # Row 0: n-1 edges
    # Row 1: n-2 edges
    # ...
    # Row i: n-1-i edges
    # Sum of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2
    # In row i, edge (i, j) is the (j-i-1)-th edge.
    # So idx = i(n-1) - i(i-1)/2 + (j-i-1)
    
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_mono_triangles(n, colors):
        # Efficient check using bitsets or just iteration
        # For each triangle (i, j, k) with i<j<k, check if colors are equal
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[get_idx(i, j, n)]
                for k in range(j + 1, n):
                    if colors[get_idx(i, k, n)] == c_ij and colors[get_idx(j, k, n)] == c_ij:
                        count += 1
        return count

    def try_find_coloring(n, time_limit):
        if time.time() > time_limit:
            return None
        
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Local search
        # Objective: minimize count_mono_triangles
        # Move: change color of one edge
        
        # Precompute neighbors for faster triangle count update?
        # For now, simple re-evaluation is O(n^3). For n=51, n^3 ~ 132k.
        # 100s / 132k ~ 750 evaluations/sec. This is too slow for Python.
        
        # We need a faster check.
        # Represent colors as a 2D list or flat list.
        # We can maintain the count incrementally.
        
        # Let's use a different approach: 
        # For each vertex v and color c, maintain the set of neighbors N_c(v).
        # A triangle (u, v, w) is monochromatic in color c if u, w in N_c(v) and (u,w) is color c.
        # This is still complex to update incrementally.
        
        # Alternative: Use a known valid coloring for n=51.
        # I will hardcode a valid coloring for n=51 if I can generate it.
        # Since I can't be 100% sure of the hardcoded values without verification, 
        # I will use a heuristic that is faster.
        
        # Actually, for n=51, a simple random search is unlikely to succeed in 100s.
        # I will use a pre-computed valid coloring for n=51.
        # I will generate it using a specific construction:
        # The 51-vertex coloring is based on the group Z_51? No.
        # It is based on the projective plane of order 4? No.
        
        # Let's use a known valid coloring for n=50 or 51 from the literature.
        # I will use the following construction for n=51:
        # Vertices 0..50.
        # Colors 0..3.
        # I will use a random restart local search with a better heuristic:
        # Minimize the number of "bad" pairs (edges in same color class that form a triangle).
        
        # Given the difficulty, I will return a known valid coloring for n=32 if I can't find better.
        # But I want to improve.
        # I will try n=40.
        
        pass

    # Fallback: Use the known valid coloring for n=32 from the seed program.
    # But let's try to find a better one.
    
    # I will use a simple backtracking for n=51 with strong pruning.
    # If it fails, return n=32.
    
    # Actually, there is a very simple construction for n=51 using 4 colors:
    # It is the "Clebsch graph" based one? No.
    
    # I will use the following:
    # A valid 4-coloring of K_51 is known to exist.
    # I will hardcode a valid coloring for n=51.
    # To avoid errors, I will generate a valid coloring for n=51 using a SAT solver logic in Python? No, too slow.
    
    # I will use a pre-computed valid coloring for n=51.
    # I will generate the colors using the following rule:
    # The coloring is symmetric under the group of order 51?
    
    # Let's use a known valid coloring for n=51.
    # I will use the one from the "Ramsey Numbers" website or similar.
    # Since I don't have access, I will use a heuristic that finds a valid coloring for n=51.
    # If it fails, I will use n=32.
    
    # To ensure a score > 1.0, I will use a known valid coloring for n=40.
    # I can construct a valid coloring for n=40 using 4 colors.
    # For example, using the doubling of a K_20 coloring?
    # R(3,3,3,3) > 40 is known.
    
    # I will use a random local search for n=51.
    # If it fails, I will use a random local search for n=40.
    # If it fails, I will use n=32.
    
    for target_n in [51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        # Try to find a valid coloring for target_n
        # Use a simple random restart local search
        best_count = float('inf')
        best_colors = None
        start_time = time.time()
        
        while time.time() - start_time < 10: # 10 seconds per n
            colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            count = count_mono_triangles(target_n, colors)
            if count == 0:
                return {'n': target_n, 'colors': colors}
            
            # Local search
            improved = True
            while improved and time.time() - start_time < 10:
                improved = False
                # Randomly pick an edge
                idx = random.randint(0, len(colors) - 1)
                old_color = colors[idx]
                # Try all other colors
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    colors[idx] = new_color
                    new_count = count_mono_triangles(target_n, colors)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count == 0:
                            return {'n': target_n, 'colors': colors}
                        break
                    else:
                        colors[idx] = old_color
        
        if best_count == 0:
            return {'n': target_n, 'colors': best_colors}

    # Fallback to n=32
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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
