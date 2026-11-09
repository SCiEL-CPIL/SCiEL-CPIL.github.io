import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to construct n=26
    n = 26
    
    # Initialize colors
    # 0..24 are affine plane points (x,y) in F5x2
    # 25 is extra point
    colors = np.zeros((n, n), dtype=int)
    
    # Fill affine part with slope coloring
    for i in range(25):
        x1, y1 = i // 5, i % 5
        for j in range(i + 1, 25):
            x2, y2 = j // 5, j % 5
            if x1 == x2:
                c = 4 # vertical
            else:
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                # slope = dy * dx^-1
                # In F5, inverse of 1,2,3,4 is 1,3,2,4
                inv = [0, 1, 3, 2, 4]
                slope = (dy * inv[dx]) % 5
                c = slope
            colors[i, j] = c
            colors[j, i] = c
            
    # Fill extra point with random colors
    for i in range(25):
        c = random.randint(0, 4)
        colors[i, 25] = c
        colors[25, i] = c

    # Function to count C4s
    def count_c4s(col_matrix):
        total = 0
        for c in range(5):
            A = (col_matrix == c).astype(int)
            # Common neighbors for each pair
            # A^2 gives number of paths of length 2
            # Diagonal of A^2 is degree
            # Off-diagonal (u,v) is common neighbors
            A2 = np.dot(A, A)
            # For each u < v, add C(A2[u,v], 2)
            for u in range(n):
                for v in range(u + 1, n):
                    k = A2[u, v]
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    # Initial count
    current_c4s = count_c4s(colors)
    best_c4s = current_c4s
    best_colors = colors.copy()
    
    # Simulated Annealing
    T = 1000.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        
        old_c = colors[u, v]
        new_c = random.randint(0, 4)
        if old_c == new_c: continue
        
        # Calculate delta C4
        # This is expensive to do exactly without full recount in Python loops
        # For simplicity and speed, we will just recount if n is small enough
        # n=26, recount is 5 * (26*25/2) * small ops.
        # 5 * 325 * 20 ops = 32500 ops.
        # 10000 iterations * 32500 = 325M ops. Might be too slow in Python.
        
        # Optimized recount:
        # We can update A2 incrementally?
        # Or just use numpy for the recount, which is fast.
        
        colors[u, v] = new_c
        colors[v, u] = new_c
        
        new_c4s = count_c4s(colors)
        delta = new_c4s - current_c4s
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_c4s = new_c4s
            if current_c4s < best_c4s:
                best_c4s = current_c4s
                best_colors = colors.copy()
                if best_c4s == 0:
                    # Found valid coloring
                    break
        else:
            colors[u, v] = old_c
            colors[v, u] = old_c
            
        T *= cooling
        if T < 1: T = 1
        
    # If we found a valid coloring for n=26, return it
    if best_c4s == 0:
        # Flatten colors to list
        # Lexicographic order (i,j) i<j
        res_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                res_colors.append(int(best_colors[i, j]))
        return {'n': n, 'colors': res_colors}
    else:
        # If 26 failed, try smaller n?
        # The seed is 11. We can just return the best we have if it's valid,
        # but best_c4s > 0 means it's invalid.
        # We should fallback to a known valid small n.
        # Let's return the n=11 seed result to ensure validity and score 1.0
        n_seed = 11
        colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
        return {'n': n_seed, 'colors': colors_seed}
