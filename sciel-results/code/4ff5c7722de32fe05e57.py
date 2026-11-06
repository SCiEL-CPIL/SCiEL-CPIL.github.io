import random
import time
from itertools import combinations

def construct():
    # ... (keep helper functions) ...
    
    def check_valid(edge_colors):
        """Check every K4 in K16 has at least 5 distinct colors."""
        for quad in combinations(range(16), 4):
            a, b, c, d = quad
            cols = {
                edge_colors[a][b], edge_colors[a][c], edge_colors[a][d],
                edge_colors[b][c], edge_colors[b][d], edge_colors[c][d]
            }
            if len(cols) < 5:
                return False
        return True

    def count_colors(edge_colors):
        return len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))

    best_result = None
    best_color_count = 120 # Max possible

    # 1. Deterministic checks
    for p in [17, 19, 23, 29]:
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                c = ((i+1)*(j+1)) % p
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        if check_valid(edge_colors):
            nc = count_colors(edge_colors)
            if nc < best_color_count:
                best_color_count = nc
                best_result = [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]

    # 2. Random search for p=19 and p=23 to potentially find < 16 colors
    # We assign random distinct non-zero residues to vertices
    for p in [19, 23]:
        start_time = time.time()
        attempts = 0
        while time.time() - start_time < 10: # 10 seconds per prime
            attempts += 1
            # Pick 16 distinct non-zero elements
            vals = random.sample(range(1, p), 16)
            edge_colors = [[0]*16 for _ in range(16)]
            for i in range(16):
                for j in range(i+1, 16):
                    c = (vals[i] * vals[j]) % p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
            
            # Quick check: count colors first
            nc = count_colors(edge_colors)
            if nc >= best_color_count:
                continue
                
            # If color count is promising, check validity
            if check_valid(edge_colors):
                best_color_count = nc
                best_result = [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]
                if best_color_count <= 13: # Theoretical min
                    break

    if best_result:
        return best_result
    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
