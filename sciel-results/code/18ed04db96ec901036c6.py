import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Seed fallback
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best_n = seed_n
    best_colors = seed_colors

    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
        
        time_alloc = 5.0
        if n <= 13:
            time_alloc = 10.0
        if n <= 11:
            time_alloc = 5.0
            
        target_end = time.time() + time_alloc
        
        found = solve_n(n, target_end)
        if found:
            best_n = n
            best_colors = found
            if n == 17:
                break

    return {'n': best_n, 'colors': best_colors}

def solve_n(n, deadline):
    vertices = list(range(n))
    
    def calc_index(i, j):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Initialize random
    col_2d = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, 3)
            col_2d[i][j] = c
            col_2d[j][i] = c

    def delta_cost(u, v, new_color):
        old_color = col_2d[u][v]
        if old_color == new_color:
            return 0
        delta = 0
        others = [x for x in range(n) if x != u and x != v]
        for combo in combinations(others, 4):
            for perm in permutations(combo):
                w, x, y, z = perm
                c1 = col_2d[v][w]
                c2 = col_2d[w][x]
                c3 = col_2d[x][y]
                c4 = col_2d[y][z]
                c5 = col_2d[z][u]
                if c1 == c2 == c3 == c4 == c5:
                    c_other = c1
                    if old_color == c_other:
                        delta -= 1
                    if new_color == c_other:
                        delta += 1
        return delta

    def is_valid_full():
        # Check all 6-subsets
        for combo in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = combo
            # Check if any color forms a C6
            # We can check the 60 cycles.
            # To speed up, we can just check if for any c, 
            # the edges of a specific cycle are all c.
            # Let's iterate permutations of 1..5 (fixing v0)
            # There are 120 permutations.
            # Each corresponds to a cycle v0-v1-...-v5-v0.
            # But (v0..v5) and (v0..v5 reversed) are same.
            # So 60 unique cycles.
            for perm in permutations([1, 2, 3, 4, 5]):
                # Cycle: v0 - v[perm[0]] - v[perm[1]] - v[perm[2]] - v[perm[3]] - v[perm[4]] - v0
                # Edges:
                # (v0, v[perm[0]]), (v[perm[0]], v[perm[1]]), ..., (v[perm[4]], v0)
                # Check if all same color
                c0 = col_2d[v0][combo[perm[0]]]
                if c0 != col_2d[combo[perm[0]]][combo[perm[1]]]: continue
                if c0 != col_2d[combo[perm[1]]][combo[perm[2]]]: continue
                if c0 != col_2d[combo[perm[2]]][combo[perm[3]]]: continue
                if c0 != col_2d[combo[perm[3]]][combo[perm[4]]]: continue
                if c0 != col_2d[combo[perm[4]]][v0]: continue
                # Found a mono C6
                return False
        return True

    # Local Search
    for _ in range(5000):
        if time.time() > deadline:
            break
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        old_color = col_2d[u][v]
        new_color = random.choice([c for c in range(4) if c != old_color])
        
        d = delta_cost(u, v, new_color)
        
        if d < 0:
            col_2d[u][v] = new_color
            col_2d[v][u] = new_color

    if is_valid_full():
        flat_colors = []
        for i in range(n):
            for j in range(i+1, n):
                flat_colors.append(col_2d[i][j])
        return flat_colors
    return None
