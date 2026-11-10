import random
import time
import itertools

N = 16
NUM_PAIRS = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    idx = 0
    for k in range(i):
        idx += (N - 1 - k)
    idx += (j - 1 - i)
    return idx

# Precompute all 4-cycles and their edge indices
all_cycles = []
edge_to_cycles = [[] for _ in range(NUM_PAIRS)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    c1 = [edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a)]
    c2 = [edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a)]
    c3 = [edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a)]
    
    for cyc in [c1, c2, c3]:
        all_cycles.append(cyc)
        for e in cyc:
            edge_to_cycles[e].append(len(all_cycles) - 1)

# Precompute edge_others for fast validity check
edge_others = [[] for _ in range(NUM_PAIRS)]
for cycle in all_cycles:
    for i in range(4):
        e = cycle[i]
        others = [cycle[j] for j in range(4) if j != i]
        edge_others[e].append(others)

def is_valid_fast(colors, e, new_color):
    for others in edge_others[e]:
        c1 = colors[others[0]]
        c2 = colors[others[1]]
        c3 = colors[others[2]]
        if c1 != c2 and c1 != c3 and c2 != c3:
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_colors = [0] * NUM_PAIRS
    best_count = 1
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(42 + restart)
        
        colors = [0] * NUM_PAIRS
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_index(i, j)
                colors[idx] = (i + j) % 2
        
        color_counts = [0] * (MAX_COLOR + 1)
        for c in colors:
            color_counts[c] += 1
        
        used_colors_count = sum(1 for c in color_counts if c > 0)
        
        no_improvement = 0
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
                
            e = random.randint(0, NUM_PAIRS - 1)
            old_color = colors[e]
            
            if random.random() < 0.8:
                found_new = False
                for _ in range(10):
                    new_color = random.randint(0, MAX_COLOR)
                    if color_counts[new_color] == 0:
                        if is_valid_fast(colors, e, new_color):
                            color_counts[old_color] -= 1
                            if color_counts[old_color] == 0:
                                used_colors_count -= 1
                            color_counts[new_color] += 1
                            used_colors_count += 1
                            colors[e] = new_color
                            no_improvement = 0
                            found_new = True
                            break
                if found_new:
                    continue
            else:
                new_color = random.randint(0, MAX_COLOR)
                if new_color == old_color:
                    continue
                if is_valid_fast(colors, e, new_color):
                    color_counts[old_color] -= 1
                    if color_counts[old_color] == 0:
                        used_colors_count -= 1
                    color_counts[new_color] += 1
                    if color_counts[new_color] == 1:
                        used_colors_count += 1
                    colors[e] = new_color
                else:
                    no_improvement += 1
            
            if no_improvement > 5000:
                break
        
        current_count = used_colors_count
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if best_count > 60:
            break

    return best_colors
