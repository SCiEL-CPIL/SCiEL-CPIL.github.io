import random
import time
import itertools
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def precompute_4cycles():
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        cycles.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
    return cycles

CYCLES = precompute_4cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci in range(NUM_CYCLES):
    for e in CYCLES[ci]:
        EDGE_CYCLES[e].append(ci)

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    for ci in range(NUM_CYCLES):
        e0, e1, e2, e3 = CYCLES[ci]
        c0, c1, c2, c3 = coloring[e0], coloring[e1], coloring[e2], coloring[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def check_cycles_for_edge(coloring, e):
    for ci in EDGE_CYCLES[e]:
        e0, e1, e2, e3 = CYCLES[ci]
        c0, c1, c2, c3 = coloring[e0], coloring[e1], coloring[e2], coloring[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Initial: block structure (4 groups of 4)
    groups = [i // 4 for i in range(N)]
    coloring = [0] * NUM_EDGES
    color_map = {}
    next_color = 1
    for i in range(N):
        for j in range(i + 1, N):
            gi, gj = groups[i], groups[j]
            if gi == gj:
                coloring[edge_idx(i, j)] = 0
            else:
                key = (min(gi, gj), max(gi, gj))
                if key not in color_map:
                    color_map[key] = next_color
                    next_color += 1
                coloring[edge_idx(i, j)] = color_map[key]
    
    best_coloring = coloring[:]
    best_score = count_colors(coloring)
    
    # SA with recolor moves
    current = coloring[:]
    current_score = best_score
    T = 5.0
    cooling = 0.99999
    min_T = 0.01
    
    random.seed(42)
    color_counts = Counter(current)
    
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current[e]
        new_color = random.randint(0, 119)
        if old_color == new_color:
            continue
        
        current[e] = new_color
        valid = check_cycles_for_edge(current, e)
        
        if valid:
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                del color_counts[old_color]
            color_counts[new_color] = color_counts.get(new_color, 0) + 1
            new_score = len(color_counts)
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_coloring = current[:]
            elif delta == 0:
                # Keep with some probability to explore
                if random.random() > 0.5:
                    current[e] = old_color
                    color_counts[old_color] += 1
                    color_counts[new_color] -= 1
                    if color_counts[new_color] == 0:
                        del color_counts[new_color]
            else:  # delta < 0
                # Accept with prob exp(delta/T)
                import math
                if random.random() < math.exp(delta / max(T, 1e-10)):
                    current_score = new_score
                else:
                    current[e] = old_color
                    color_counts[old_color] += 1
                    color_counts[new_color] -= 1
                    if color_counts[new_color] == 0:
                        del color_counts[new_color]
        else:
            current[e] = old_color
        
        T *= cooling
        if T < min_T:
            break
    
    return best_coloring
