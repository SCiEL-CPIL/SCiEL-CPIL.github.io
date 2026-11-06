import time
import random
import math
from collections import defaultdict

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles and for each edge, which cycles contain it
CYCLES = []
edge_to_cycles = defaultdict(list)

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
                CYCLES.append(e1)
                for e in e1:
                    edge_to_cycles[e].append(len(CYCLES) - 1)
                
                e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
                CYCLES.append(e2)
                for e in e2:
                    edge_to_cycles[e].append(len(CYCLES) - 1)
                
                e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))
                CYCLES.append(e3)
                for e in e3:
                    edge_to_cycles[e].append(len(CYCLES) - 1)

NUM_CYCLES = len(CYCLES)

def check_cycle_valid(colors, cycle):
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    return not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)

def check_affected_valid(colors, edge):
    for ci in edge_to_cycles[edge]:
        if not check_cycle_valid(colors, CYCLES[ci]):
            return False
    return True

def count_distinct_fast(colors):
    return len(set(colors))

def is_fully_valid(colors):
    for c in CYCLES:
        if not check_cycle_valid(colors, c):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_colors = None
    best_score = 0
    
    # Try multiple initial structures
    candidates = []
    
    # Structure 1: color = i
    c1 = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            c1[idx] = i
            idx += 1
    if is_fully_valid(c1):
        candidates.append(c1)
    
    # Structure 2: color = (i + j) % k for various k
    for k in range(2, 25):
        c = [0] * NUM_EDGES
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c[idx] = (i + j) % k
                idx += 1
        if is_fully_valid(c):
            candidates.append(c)
    
    # Structure 3: color = (i * j) % k
    for k in range(2, 25):
        c = [0] * NUM_EDGES
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c[idx] = (i * j) % k
                idx += 1
        if is_fully_valid(c):
            candidates.append(c)
    
    # Structure 4: color = (i - j) % k (but i<j so j-i)
    for k in range(2, 25):
        c = [0] * NUM_EDGES
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c[idx] = (j - i) % k
                idx += 1
        if is_fully_valid(c):
            candidates.append(c)
    
    # Pick best candidate
    for c in candidates:
        s = count_distinct_fast(c)
        if s > best_score:
            best_score = s
            best_colors = c[:]
    
    if best_colors is None:
        # Fallback
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    current_colors = best_colors[:]
    current_score = best_score
    used_colors = set(current_colors)
    
    # Local search / simulated annealing
    T = 3.0
    T_min = 0.01
    alpha = 0.9998
    
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # 70% chance try existing color, 30% try new
        if random.random() < 0.3 and len(used_colors) < 119:
            available = set(range(120)) - used_colors
            if available:
                new_color = random.choice(list(available))
            else:
                new_color = random.randint(0, 119)
        else:
            new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        current_colors[e] = new_color
        
        if check_affected_valid(current_colors, e):
            new_score = count_distinct_fast(current_colors)
            delta = new_score - current_score
            
            if new_score > best_score:
                best_score = new_score
                best_colors = current_colors[:]
            
            if delta >= 0:
                current_score = new_score
                used_colors = set(current_colors)
            else:
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                    used_colors = set(current_colors)
                else:
                    current_colors[e] = old_color
        else:
            current_colors[e] = old_color
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_colors
