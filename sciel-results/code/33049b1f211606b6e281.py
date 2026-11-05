import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = [edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(a,d)]
                    cycles.append(e1)
                    e2 = [edge_index(a,b), edge_index(b,d), edge_index(c,d), edge_index(a,c)]
                    cycles.append(e2)
                    e3 = [edge_index(a,c), edge_index(b,c), edge_index(b,d), edge_index(a,d)]
                    cycles.append(e3)
    return cycles

FOUR_CYCLES = build_four_cycles()

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    colors = [0] * NUM_EDGES
    best_colors = colors[:]
    best_count = 1
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for idx, cycle in enumerate(FOUR_CYCLES):
        for e in cycle:
            edge_to_cycles[e].append(idx)
    
    random.seed(42)
    
    for iteration in range(5000000):
        if time.time() > deadline:
            break
        
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        used_colors = set(colors)
        candidates = []
        for c in range(120):
            if c not in used_colors:
                candidates.append(c)
        candidates.extend([random.randint(0, 119) for _ in range(3)])
        random.shuffle(candidates)
        
        improved = False
        for new_color in candidates:
            if new_color == old_color:
                continue
            
            valid = True
            for cyc_idx in edge_to_cycles[e]:
                cycle = FOUR_CYCLES[cyc_idx]
                c_set = set()
                for ee in cycle:
                    if ee == e:
                        c_set.add(new_color)
                    else:
                        c_set.add(colors[ee])
                if len(c_set) == 4:
                    valid = False
                    break
            
            if valid:
                colors[e] = new_color
                new_count = len(set(colors))
                if new_count > best_count:
                    best_count = new_count
                    best_colors = colors[:]
                improved = True
                break
    
    return best_colors
