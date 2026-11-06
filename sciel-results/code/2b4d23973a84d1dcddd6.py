"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import copy

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

# Precompute edges and cycles
edge_idx = {}
edges = []
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[(i, j)] = len(edges)
        edges.append((i, j))

all_cycles = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = edge_idx[(a,b)]
                e2 = edge_idx[(b,c)]
                e3 = edge_idx[(c,d)]
                e4 = edge_idx[(d,a)]
                all_cycles.append((e1, e2, e3, e4))
                
                # Cycle 2: a-b-d-c-a
                e1 = edge_idx[(a,b)]
                e2 = edge_idx[(b,d)]
                e3 = edge_idx[(d,c)]
                e4 = edge_idx[(c,a)]
                all_cycles.append((e1, e2, e3, e4))
                
                # Cycle 3: a-c-b-d-a
                e1 = edge_idx[(a,c)]
                e2 = edge_idx[(c,b)]
                e3 = edge_idx[(b,d)]
                e4 = edge_idx[(d,a)]
                all_cycles.append((e1, e2, e3, e4))

# Precompute cycle info for each edge
cycle_info_for_edge = [[] for _ in range(NUM_EDGES)]
for idx, cyc in enumerate(all_cycles):
    for k, e in enumerate(cyc):
        others = [cyc[j] for j in range(4) if j != k]
        cycle_info_for_edge[e].append((idx, k, others))

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # State
    colors = [0] * NUM_EDGES
    freq = [0] * (MAX_COLOR + 1)
    freq[0] = NUM_EDGES
    distinct_colors = 1
    violation_count = 0
    
    best_colors = colors.copy()
    best_distinct = 1
    
    def get_violations_for_edge_change(edge_idx, old_color, new_color):
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        delta_viol = 0
        for idx, pos, others in cycle_info_for_edge[edge_idx]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            old_distinct = (old_color != c1 and old_color != c2 and old_color != c3 and 
                            c1 != c2 and c1 != c3 and c2 != c3)
            
            new_distinct = (new_color != c1 and new_color != c2 and new_color != c3 and 
                            c1 != c2 and c1 != c3 and c2 != c3)
            
            if old_distinct and not new_distinct:
                delta_viol -= 1
            elif not old_distinct and new_distinct:
                delta_viol += 1
                
        return delta_viol, delta_distinct

    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        edge_idx = random.randint(0, NUM_EDGES - 1)
        new_color = random.randint(0, MAX_COLOR)
        
        if new_color == colors[edge_idx]:
            continue
            
        old_color = colors[edge_idx]
        
        delta_viol, delta_distinct = get_violations_for_edge_change(edge_idx, old_color, new_color)
        
        delta_E = 100 * delta_viol - delta_distinct
        
        accept = False
        if delta_E < 0:
            accept = True
        else:
            if T > 0:
                if random.random() < pow(2.718, -delta_E / T):
                    accept = True
        
        if accept:
            colors[edge_idx] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            violation_count += delta_viol
            distinct_colors += delta_distinct
            
            if violation_count == 0 and distinct_colors > best_distinct:
                best_distinct = distinct_colors
                best_colors = colors.copy()
                
        T *= cooling
        if T < min_T:
            T = min_T

    return best_colors
# EVOLVE-BLOCK-END
