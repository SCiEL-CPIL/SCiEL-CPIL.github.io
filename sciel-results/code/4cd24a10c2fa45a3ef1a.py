"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge index mapping
def build_edge_map():
    emap = {}
    idx = 0
    for a in range(N):
        for b in range(a + 1, N):
            emap[(a, b)] = idx
            idx += 1
    return emap

EDGE_MAP = build_edge_map()

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return EDGE_MAP[(i, j)]

# Precompute all 4-cycles (3 per 4-vertex subset)
def precompute_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles on {a,b,c,d}
                    e1 = (get_edge_idx(a,b), get_edge_idx(b,c), get_edge_idx(c,d), get_edge_idx(d,a))
                    e2 = (get_edge_idx(a,b), get_edge_idx(b,d), get_edge_idx(d,c), get_edge_idx(c,a))
                    e3 = (get_edge_idx(a,c), get_edge_idx(c,b), get_edge_idx(b,d), get_edge_idx(d,a))
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

CYCLES = precompute_cycles()
NUM_CYCLES = len(CYCLES)

# For each edge, which cycles does it belong to?
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_TO_CYCLES[e].append(ci)

def count_violations(colors):
    v = 0
    for cycle in CYCLES:
        c = [colors[e] for e in cycle]
        if len(set(c)) == 4:
            v += 1
    return v

def count_distinct(colors):
    return len(set(colors))

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    random.seed(12345)
    
    # Initialize
    colors = [0] * NUM_EDGES
    best_colors = colors[:]
    best_num_colors = 1
    
    # SA parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Choose move type
        if random.random() < 0.4:
            # Try to introduce a new color
            used = set(colors)
            if len(used) >= MAX_COLOR + 1:
                new_color = random.randint(0, MAX_COLOR)
            else:
                available = [c for c in range(MAX_COLOR + 1) if c not in used]
                new_color = random.choice(available)
        else:
            # Random color change
            new_color = random.randint(0, MAX_COLOR)
        
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        if new_color == old_color:
            continue
        
        # Evaluate impact
        affected = EDGE_TO_CYCLES[e]
        delta_viol = 0
        for ci in affected:
            cyc = CYCLES[ci]
            others = [colors[ce] for ce in cyc if ce != e]
            old_d = len(set(others + [old_color]))
            new_d = len(set(others + [new_color]))
            old_v = 1 if old_d == 4 else 0
            new_v = 1 if new_d == 4 else 0
            delta_viol += (new_v - old_v)
        
        # Global distinct color change
        old_global = len(set(colors))
        colors[e] = new_color
        new_global = len(set(colors))
        delta_global = new_global - old_global
        
        # Acceptance
        if delta_viol < 0:
            accept = True
        elif delta_viol == 0:
            if delta_global >= 0:
                accept = True
            else:
                accept = (random.random() < math.exp(-0.5 / T))
        else:
            # Positive delta_viol: penalize
            if delta_global > 0:
                # Trade: more colors but more violations
                accept = (random.random() < math.exp(-delta_viol * 3.0 / T))
            else:
                accept = (random.random() < math.exp(-delta_viol * 5.0 / T))
        
        if accept:
            if new_global > best_num_colors:
                best_num_colors = new_global
                best_colors = colors[:]
        else:
            colors[e] = old_color
        
        T = max(T * cooling, T_min)
    
    # Final verification
    if count_violations(best_colors) == 0:
        return best_colors
    # Fallback
    return [0] * NUM_EDGES
# EVOLVE-BLOCK-END
