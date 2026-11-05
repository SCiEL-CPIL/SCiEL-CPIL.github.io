import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# Precompute edge indices
edge_indices = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_indices[(i, j)] = idx
        idx += 1

# Precompute all 4-cycles
all_cycles = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    # The three cycles on vertices a,b,c,d:
    # Cycle 1: a-b-c-d-a -> edges (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: a-b-d-c-a -> edges (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: a-c-b-d-a -> edges (a,c), (c,b), (b,d), (d,a)
    e1 = edge_indices[(min(a,b), max(a,b))]
    e2 = edge_indices[(min(b,c), max(b,c))]
    e3 = edge_indices[(min(c,d), max(c,d))]
    e4 = edge_indices[(min(d,a), max(d,a))]
    all_cycles.append((e1, e2, e3, e4))
    
    e1 = edge_indices[(min(a,b), max(a,b))]
    e2 = edge_indices[(min(b,d), max(b,d))]
    e3 = edge_indices[(min(c,d), max(c,d))]
    e4 = edge_indices[(min(c,a), max(c,a))]
    all_cycles.append((e1, e2, e3, e4))
    
    e1 = edge_indices[(min(a,c), max(a,c))]
    e2 = edge_indices[(min(b,c), max(b,c))]
    e3 = edge_indices[(min(b,d), max(b,d))]
    e4 = edge_indices[(min(d,a), max(d,a))]
    all_cycles.append((e1, e2, e3, e4))

NUM_CYCLES = len(all_cycles)
cycles_np = np.array(all_cycles, dtype=np.int32)

# For each edge, which cycles contain it
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(all_cycles):
    for e in cycle:
        edge_to_cycles[e].append(ci)

edge_cycle_indices = []
for e in range(NUM_EDGES):
    edge_cycle_indices.append(np.array(edge_to_cycles[e], dtype=np.int32))

def is_valid(colors_np):
    """Check if the coloring is valid (no rainbow 4-cycle)."""
    cycle_colors = colors_np[cycles_np]
    sorted_colors = np.sort(cycle_colors, axis=1)
    rainbow = (sorted_colors[:, 0] != sorted_colors[:, 1]) | \
              (sorted_colors[:, 1] != sorted_colors[:, 2]) | \
              (sorted_colors[:, 2] != sorted_colors[:, 3])
    return not np.any(rainbow)

def count_violations(colors_np):
    """Count the number of rainbow 4-cycles."""
    cycle_colors = colors_np[cycles_np]
    sorted_colors = np.sort(cycle_colors, axis=1)
    rainbow = (sorted_colors[:, 0] != sorted_colors[:, 1]) | \
              (sorted_colors[:, 1] != sorted_colors[:, 2]) | \
              (sorted_colors[:, 2] != sorted_colors[:, 3])
    return np.sum(rainbow)

def local_violation_count(colors_np, e_idx, new_color):
    """Count violations only in cycles containing edge e_idx."""
    cycle_indices = edge_cycle_indices[e_idx]
    if len(cycle_indices) == 0:
        return 0
    
    cycle_colors = colors_np[cycles_np[cycle_indices]].copy()
    
    # Replace the color of e_idx with new_color in each affected cycle
    for i in range(len(cycle_indices)):
        cycle = cycles_np[cycle_indices[i]]
        for j in range(4):
            if cycle[j] == e_idx:
                cycle_colors[i, j] = new_color
                break
    
    sorted_colors = np.sort(cycle_colors, axis=1)
    rainbow = (sorted_colors[:, 0] != sorted_colors[:, 1]) | \
              (sorted_colors[:, 1] != sorted_colors[:, 2]) | \
              (sorted_colors[:, 2] != sorted_colors[:, 3])
    return np.sum(rainbow)

def greedy_construct():
    """Greedy construction: assign colors one edge at a time, preferring new colors."""
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
    
    rng = random.Random(42)
    
    for e_idx in range(NUM_EDGES):
        # Try to find a color that doesn't create any violations
        best_color = None
        best_violations = float('inf')
        
        # First, try unused colors (to maximize diversity)
        unused_colors = np.where(color_counts == 0)[0]
        if len(unused_colors) > 0:
            # Shuffle for variety
            shuffled = unused_colors.copy()
            rng.shuffle(shuffled)
            for c in shuffled[:20]:  # Try up to 20 random unused colors
                v = local_violation_count(colors, e_idx, c)
                if v == 0:
                    best_color = c
                    break
                elif v < best_violations:
                    best_violations = v
                    best_color = c
        
        # If no good unused color found, try used colors
        if best_color is None or best_violations > 0:
            used_colors = np.where(color_counts > 0)[0]
            if len(used_colors) > 0:
                shuffled = used_colors.copy()
                rng.shuffle(shuffled)
                for c in shuffled:
                    v = local_violation_count(colors, e_idx, c)
                    if v == 0:
                        best_color = c
                        break
                    elif v < best_violations:
                        best_violations = v
                        best_color = c
        
        if best_color is None:
            best_color = 0
        
        colors[e_idx] = best_color
        color_counts[best_color] += 1
    
    return colors, color_counts

def simulated_annealing(colors, color_counts, start_time, deadline):
    """Simulated annealing to improve the coloring."""
    colors = colors.copy()
    color_counts = color_counts.copy()
    
    best_colors = colors.copy()
    best_score = int(np.sum(color_counts > 0))
    
    current_score = best_score
    temp = 100.0
    cooling = 0.9995
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        # Pick a random edge
        e_idx = rng.randint(0, NUM_EDGES - 1)
        
        # Decide which color to try - bias towards unused colors
        if rng.random() < 0.7:
            unused = np.where(color_counts == 0)[0]
            if len(unused) > 0:
                new_color = unused[rng.randint(0, len(unused) - 1)]
            else:
                new_color = rng.randint(0, MAX_COLOR - 1)
        else:
            new_color = rng.randint(0, MAX_COLOR - 1)
        
        if colors[e_idx] == new_color:
            continue
        
        old_color = colors[e_idx]
        
        v = local_violation_count(colors, e_idx, new_color)
        
        if v == 0:
            colors[e_idx] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            if color_counts[old_color] == 0:
                current_score -= 1
            if color_counts[new_color] == 1:
                current_score += 1
            
            if current_score > best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            if rng.random() < np.exp(-v / temp):
                colors[e_idx] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    return best_colors, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Greedy construction
    colors, color_counts = greedy_construct()
    
    # Simulated annealing
    best_colors, best_score = simulated_annealing(colors, color_counts, start_time, deadline)
    
    return [int(c) for c in best_colors]
