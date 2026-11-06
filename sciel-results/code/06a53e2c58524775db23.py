"""Combined approach: structured initializations + vectorized SA + greedy polish."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
def precompute_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
        e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

ALL_CYCLES = precompute_cycles()
NUM_CYCLES = len(ALL_CYCLES)

# For each edge, which cycles contain it
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(ALL_CYCLES):
    for e in cycle:
        EDGE_TO_CYCLES[e].append(ci)

# Precompute cycle edge arrays as numpy for vectorized validation
CYCLE_EDGES = np.array(ALL_CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

def count_violations(coloring):
    """Count number of 4-cycles with 4 distinct colors using vectorized ops."""
    colors = np.array(coloring, dtype=np.int32)
    cycle_colors = colors[CYCLE_EDGES]  # shape (NUM_CYCLES, 4)
    # Check if all 4 are distinct
    # Sort each row and check for duplicates
    sorted_colors = np.sort(cycle_colors, axis=1)
    # A row has all distinct if no two adjacent elements are equal after sorting
    has_dup = np.any(sorted_colors[:, :-1] == sorted_colors[:, 1:], axis=1)
    return int(np.sum(~has_dup))

def count_colors(coloring):
    return len(set(coloring))

def check_edge_change_valid(coloring, edge, new_color):
    """Check if changing coloring[edge] to new_color would create any 4-distinct-color cycle."""
    old_color = coloring[edge]
    if old_color == new_color:
        return True
    
    for ci in EDGE_TO_CYCLES[edge]:
        e1, e2, e3, e4 = ALL_CYCLES[ci]
        other_edges = [e for e in (e1, e2, e3, e4) if e != edge]
        other_colors = {coloring[e] for e in other_edges}
        if len(other_colors) == 3 and new_color not in other_colors:
            return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    random.seed(12345)
    
    # Phase 1: Generate multiple initial structured colorings
    def generate_structured_coloring(seed):
        """Generate a valid coloring using a group structure."""
        rng = random.Random(seed)
        
        num_groups = rng.randint(3, 6)
        
        groups = list(range(N))
        for i in range(N):
            groups[i] = i % num_groups
        
        group_pair_colors = {}
        color = 1
        for g1 in range(num_groups):
            for g2 in range(g1 + 1, num_groups):
                group_pair_colors[(g1, g2)] = color
                color += 1
        
        coloring = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                g1, g2 = groups[i], groups[j]
                if g1 == g2:
                    coloring[edge_idx(i, j)] = 0
                else:
                    key = (min(g1, g2), max(g1, g2))
                    coloring[edge_idx(i, j)] = group_pair_colors.get(key, 0)
        
        return coloring
    
    def generate_modular_coloring(k, formula_type=0):
        """Generate a modular coloring."""
        coloring = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                if formula_type == 0:
                    coloring[idx] = (i - j) % k
                elif formula_type == 1:
                    coloring[idx] = (i + j) % k
                elif formula_type == 2:
                    coloring[idx] = (i * j) % k if k > 1 else 0
        return coloring
    
    # Try to find the best initial coloring
    best_initial = None
    best_initial_score = 0
    
    # Try structured colorings
    for seed in range(30):
        if time.time() > deadline - 60:
            break
        coloring = generate_structured_coloring(seed)
        if count_violations(coloring) == 0:
            score = count_colors(coloring)
            if score > best_initial_score:
                best_initial_score = score
                best_initial = coloring[:]
    
    # Try modular colorings for various k
    for k in range(10, 1, -1):
        if time.time() > deadline - 60:
            break
        for ft in range(3):
            coloring = generate_modular_coloring(k, ft)
            if count_violations(coloring) == 0:
                score = count_colors(coloring)
                if score > best_initial_score:
                    best_initial_score = score
                    best_initial = coloring[:]
    
    if best_initial is None:
        best_initial = [0] * NUM_EDGES
        best_initial_score = 1
    
    # Phase 2: Simulated annealing with fast validity checks
    coloring = best_initial[:]
    current_colors = set(coloring)
    current_score = best_initial_score
    best_coloring = coloring[:]
    best_score = current_score
    
    initial_temp = 10.0
    final_temp = 0.1
    cooling_rate = 0.9995
    
    max_iterations = 200000
    temp = initial_temp
    no_improve = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        e = random.randint(0, NUM_EDGES - 1)
        old_color = coloring[e]
        
        if random.random() < 0.3 and len(current_colors) < 119:
            available = [c for c in range(120) if c not in current_colors]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, 119)
        else:
            if current_colors:
                new_color = random.choice(list(current_colors))
            else:
                new_color = 0
        
        if new_color == old_color:
            continue
        
        if not check_edge_change_valid(coloring, e, new_color):
            continue
        
        coloring[e] = new_color
        
        if new_color not in current_colors:
            current_colors.add(new_color)
        if coloring.count(old_color) == 0:
            current_colors.discard(old_color)
        
        new_score = len(current_colors)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_score = new_score
            no_improve = 0
        else:
            coloring[e] = old_color
            if new_color not in current_colors:
                current_colors.add(new_color)
            if old_color not in current_colors:
                current_colors.discard(old_color)
            current_score = len(current_colors)
            no_improve += 1
        
        if current_score > best_score:
            best_score = current_score
            best_coloring = coloring[:]
            no_improve = 0
        
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
    
    # Phase 3: Greedy polish
    coloring = best_coloring[:]
    current_colors = set(coloring)
    current_score = len(current_colors)
    
    for e in range(NUM_EDGES):
        if time.time() > deadline - 3:
            break
        
        old_color = coloring[e]
        available = [c for c in range(120) if c not in current_colors]
        
        for new_color in available:
            if check_edge_change_valid(coloring, e, new_color):
                coloring[e] = new_color
                current_colors.add(new_color)
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_coloring = coloring[:]
                break
    
    return best_coloring
# EVOLVE-BLOCK-END
