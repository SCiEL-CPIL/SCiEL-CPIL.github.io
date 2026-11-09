import random
import time
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
# Build edge position lookup: pos[(u,v)] = 1-based index
pos = {}
for i, (u, v) in enumerate(P['edge_order']):
    pos[(u, v)] = i + 1
    pos[(v, u)] = i + 1

def get_pos(u, v):
    return pos[(u, v)]

def calc_cost(cycle):
    """Calculate the max edge index for a cycle (list of vertices in order, with implicit closing edge)."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_pos(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def two_opt(cycle):
    """Try all 2-opt moves, return best cycle and its cost."""
    n = len(cycle)
    best = list(cycle)
    best_cost = calc_cost(best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                # Reverse segment i+1..j
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                cost = calc_cost(new_cycle)
                if cost < best_cost:
                    best = new_cycle
                    best_cost = cost
                    improved = True
    return best, best_cost

def or_opt(cycle, k):
    """Try or-opt moves of length k."""
    n = len(cycle)
    best = list(cycle)
    best_cost = calc_cost(best)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n - k):
                if j == i or (j + k - 1 == i) or (j == i + k):
                    continue
                # Extract segment of length k starting at j
                seg = cycle[j:j+k]
                # Remove segment
                remaining = cycle[:j] + cycle[j+k:]
                # Try inserting seg at position i in remaining
                new_cycle = remaining[:i] + seg + remaining[i:]
                cost = calc_cost(new_cycle)
                if cost < best_cost:
                    best = new_cycle
                    best_cost = cost
                    improved = True
    return best, best_cost

def simulated_annealing(initial_cycle, t_start=100.0, t_end=0.01, n_iter=50000, seed=42):
    """Simulated annealing with 2-opt and or-opt moves."""
    rng = random.Random(seed)
    n = len(initial_cycle)
    current = list(initial_cycle)
    current_cost = calc_cost(current)
    best = list(current)
    best_cost = current_cost
    
    for i in range(n_iter):
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Choose a random move
        move_type = rng.random()
        if move_type < 0.5:
            # 2-opt
            i1 = rng.randint(0, n - 2)
            i2 = rng.randint(i1 + 1, n - 1)
            if i2 == n - 1 and i1 == 0:
                i2 = n - 2
            new_cycle = current[:i1+1] + current[i1+1:i2+1][::-1] + current[i2+1:]
        elif move_type < 0.8:
            # or-opt k=1
            i1 = rng.randint(0, n - 1)
            i2 = rng.randint(0, n - 2)
            if i2 == i1 or i2 == i1 - 1:
                i2 = (i2 + 1) % n
            seg = [current[i1]]
            remaining = current[:i1] + current[i1+1:]
            new_cycle = remaining[:i2] + seg + remaining[i2:]
        else:
            # or-opt k=2
            i1 = rng.randint(0, n - 2)
            i2 = rng.randint(0, n - 3)
            if i2 == i1 or i2 == i1 - 1 or i2 == i1 + 1:
                i2 = (i2 + 2) % n
            seg = current[i1:i1+2]
            remaining = current[:i1] + current[i1+2:]
            new_cycle = remaining[:i2] + seg + remaining[i2:]
        
        new_cost = calc_cost(new_cycle)
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
            current = new_cycle
            current_cost = new_cost
            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_cycle = list(range(N))
    best_score = calc_cost(best_cycle)
    
    # Multiple restarts with different seeds
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
    for seed in seeds:
        if time.time() > deadline:
            break
        # Random initial permutation
        rng = random.Random(seed)
        init = list(range(N))
        rng.shuffle(init)
        
        # Run simulated annealing
        cycle, cost = simulated_annealing(init, t_start=50.0, t_end=0.01, n_iter=30000, seed=seed)
        
        if cost < best_score:
            best_score = cost
            best_cycle = cycle
        
        # Polish with 2-opt
        cycle2, cost2 = two_opt(best_cycle)
        if cost2 < best_score:
            best_score = cost2
            best_cycle = cycle2
        
        if time.time() > deadline:
            break
    
    # Final polish
    if time.time() < deadline - 5:
        final_cycle, final_cost = two_opt(best_cycle)
        if final_cost < best_score:
            best_score = final_cost
            best_cycle = final_cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
