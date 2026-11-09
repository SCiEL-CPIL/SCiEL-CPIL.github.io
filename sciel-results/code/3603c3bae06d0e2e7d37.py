"""Simulated Annealing with 2-opt moves to minimize the max edge arrival index of a Hamilton cycle."""
import random
import time
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
# Build edge position lookup: edge_pos[(u,v)] = 1-based index in edge_order
edge_pos = {}
for idx, (u, v) in enumerate(P['edge_order'], 1):
    edge_pos[(min(u,v), max(u,v))] = idx

def get_pos(u, v):
    a, b = (u, v) if u < v else (v, u)
    return edge_pos[(a, b)]

def calc_cost(cycle):
    """Return the max 1-based arrival index of edges in the Hamilton cycle."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i+1) % n]
        idx = get_pos(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def two_opt(cycle, i, j):
    """Apply a 2-opt move: reverse segment from i to j (inclusive).
    Returns a new cycle (list)."""
    if i == j:
        return cycle[:]
    if i > j:
        i, j = j, i
    n = len(cycle)
    # Reverse the segment [i..j]
    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
    return new_cycle

def random_two_opt(cycle, rng):
    """Pick a random 2-opt move."""
    n = len(cycle)
    i = rng.randint(0, n-1)
    j = rng.randint(0, n-1)
    return two_opt(cycle, i, j)

def simulated_annealing(initial_cycle, rng, time_budget):
    """Run simulated annealing with 2-opt moves."""
    current = initial_cycle[:]
    current_cost = calc_cost(current)
    best = current[:]
    best_cost = current_cost
    
    T_start = 50.0
    T_end = 0.01
    # We'll estimate number of iterations and set cooling schedule
    # Start with a reasonable number; we'll check time in the loop
    t0 = time.time()
    
    iteration = 0
    total_iters = 500000  # will be cut by time budget
    
    while iteration < total_iters:
        elapsed = time.time() - t0
        if elapsed > time_budget:
            break
        
        # Generate a 2-opt neighbor
        neighbor = random_two_opt(current, rng)
        neighbor_cost = calc_cost(neighbor)
        
        # Cooling schedule
        progress = iteration / total_iters
        T = T_start * math.exp(-10.0 * progress)
        
        delta = neighbor_cost - current_cost
        if delta < 0:
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        else:
            if T > 0:
                accept_prob = math.exp(-delta / T)
                if rng.random() < accept_prob:
                    current = neighbor
                    current_cost = neighbor_cost
        
        iteration += 1
    
    return best, best_cost

def greedy_local_search(cycle, rng):
    """Greedy hill climbing with 2-opt until no improvement."""
    current = cycle[:]
    current_cost = calc_cost(current)
    improved = True
    n = len(current)
    while improved:
        improved = False
        # Try all 2-opt moves
        for i in range(n):
            for j in range(i+1, n):
                neighbor = two_opt(current, i, j)
                nc = calc_cost(neighbor)
                if nc < current_cost:
                    current = neighbor
                    current_cost = nc
                    improved = True
                    break
            if improved:
                break
    return current, current_cost

def construct():
    deadline = 90.0
    t_start = time.time()
    
    rng = random.Random(614)
    
    # Start with identity cycle
    initial = list(range(N))
    
    # First do a greedy local search to get a good starting point
    current, current_cost = greedy_local_search(initial, rng)
    best = current[:]
    best_cost = current_cost
    
    # Run simulated annealing with multiple restarts
    restart_count = 0
    while time.time() - t_start < deadline - 5.0:  # keep 5s safety margin
        # Run SA from current best (or a perturbed version)
        if restart_count > 0:
            # Perturb the best solution
            perturbed = best[:]
            for _ in range(rng.randint(2, 5)):
                p = random_two_opt(perturbed, rng)
                perturbed = p
            sa_start = perturbed
        else:
            sa_start = current[:]
        
        # Time budget for this SA run
        elapsed = time.time() - t_start
        remaining = deadline - 5.0 - elapsed
        if remaining < 2.0:
            break
        sa_budget = min(remaining * 0.5, 10.0)  # each SA run gets up to 10s
        
        sa_result, sa_cost = simulated_annealing(sa_start, rng, sa_budget)
        
        if sa_cost < best_cost:
            best = sa_result[:]
            best_cost = sa_cost
            current = sa_result[:]
        
        restart_count += 1
    
    # Final greedy polish
    if time.time() - t_start < deadline - 1.0:
        final, final_cost = greedy_local_search(best, rng)
        if final_cost < best_cost:
            best = final
            best_cost = final_cost
    
    return best
# EVOLVE-BLOCK-END
