"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Precompute edge position lookup
edge_pos = {}
for i, (a, b) in enumerate(EDGE_ORDER):
    edge_pos[(a, b)] = i + 1  # 1-based
    edge_pos[(b, a)] = i + 1

def get_cost(perm):
    """Compute the max edge position (score) for a Hamilton cycle."""
    max_pos = 0
    n = len(perm)
    for i in range(n):
        a = perm[i]
        b = perm[(i + 1) % n]
        pos = edge_pos[(a, b)]
        if pos > max_pos:
            max_pos = pos
    return max_pos

def get_cost_detail(perm):
    """Compute edge positions for all edges in the cycle."""
    positions = []
    n = len(perm)
    for i in range(n):
        a = perm[i]
        b = perm[(i + 1) % n]
        positions.append(edge_pos[(a, b)])
    return positions

def simulated_annealing(perm, time_limit=60.0, rng=None):
    """Simulated annealing with 2-opt moves to minimize max edge index."""
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    current = list(perm)
    current_cost = get_cost(current)
    best = list(current)
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() - start_time < time_limit - 2.0:
        # 2-opt move: pick two positions and reverse the segment
        i = rng.randint(0, N - 1)
        j = rng.randint(0, N - 1)
        if i > j:
            i, j = j, i
        if j - i < 2:  # Need at least 2 edges to reverse
            continue
        
        # Apply 2-opt
        new_perm = current[:i] + current[i:j+1][::-1] + current[j+1:]
        
        # Compute new cost
        new_cost = get_cost(new_perm)
        
        # Accept or reject
        delta = new_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current = new_perm
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best = list(current)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_cost

def greedy_construct():
    """Greedy construction: build cycle edge by edge preferring early edges."""
    # Start with a random permutation
    rng = random.Random(123)
    perm = list(range(N))
    rng.shuffle(perm)
    return perm

def local_search(perm, time_limit=60.0, rng=None):
    """Local search with 2-opt and swap moves."""
    if rng is None:
        rng = random.Random(99)
    
    start_time = time.time()
    current = list(perm)
    current_cost = get_cost(current)
    best = list(current)
    best_cost = current_cost
    
    while time.time() - start_time < time_limit - 2.0:
        improved = False
        
        # Try 2-opt moves
        for i in range(N - 1):
            for j in range(i + 2, N):
                new_perm = current[:i] + current[i:j+1][::-1] + current[j+1:]
                new_cost = get_cost(new_perm)
                if new_cost < current_cost:
                    current = new_perm
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best = list(current)
                    improved = True
                    break
            if improved:
                break
        
        # Try swap moves
        if not improved:
            for i in range(N):
                for j in range(i + 1, N):
                    new_perm = list(current)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_cost = get_cost(new_perm)
                    if new_cost < current_cost:
                        current = new_perm
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best = list(current)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Stuck in local minimum, do a random perturbation
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i > j:
                i, j = j, i
            if j - i >= 2:
                current = current[:i] + current[i:j+1][::-1] + current[j+1:]
                current_cost = get_cost(current)
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    best_perm = list(range(N))
    best_cost = get_cost(best_perm)
    
    # Multiple restarts with different strategies
    rng_master = random.Random(614)
    
    # First: simulated annealing from identity
    t0 = time.time()
    perm1, cost1 = simulated_annealing(list(range(N)), time_limit=min(30.0, deadline - (time.time() - start_time) - 2.0), rng=random.Random(1))
    if cost1 < best_cost:
        best_cost = cost1
        best_perm = perm1
    
    # Second: simulated annealing from random
    if time.time() - start_time < deadline - 2.0:
        perm_init = list(range(N))
        random.Random(2).shuffle(perm_init)
        perm2, cost2 = simulated_annealing(perm_init, time_limit=min(30.0, deadline - (time.time() - start_time) - 2.0), rng=random.Random(2))
        if cost2 < best_cost:
            best_cost = cost2
            best_perm = perm2
    
    # Third: local search from best so far
    if time.time() - start_time < deadline - 2.0:
        perm3, cost3 = local_search(best_perm, time_limit=min(20.0, deadline - (time.time() - start_time) - 2.0), rng=random.Random(3))
        if cost3 < best_cost:
            best_cost = cost3
            best_perm = perm3
    
    # Fourth: more SA from best
    if time.time() - start_time < deadline - 2.0:
        perm4, cost4 = simulated_annealing(best_perm, time_limit=min(20.0, deadline - (time.time() - start_time) - 2.0), rng=random.Random(4))
        if cost4 < best_cost:
            best_cost = cost4
            best_perm = perm4
    
    # Final polish: local search from best
    if time.time() - start_time < deadline - 1.0:
        perm5, cost5 = local_search(best_perm, time_limit=max(1.0, deadline - (time.time() - start_time) - 1.0), rng=random.Random(5))
        if cost5 < best_cost:
            best_cost = cost5
            best_perm = perm5
    
    return best_perm
# EVOLVE-BLOCK-END
