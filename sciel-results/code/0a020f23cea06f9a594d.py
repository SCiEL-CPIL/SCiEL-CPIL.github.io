import random
import time
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
# Precompute edge indices
edge_idx = {}
for i, (u, v) in enumerate(P['edge_order']):
    edge_idx[(u, v)] = i
    edge_idx[(v, u)] = i

def get_edge_index(u, v):
    return edge_idx[(u, v)]

def calc_cost(perm):
    max_idx = 0
    for i in range(n):
        u = perm[i]
        v = perm[(i + 1) % n]
        idx = get_edge_index(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def simulated_annealing(initial_perm, time_limit):
    current = list(initial_perm)
    current_cost = calc_cost(current)
    best = list(current)
    best_cost = current_cost
    
    T = 1000.0
    cool_rate = 0.995
    
    start_time = time.time()
    rng = random.Random(614) # Use fixed seed for reproducibility of the search itself? 
    # Actually, using a different seed for the search might help exploration. 
    # But the task says "pinned uniformly shuffled... random.Random(614)". 
    # That refers to the edge list generation. My search should probably use random for diversity.
    # I'll use the system random or a seeded one. Let's use a seeded one for consistency in debugging, 
    # but vary it if needed. Let's use 42 for the SA seed.
    search_rng = random.Random(42)

    while time.time() - start_time < time_limit:
        # Generate a move
        move_type = search_rng.random()
        new_perm = list(current)
        
        if move_type < 0.5:
            # 2-opt
            i = search_rng.randint(0, n - 1)
            j = search_rng.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if i == j:
                continue
            new_perm[i:j+1] = reversed(new_perm[i:j+1])
        elif move_type < 0.8:
            # Swap
            i = search_rng.randint(0, n - 1)
            j = search_rng.randint(0, n - 1)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insert
            i = search_rng.randint(0, n - 1)
            j = search_rng.randint(0, n - 1)
            if i == j:
                continue
            v = new_perm.pop(i)
            new_perm.insert(j, v)

        new_cost = calc_cost(new_perm)
        delta = new_cost - current_cost
        
        if delta < 0 or search_rng.random() < math.exp(-delta / T):
            current = new_perm
            current_cost = new_cost
            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost
        
        T *= cool_rate
        if T < 0.1:
            break
            
    return best, best_cost

def construct():
    # Deadline guard
    deadline = time.time() + 55.0
    
    # Initial solution: random permutation
    initial = list(range(n))
    random.Random(1).shuffle(initial)
    
    best_perm, best_cost = simulated_annealing(initial, 5.0)
    
    # Try multiple restarts with different initial states if time permits
    # We can also try greedy construction as a starting point
    # Greedy: start at 0, always pick the neighbor with the smallest edge index not yet visited
    # This might get stuck in a local optimum, but provides a good seed for SA
    
    start_time = time.time()
    restarts = 0
    while time.time() < deadline - 5.0:
        # Create a diverse initial state
        init = list(range(n))
        random.Random(restarts).shuffle(init)
        
        perm, cost = simulated_annealing(init, 5.0)
        
        if cost < best_cost:
            best_perm = perm
            best_cost = cost
            
        restarts += 1
        
    return best_perm
# EVOLVE-BLOCK-END
