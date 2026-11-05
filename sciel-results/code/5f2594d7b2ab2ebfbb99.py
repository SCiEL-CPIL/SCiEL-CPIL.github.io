"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_N = 24

# Precompute the edge order from the given data
_EDGE_DATA = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build arrival index: edge_index[(u,v)] = 1-based position in the shuffled list
_EDGE_INDEX = {}
for i, (u, v) in enumerate(_EDGE_DATA):
    if u > v:
        u, v = v, u
    _EDGE_INDEX[(u, v)] = i + 1

def _edge_score(u, v):
    """Get the arrival index (1-based) for edge (u,v)."""
    if u > v:
        u, v = v, u
    return _EDGE_INDEX.get((u, v), float('inf'))

def _cycle_score(cycle):
    """Compute the score of a cycle: max arrival index of its edges."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        s = _edge_score(u, v)
        if s > max_idx:
            max_idx = s
    return max_idx

def _random_cycle(rng):
    """Generate a random Hamilton cycle."""
    perm = list(range(_N))
    rng.shuffle(perm)
    return perm

def _local_search(cycle, rng, deadline):
    """
    Improve a cycle using 2-opt moves until no improvement is found or deadline is reached.
    Returns the best cycle found.
    """
    n = len(cycle)
    current_cycle = list(cycle)
    current_score = _cycle_score(current_cycle)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try all 2-opt moves
        for i in range(n):
            for j in range(i + 1, n):
                # Skip trivial moves
                if i == 0 and j == n - 1:
                    continue
                # Calculate new score efficiently
                # Edges affected: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # New edges: (i-1, j), (i, j+1)
                
                # We need to be careful with indices modulo n
                u1 = current_cycle[(i - 1) % n]
                u2 = current_cycle[i]
                u3 = current_cycle[(i + 1) % n]
                
                u4 = current_cycle[(j - 1) % n]
                u5 = current_cycle[j]
                u6 = current_cycle[(j + 1) % n]
                
                # If i and j are adjacent, the move is different
                if (i + 1) % n == j or (j + 1) % n == i:
                    continue # Skip adjacent reversals as they are trivial or handled differently? 
                    # Actually, reversing a segment of length 2 (i, i+1) swaps them. 
                    # Let's just compute the new cycle to be safe, though it's slower.
                    # For speed, let's only do non-adjacent 2-opts in the inner loop if possible.
                    # But for n=24, full 2-opt is O(n^2) = 576 checks. 
                    # Each check involves 4 edge lookups. 576 * 4 * 2 (old/new) is manageable.
                    pass

                # Construct new cycle for score calculation (slower but safer for logic)
                new_cycle = current_cycle[:i] + list(reversed(current_cycle[i:j+1])) + current_cycle[j+1:]
                new_score = _cycle_score(new_cycle)
                
                if new_score < current_score:
                    current_cycle = new_cycle
                    current_score = new_score
                    improved = True
                    # Restart loop from beginning of i,j to find more improvements
                    break
            if improved:
                break
    return current_cycle, current_score

def _perturb(cycle, rng):
    """
    Perturb the cycle by performing a 2-opt move and a random re-insertion.
    """
    n = len(cycle)
    new_cycle = list(cycle)
    
    # 1. Random 2-opt
    i = rng.randint(0, n - 1)
    j = rng.randint(0, n - 1)
    if i > j:
        i, j = j, i
    if i != j and not (i == 0 and j == n - 1):
        new_cycle = new_cycle[:i] + list(reversed(new_cycle[i:j+1])) + new_cycle[j+1:]
        
    # 2. Random re-insertion of a random vertex
    v_idx = rng.randint(0, n - 1)
    v = new_cycle[v_idx]
    new_cycle.pop(v_idx)
    # Insert at a new random position
    ins_idx = rng.randint(0, n - 1)
    new_cycle.insert(ins_idx, v)
    
    return new_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin: 80s out of 90s limit
    
    rng = random.Random(614) # Use the seed from the problem description? Or a fixed one? 
    # The problem says "locally shuffled with random.Random(614)". 
    # It's best to use a deterministic seed for reproducibility in our own search, 
    # but the edge list is fixed. Let's use a different seed for our search to explore.
    rng = random.Random(42)
    
    # Initial solution: Greedy construction
    # Start with a random vertex and always pick the neighbor with the lowest edge score
    start_v = rng.randint(0, _N - 1)
    visited = [False] * _N
    visited[start_v] = True
    current_cycle = [start_v]
    current = start_v
    
    for _ in range(_N - 1):
        best_next = -1
        best_score = float('inf')
        for v in range(_N):
            if not visited[v]:
                s = _edge_score(current, v)
                if s < best_score:
                    best_score = s
                    best_next = v
        current_cycle.append(best_next)
        visited[best_next] = True
        current = best_next
        
    # Initial local search
    best_cycle, best_score = _local_search(current_cycle, rng, deadline)
    
    # Iterated Local Search
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    
    while time.time() < deadline - 5:
        # Perturb
        perturbed_cycle = _perturb(best_cycle, rng)
        
        # Local search on perturbed cycle
        local_best_cycle, local_best_score = _local_search(perturbed_cycle, rng, deadline)
        
        # Acceptance criterion (Metropolis)
        delta = local_best_score - best_score
        if delta < 0:
            best_cycle = local_best_cycle
            best_score = local_best_score
        else:
            # Accept with probability exp(-delta / T)
            if T > T_min:
                accept_prob = math.exp(-delta / T)
                if rng.random() < accept_prob:
                    best_cycle = local_best_cycle
                    best_score = local_best_score
            
            # Cool down
            T *= cooling_rate
    
    return best_cycle
# EVOLVE-BLOCK-END
