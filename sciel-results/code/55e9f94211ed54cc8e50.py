"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _check_validity(S_indices):
    """
    Check if S_indices is a valid no-isosceles set.
    Returns True if no three points in S form an isosceles triangle.
    """
    S_list = list(S_indices)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ab = DIST2[a, b]
            # Check if there is a k such that d(a,k) == d(b,k) or d(a,k) == d_ab or d(b,k) == d_ab
            # Actually, simpler: for any pair (a,b), check all k in S
            # But we can just check all triples.
            # O(n^3) is fine for small n.
            for k in range(j + 1, n):
                c = S_list[k]
                d_ac = DIST2[a, c]
                d_bc = DIST2[b, c]
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return False
    return True

def _is_blocked(u, S_indices):
    """
    Check if point u is blocked by the set S_indices.
    u is blocked if there exist a, b in S_indices such that triangle (u, a, b) is isosceles.
    """
    S_list = list(S_indices)
    n = len(S_list)
    if n < 2:
        return False
    
    for i in range(n):
        a = S_list[i]
        d_ua = DIST2[u, a]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ub = DIST2[u, b]
            d_ab = DIST2[a, b]
            if d_ua == d_ub or d_ua == d_ab or d_ub == d_ab:
                return True
    return False

def _maximality_violations(S_indices):
    """
    Return the number of points u not in S_indices that are NOT blocked by S_indices.
    """
    S_set = set(S_indices)
    violations = 0
    for u in range(V):
        if u in S_set:
            continue
        if not _is_blocked(u, S_indices):
            violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # 10s buffer for safety
    
    best_size = V
    best_set = []
    
    # Simulated Annealing
    # State: S (set of indices)
    # Energy: |S| * 100 + 10 * violations
    # We want to minimize energy.
    
    # Initialize with a random valid set (greedy)
    def get_random_greedy_set():
        order = list(range(V))
        random.shuffle(order)
        S = []
        for p in order:
            # Check if adding p keeps S valid
            if _check_validity(S + [p]):
                S.append(p)
        return S

    S = get_random_greedy_set()
    current_energy = len(S) * 100 + 10 * _maximality_violations(S)
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.995
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a neighbor
        # Moves:
        # 1. Add a random point p not in S
        # 2. Remove a random point p from S
        # 3. Swap a random point p in S with a random point q not in S
        
        move_type = random.random()
        
        if move_type < 0.3 and len(S) > 0:
            # Remove
            idx = random.randint(0, len(S) - 1)
            p = S[idx]
            S_new = S[:idx] + S[idx+1:]
        elif move_type < 0.6:
            # Add
            candidates = [p for p in range(V) if p not in S]
            if candidates:
                p = random.choice(candidates)
                S_new = S + [p]
            else:
                S_new = S[:]
        else:
            # Swap
            if len(S) > 0:
                idx = random.randint(0, len(S) - 1)
                p = S[idx]
                candidates = [q for q in range(V) if q not in S]
                if candidates:
                    q = random.choice(candidates)
                    S_new = S[:idx] + [q] + S[idx+1:]
                else:
                    S_new = S[:]
            else:
                S_new = S[:]
        
        # Check validity of S_new
        if not _check_validity(S_new):
            # Invalid move, reject
            continue
            
        # Calculate new energy
        violations_new = _maximality_violations(S_new)
        energy_new = len(S_new) * 100 + 10 * violations_new
        
        # Acceptance
        if energy_new < current_energy:
            S = S_new
            current_energy = energy_new
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
                if best_size <= 8:
                    break
        else:
            delta_E = energy_new - current_energy
            if delta_E < 0:
                S = S_new
                current_energy = energy_new
            else:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    S = S_new
                    current_energy = energy_new
        
        T *= cooling_rate
        
        # Restart if stuck?
        # For now, just cool down.
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
