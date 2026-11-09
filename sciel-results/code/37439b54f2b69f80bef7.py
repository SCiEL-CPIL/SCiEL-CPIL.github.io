import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_distances(all_points):
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix

def _can_add(x, chosen_indices, dist_matrix):
    """
    Check if point x can be added to chosen set without creating isosceles triangle.
    """
    k = len(chosen_indices)
    if k == 0:
        return True
    
    distances_from_x = set()
    
    for p in chosen_indices:
        d = dist_matrix[x][p]
        if d in distances_from_x:
            return False
        distances_from_x.add(d)
        
    for p in chosen_indices:
        d_xp = dist_matrix[x][p]
        for q in chosen_indices:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
                
    return True

def _is_maximal(chosen_indices, all_indices, dist_matrix):
    """
    Check if chosen is maximal (no further point can be added).
    """
    chosen_set = set(chosen_indices)
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen_indices, dist_matrix):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, rng):
    """
    Construct a maximal set by adding points in random order.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen_indices = []
    for p in order:
        if _can_add(p, chosen_indices, dist_matrix):
            chosen_indices.append(p)
    return chosen_indices

def _simulated_annealing(all_indices, dist_matrix, rng, deadline, initial_indices):
    """
    Simulated annealing to minimize size of maximal no-isosceles set.
    Moves: 
      1. Remove one point (if result is still maximal)
      2. Swap one point with another (if result is maximal)
    Cost function: size of set (lower is better)
    """
    s = list(initial_indices)
    n = len(s)
    
    # Initial temperature
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_s = list(s)
    best_size = n
    
    current_size = n
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.4 and n > 1:
            # Try removing a point
            i = rng.randrange(n)
            candidate = [s[k] for k in range(n) if k != i]
            
            if _is_maximal(candidate, all_indices, dist_matrix):
                new_size = len(candidate)
                delta = new_size - current_size
                
                # Accept if better or with probability exp(-delta/T)
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                    s = candidate
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_s = list(s)
                    n = len(s)
            T = max(T * cooling, T_min)
            
        elif move_type < 0.8:
            # Try swapping one point with another
            if n > 0:
                i = rng.randrange(n)
                # Pick a candidate to add
                candidates = [x for x in all_indices if x not in set(s)]
                if candidates:
                    j = rng.randrange(len(candidates))
                    r = candidates[j]
                    
                    # Remove s[i], add r
                    base = [s[k] for k in range(n) if k != i]
                    if _can_add(r, base, dist_matrix):
                        candidate = base + [r]
                        
                        if _is_maximal(candidate, all_indices, dist_matrix):
                            new_size = len(candidate)
                            delta = new_size - current_size
                            
                            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                                s = candidate
                                current_size = new_size
                                if current_size < best_size:
                                    best_size = current_size
                                    best_s = list(s)
                                n = len(s)
                    T = max(T * cooling, T_min)
                    
        else:
            # Try 2-opt style: remove two, add one or two
            if n >= 2:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i2 == i1:
                    i2 = (i2 + 1) % n
                    
                base = [s[k] for k in range(n) if k != i1 and k != i2]
                
                # Try adding one point
                candidates = [x for x in all_indices if x not in set(base)]
                if candidates:
                    # Try a few random candidates
                    for _ in range(3):
                        r = rng.choice(candidates)
                        if _can_add(r, base, dist_matrix):
                            candidate = base + [r]
                            if _is_maximal(candidate, all_indices, dist_matrix):
                                new_size = len(candidate)
                                delta = new_size - current_size
                                if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                                    s = candidate
                                    current_size = new_size
                                    if current_size < best_size:
                                        best_size = current_size
                                        best_s = list(s)
                                    n = len(s)
                                    break
                T = max(T * cooling, T_min)
                
        # Cool down
        T = max(T * cooling, T_min)
        
    return best_s

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix = _precompute_distances(all_points)
    
    best_indices = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Run multiple restarts with simulated annealing
    restart_count = 0
    while time.time() < _DEADLINE:
        restart_count += 1
        
        # 1. Greedy Construction
        initial_indices = _construct_greedy(all_indices, dist_matrix, rng)
        
        # 2. Simulated Annealing
        # Allocate time for this restart
        time_per_restart = min(30, (_DEADLINE - time.time()) / max(1, 10 - restart_count))
        sa_deadline = time.time() + time_per_restart
        
        sa_indices = _simulated_annealing(
            all_indices, dist_matrix, rng, sa_deadline, initial_indices
        )
        
        # 3. Update best
        if len(sa_indices) < best_size:
            best_size = len(sa_indices)
            best_indices = sa_indices
            
        # If we've found a very good solution, we can stop early
        if best_size <= 12:
            break
            
    if best_indices is None:
        best_indices = _construct_greedy(all_indices, dist_matrix, rng)
        
    return [all_points[i] for i in best_indices]
