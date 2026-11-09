"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_valid_addition(pos, points, dist_sets):
    """
    Check if adding 'pos' to 'points' maintains the no-isosceles property.
    Returns True if valid, False otherwise.
    Also computes the new distance set for the new point if valid.
    """
    new_dists = set()
    dists_list = []
    
    # 1. Check if 'pos' is the apex of an isosceles triangle with any two existing points
    for p in points:
        d2 = _sq_dist(pos, p)
        if d2 in new_dists:
            return False, None
        new_dists.add(d2)
        dists_list.append(d2)
        
    # 2. Check if any existing point is the apex of an isosceles triangle involving 'pos'
    for i, d2 in enumerate(dists_list):
        if d2 in dist_sets[i]:
            return False, None
            
    return True, new_dists

def _simulate_annealing(timeout):
    """
    Simulated Annealing to maximize the number of points.
    State: Set of points.
    Moves: Add, Remove, Swap.
    Energy: -len(points) (we want to minimize energy, i.e., maximize size).
    """
    start_time = time.time()
    
    # Initial state: Greedy construction to get a decent start
    points = []
    dist_sets = []
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_positions)
    
    # Quick greedy init
    for pos in all_positions:
        if time.time() - start_time > timeout * 0.1: # Spend 10% on init
            break
        valid, new_dists = _is_valid_addition(pos, points, dist_sets)
        if valid:
            points.append(pos)
            # Update existing dist sets
            for i, p in enumerate(points[:-1]):
                dist_sets[i].add(_sq_dist(pos, p))
            dist_sets.append(new_dists)
            
    current_points = points
    current_dist_sets = dist_sets
    current_size = len(current_points)
    best_points = current_points[:]
    best_size = current_size
    
    # SA Parameters
    T = 5.0  # Initial temperature
    T_min = 0.01
    alpha = 0.995 # Cooling rate
    
    # Pre-compute neighbors for faster swap? No, N^2 is 4096, which is small enough.
    # But checking validity is O(N).
    
    while time.time() - start_time < timeout:
        if T < T_min:
            T = T_min
            
        # Generate a move
        move_type = random.random()
        new_points = current_points[:]
        new_dist_sets = [s.copy() for s in current_dist_sets]
        
        if move_type < 0.4:
            # Add a random point
            if not current_points:
                continue
            pos = random.choice(all_positions)
            if pos in current_points:
                continue
            valid, new_dists = _is_valid_addition(pos, current_points, current_dist_sets)
            if valid:
                new_points.append(pos)
                for i, p in enumerate(current_points):
                    new_dist_sets[i].add(_sq_dist(pos, p))
                new_dist_sets.append(new_dists)
            else:
                # Invalid move, reject
                continue
                
        elif move_type < 0.7:
            # Remove a random point
            if not current_points:
                continue
            idx = random.randint(0, len(current_points) - 1)
            removed_pos = current_points[idx]
            
            # Remove from list
            new_points.pop(idx)
            new_dist_sets.pop(idx)
            
            # Update remaining dist sets
            for i in range(len(new_points)):
                new_dist_sets[i].discard(_sq_dist(new_points[i], removed_pos))
                
        else:
            # Swap: Remove one, try to add another
            if len(current_points) < 2:
                continue
            idx = random.randint(0, len(current_points) - 1)
            removed_pos = current_points[idx]
            
            # Temporarily remove
            temp_points = current_points[:]
            temp_dist_sets = [s.copy() for s in current_dist_sets]
            temp_points.pop(idx)
            temp_dist_sets.pop(idx)
            for i in range(len(temp_points)):
                temp_dist_sets[i].discard(_sq_dist(temp_points[i], removed_pos))
                
            # Try to add a random new point
            pos = random.choice(all_positions)
            if pos == removed_pos:
                continue
            if pos in temp_points:
                continue
                
            valid, new_dists = _is_valid_addition(pos, temp_points, temp_dist_sets)
            if valid:
                new_points = temp_points + [pos]
                new_dist_sets = temp_dist_sets
                for i, p in enumerate(temp_points):
                    new_dist_sets[i].add(_sq_dist(pos, p))
                new_dist_sets.append(new_dists)
            else:
                continue

        new_size = len(new_points)
        delta_e = (current_size - new_size) # Energy diff: E_new - E_old = (-new_size) - (-current_size)
        
        # Acceptance probability
        if delta_e <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_e / T)
            
        if accept:
            current_points = new_points
            current_dist_sets = new_dist_sets
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_points = current_points[:]
                
        T *= alpha
        
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave safety margin
    
    # Run multiple SA restarts to find a good solution
    best_points = []
    best_score = 0
    
    # We can run a few short SA runs or one long one.
    # Given the 300s limit, let's do a main SA run with a decent time budget,
    # and maybe a couple of quick restarts if time permits.
    
    # Strategy: 
    # 1. One long SA run (80% time)
    # 2. If time remains, short SA restarts (20% time)
    
    main_time = (deadline - start_time) * 0.8
    res_points = _simulate_annealing(main_time)
    if len(res_points) > best_score:
        best_points = res_points
        best_score = len(res_points)
        
    # Check time
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 10:
        # Try a few quick restarts
        # Each restart gets a small fraction of remaining time
        num_restarts = min(3, int(remaining / 10))
        for _ in range(num_restarts):
            if time.time() - start_time > deadline - 5:
                break
            restart_time = min(5.0, (deadline - time.time()) / (num_restarts + 1))
            res_points = _simulate_annealing(restart_time)
            if len(res_points) > best_score:
                best_points = res_points
                best_score = len(res_points)
                
    return best_points

# EVOLVE-BLOCK-END
