"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

DEADLINE = 325.0  # wall-clock safety margin before 330s kill


def _is_isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    """Check if point p can be added to set chosen without forming isosceles triangles."""
    # Optimization: If chosen is small, this is fast.
    # We check all pairs in chosen.
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i + 1, n):
            b = chosen[j]
            if _is_isosceles(p, a, b):
                return False
    return True


def _make_maximal(chosen, rng):
    """Greedily add points to chosen to make it maximal. Returns new list."""
    chosen_set = set(chosen)
    candidates = [p for p in itertools.product(range(N), repeat=2) if p not in chosen_set]
    rng.shuffle(candidates)
    
    new_chosen = list(chosen)
    new_chosen_set = set(new_chosen)
    
    for p in candidates:
        if _can_add(p, new_chosen):
            new_chosen.append(p)
            new_chosen_set.add(p)
            
    return new_chosen


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    # Initial solution: Greedy maximal set
    order = list(all_points)
    rng.shuffle(order)
    current = []
    for p in order:
        if _can_add(p, current):
            current.append(p)
    current = _make_maximal(current, rng)
    
    best = list(current)
    best_len = len(best)
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    max_iterations = 50000
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Generate a neighbor
        new_set = list(current)
        
        move_type = rng.random()
        if move_type < 0.5:
            # Move 1: Remove a random point, then refill to maximal
            if len(new_set) == 0:
                continue
            idx = rng.randint(0, len(new_set) - 1)
            new_set.pop(idx)
            new_set = _make_maximal(new_set, rng)
        else:
            # Move 2: Swap a point in with a point out, then refill
            if len(new_set) == 0:
                continue
            idx_in = rng.randint(0, len(new_set) - 1)
            point_out = new_set[idx_in]
            new_set.pop(idx_in)
            
            # Pick a random point not in new_set to try adding
            candidates = [p for p in all_points if p not in new_set]
            if not candidates:
                continue
            point_in = rng.choice(candidates)
            
            # Check if point_in can be added to new_set (without point_out)
            if _can_add(point_in, new_set):
                new_set.append(point_in)
                # Refill to maximal
                new_set = _make_maximal(new_set, rng)
            else:
                # If point_in can't be added, just refill the set without it
                # This is effectively a "remove and refill" but with a different random seed for refill order
                new_set = _make_maximal(new_set, rng)
        
        # Evaluate
        delta_len = len(new_set) - len(current)
        
        # Acceptance
        if delta_len < 0:
            # Improvement
            current = new_set
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)
        else:
            # Worse or same
            if delta_len == 0:
                # Accept with some probability to explore
                if rng.random() < 0.5:
                    current = new_set
            else:
                # Accept with probability exp(-delta / T)
                prob = math.exp(-delta_len / T)
                if rng.random() < prob:
                    current = new_set
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        iter_count += 1
        
        # Occasionally reheat to escape local minima? 
        # Maybe not needed if we have enough iterations and cooling is slow.
        
    return best
# EVOLVE-BLOCK-END
