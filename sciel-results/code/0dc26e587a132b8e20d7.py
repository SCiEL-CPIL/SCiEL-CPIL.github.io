"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

N = 16
DEADLINE = 320  # seconds, with safety margin

_init_time = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - _init_time)

def _point_to_idx(p):
    return p[0] * N + p[1]

def _idx_to_point(idx):
    return (idx // N, idx % N)

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_valid_addition(p, chosen):
    """Check if p can be added to chosen without creating isosceles."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _isosceles(p, chosen[i], chosen[j]):
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if chosen is maximal: no point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _is_valid_addition(p, chosen):
            return False
    return True

def _greedy_upper_bound(chosen, remaining, order):
    """Greedy: keep adding points from order that are valid, return total size."""
    current = list(chosen)
    current_set = set(current)
    for p in order:
        if p in current_set:
            continue
        if _is_valid_addition(p, current):
            current.append(p)
            current_set.add(p)
    # Make it maximal by adding blocking points
    # Actually for upper bound, just count what we have
    return len(current)

def construct():
    global _init_time
    _init_time = time.time()
    
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    
    # Precompute: for each point, which other points conflict with it
    # conflict[i][j] = True if i and j cannot both be in a no-isosceles set
    # Actually, the constraint is on triples, not pairs. Two points are always OK together.
    # The issue is when adding a third point creates isosceles.
    
    # Better: for a set S, point p can be added iff for all pairs (a,b) in S, p,a,b not isosceles.
    
    # Strategy: Branch and bound
    # State: current chosen set, index in ordering
    # We want to minimize |chosen| while ensuring maximality
    
    # Use a randomized restart approach with different orderings
    best_solution = None
    best_size = n_points + 1
    
    # First, get a quick greedy solution as initial upper bound
    random.seed(42)
    order = list(range(n_points))
    random.shuffle(order)
    greedy_chosen = []
    for idx in order:
        p = all_points[idx]
        if _is_valid_addition(p, greedy_chosen):
            greedy_chosen.append(p)
    # Make maximal
    greedy_chosen_set = set(greedy_chosen)
    for p in all_points:
        if p not in greedy_chosen_set and _is_valid_addition(p, greedy_chosen):
            greedy_chosen.append(p)
            greedy_chosen_set.add(p)
    best_solution = greedy_chosen
    best_size = len(greedy_chosen)
    
    # Now try branch and bound with different strategies
    # For efficiency, we'll use a simpler approach: local search / hill climbing
    
    # Local search: start from greedy, try to remove points while maintaining validity and maximality
    current = list(best_solution)
    current_size = best_size
    
    for iteration in range(1000):
        if _time_remaining() < 5:
            break
        
        # Try to remove a point and see if the set is still maximal
        # If removing a point makes it non-maximal, we need to add back some points
        improved = False
        for i in range(len(current)):
            if _time_remaining() < 5:
                break
            candidate = current[:i] + current[i+1:]
            # Check if still no-isosceles (should be, since we're removing)
            # Check if maximal
            candidate_set = set(candidate)
            is_max = True
            for p in all_points:
                if p in candidate_set:
                    continue
                if _is_valid_addition(p, candidate):
                    is_max = False
                    break
            
            if is_max:
                # We can remove this point
                current = candidate
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_solution = list(current)
                    best_size = current_size
                break
        
        if not improved:
            # Try swapping: remove one, add another that allows removal of more
            # Or just try random perturbations
            if len(current) > 0 and random.random() < 0.3:
                # Randomly remove a point, then greedily add back to make maximal
                idx = random.randint(0, len(current) - 1)
                perturbed = current[:idx] + current[idx+1:]
                perturbed_set = set(perturbed)
                # Add back points to make maximal
                for p in all_points:
                    if p not in perturbed_set and _is_valid_addition(p, perturbed):
                        perturbed.append(p)
                        perturbed_set.add(p)
                if len(perturbed) < current_size:
                    current = perturbed
                    current_size = len(current)
                    if current_size < best_size:
                        best_solution = list(current)
                        best_size = current_size
    
    return best_solution
