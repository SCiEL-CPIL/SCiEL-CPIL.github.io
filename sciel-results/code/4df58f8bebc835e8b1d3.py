"""No-isosceles set construction on [0, N)^2 grid."""
import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    # Pre-compute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def squared_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy
    
    def is_valid_addition(candidate, current_set):
        """Check if adding candidate to current_set creates any isosceles triangle."""
        # For each pair of existing points, check if candidate forms isosceles with them
        # An isosceles triangle has at least two equal sides
        n = len(current_set)
        for i in range(n):
            p_i = current_set[i]
            d_ic = squared_dist(p_i, candidate)
            for j in range(i+1, n):
                p_j = current_set[j]
                d_ij = squared_dist(p_i, p_j)
                d_jc = squared_dist(p_j, candidate)
                # Check if any two distances are equal (and non-zero)
                if d_ic > 0 and d_jc > 0 and d_ij > 0:
                    if d_ic == d_ij or d_ic == d_jc or d_ij == d_jc:
                        return False
        return True
    
    def greedy_construct(starting_points=None, max_iter=50000):
        """Greedy construction: add points that don't create isosceles triangles."""
        if starting_points is None:
            current_set = []
        else:
            current_set = list(starting_points)
        
        # Try to add points in a randomized order
        remaining = [p for p in all_points if p not in set(current_set)]
        random.shuffle(remaining)
        
        for candidate in remaining:
            if time.time() > deadline:
                break
            if is_valid_addition(candidate, current_set):
                current_set.append(candidate)
        
        return current_set
    
    def local_search(initial_set, max_iterations=10000):
        """Local search: try swapping points to improve set size."""
        current_set = list(initial_set)
        current_size = len(current_set)
        best_set = list(current_set)
        best_size = current_size
        
        # Precompute which points are in the set for quick lookup
        in_set = set(current_set)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to add a random point
            if len(remaining_points := [p for p in all_points if p not in in_set]) > 0:
                candidate = random.choice(remaining_points)
                if is_valid_addition(candidate, current_set):
                    current_set.append(candidate)
                    in_set.add(candidate)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
            
            # Occasionally try to remove a point and add a different one
            if len(current_set) > 0 and random.random() < 0.3:
                idx = random.randint(0, len(current_set) - 1)
                removed = current_set.pop(idx)
                in_set.discard(removed)
                
                # Try to add a better point
                remaining = [p for p in all_points if p not in in_set]
                random.shuffle(remaining)
                added = False
                for candidate in remaining[:100]:  # limit search
                    if is_valid_addition(candidate, current_set):
                        current_set.append(candidate)
                        in_set.add(candidate)
                        added = True
                        if len(current_set) > best_size:
                            best_set = list(current_set)
                            best_size = len(current_set)
                        break
                
                if not added:
                    # Put the removed point back
                    current_set.append(removed)
                    in_set.add(removed)
        
        return best_set
    
    # Strategy: multiple restarts with greedy + local search
    best_overall = []
    best_size_overall = 0
    
    # First, do a few pure greedy runs with different random seeds
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        
        # Greedy construction
        current_set = greedy_construct()
        if len(current_set) > best_size_overall:
            best_overall = list(current_set)
            best_size_overall = len(current_set)
        
        # Local search from the greedy result
        if time.time() < deadline - 30:
            improved = local_search(current_set, max_iterations=5000)
            if len(improved) > best_size_overall:
                best_overall = list(improved)
                best_size_overall = len(improved)
    
    # If we have time, try more sophisticated approaches
    if time.time() < deadline - 60:
        # Try starting from different initial configurations
        for init_strategy in range(5):
            if time.time() > deadline:
                break
            
            random.seed(99999 + init_strategy * 777)
            
            # Start with a small random seed set
            seed_size = min(5, N)
            initial = random.sample(all_points, seed_size)
            
            # Check if initial set is valid
            valid_initial = []
            for p in initial:
                if is_valid_addition(p, valid_initial):
                    valid_initial.append(p)
            
            current_set = greedy_construct(starting_points=valid_initial)
            if len(current_set) > best_size_overall:
                best_overall = list(current_set)
                best_size_overall = len(current_set)
            
            if time.time() < deadline - 30:
                improved = local_search(current_set, max_iterations=5000)
                if len(improved) > best_size_overall:
                    best_overall = list(improved)
                    best_size_overall = len(improved)
    
    return best_overall
