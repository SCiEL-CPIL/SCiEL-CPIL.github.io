"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

N = 16
DEADLINE = 325  # Safety margin before 330s kill

# Precompute all grid points
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
POINT_INDEX = {p: i for i, p in enumerate(ALL_POINTS)}

def dist_sq(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def is_isosceles(a, b, c):
    d1 = dist_sq(a, b)
    d2 = dist_sq(a, c)
    d3 = dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    
    # Strategy: Use randomized local search to find small maximal no-isosceles sets
    # 1. Start with greedy solution
    # 2. Try to remove points while maintaining maximality
    # 3. Use simulated annealing / hill climbing
    
    # First, get a greedy maximal solution as starting point
    def greedy_construct(order=None):
        if order is None:
            order = list(ALL_POINTS)
        chosen = []
        for p in order:
            ok = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_isosceles(p, chosen[i], chosen[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)
        return chosen
    
    # Check if a set is maximal: no point can be added
    def is_maximal(chosen):
        chosen_set = set(chosen)
        for p in ALL_POINTS:
            if p in chosen_set:
                continue
            # Can we add p?
            can_add = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_isosceles(p, chosen[i], chosen[j]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                return False
        return True
    
    # Check if a set is no-isosceles
    def is_no_isosceles(chosen):
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                for k in range(j+1, len(chosen)):
                    if is_isosceles(chosen[i], chosen[j], chosen[k]):
                        return False
        return True
    
    # Try to minimize: remove points one by one if possible while maintaining maximality
    def try_minimize(chosen, max_iters=10000):
        chosen = list(chosen)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > DEADLINE - 10:
                break
            # Shuffle order for randomness
            indices = list(range(len(chosen)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Try removing chosen[idx]
                new_chosen = chosen[:idx] + chosen[idx+1:]
                if not new_chosen:
                    continue
                if not is_no_isosceles(new_chosen):
                    continue
                if is_maximal(new_chosen):
                    chosen = new_chosen
                    improved = True
                    break
        return chosen
    
    # Run multiple greedy constructions with different orderings
    best = None
    best_size = float('inf')
    
    # Try several random orderings
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Different heuristics for ordering
        if attempt % 5 == 0:
            # Row-major
            order = list(ALL_POINTS)
        elif attempt % 5 == 1:
            # Reverse row-major
            order = list(reversed(ALL_POINTS))
        elif attempt % 5 == 2:
            # Random
            order = list(ALL_POINTS)
            random.shuffle(order)
        elif attempt % 5 == 3:
            # Corners and edges first
            order = sorted(ALL_POINTS, key=lambda p: min(p[0], N-1-p[0], p[1], N-1-p[1]))
        else:
            # Center outward
            order = sorted(ALL_POINTS, key=lambda p: -min(p[0], N-1-p[0], p[1], N-1-p[1]))
        
        chosen = greedy_construct(order)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = list(chosen)
        
        # Try to minimize
        minimized = try_minimize(chosen, max_iters=5000)
        if len(minimized) < best_size:
            best_size = len(minimized)
            best = minimized
    
    # If we still have time, do more aggressive local search
    if best is not None and time.time() - start_time < DEADLINE - 60:
        # Try removing points more aggressively with restarts
        for restart in range(20):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            current = list(best)
            # Perturb: randomly remove a point and see if we can recover
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                perturbed = current[:idx] + current[idx+1:]
                if perturbed and is_no_isosceles(perturbed):
                    # Try to restore maximality by adding points
                    for p in ALL_POINTS:
                        if p in set(perturbed):
                            continue
                        can_add = True
                        for i in range(len(perturbed)):
                            for j in range(i+1, len(perturbed)):
                                if is_isosceles(p, perturbed[i], perturbed[j]):
                                    can_add = False
                                    break
                            if not can_add:
                                break
                        if can_add:
                            perturbed.append(p)
                            break  # Add just one to try to maintain small size
                    
                    if is_no_isosceles(perturbed) and is_maximal(perturbed):
                        if len(perturbed) < best_size:
                            best_size = len(perturbed)
                            best = perturbed
    
    if best is None:
        # Fallback to simple greedy
        best = greedy_construct()
    
    return best

# EVOLVE-BLOCK-END
