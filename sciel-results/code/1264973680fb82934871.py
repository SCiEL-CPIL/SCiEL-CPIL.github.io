import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    # Objective function: Max multiplicity of a single positive doubled area
    def get_score(points):
        counts = {}
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    # Doubled area determinant
                    area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    if area > 0:
                        counts[area] = counts.get(area, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    # Fast update logic for local search
    # points: list of (x, y)
    # counts: dict of area -> count
    # max_mult: current max count
    
    def local_search(points, time_limit):
        points = [tuple(p) for p in points]
        n = len(points)
        
        # Precompute indices for pairs
        # For a given i, we need all pairs (j, k) with j, k != i
        # We can just loop j in range(n), k in range(j, n) and skip i
        
        # Initialize counts and max_mult
        counts = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    if area > 0:
                        counts[area] = counts.get(area, 0) + 1
        
        if not counts:
            max_mult = 0
        else:
            max_mult = max(counts.values())
        
        best_points = points[:]
        best_score = max_mult
        
        occupied = set(points)
        
        # Local search loop
        while time.time() < time_limit:
            # Pick a random point to move
            i = random.randint(0, n-1)
            old_p = points[i]
            
            # Pick a new random point
            new_p = None
            attempts = 0
            while attempts < 100:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                cand = (x, y)
                if cand not in occupied:
                    new_p = cand
                    break
                attempts += 1
            
            if new_p is None:
                continue
                
            # Calculate change
            # Remove triples involving old_p
            # Add triples involving new_p
            
            # We need to iterate over all pairs (j, k) such that j, k != i
            # And j < k to avoid duplicates?
            # The area is symmetric, so order of j, k doesn't matter for the set of triples.
            # We just need to consider each pair (j, k) once.
            
            improved = False
            old_max = max_mult
            
            # We'll store the changes to revert if not improved
            changes = []
            
            for j in range(n):
                if j == i: continue
                xj, yj = points[j]
                for k in range(j+1, n):
                    if k == i: continue
                    xk, yk = points[k]
                    
                    # Area with old point
                    a_old = abs(old_p[0]*(yj - yk) + xj*(yk - old_p[1]) + xk*(old_p[1] - yj))
                    # Area with new point
                    a_new = abs(new_p[0]*(yj - yk) + xj*(yk - new_p[1]) + xk*(new_p[1] - yj))
                    
                    if a_old > 0:
                        counts[a_old] -= 1
                        changes.append((a_old, -1))
                        if counts[a_old] == 0:
                            del counts[a_old]
                            
                    if a_new > 0:
                        if a_new in counts:
                            counts[a_new] += 1
                            changes.append((a_new, 1))
                        else:
                            counts[a_new] = 1
                            changes.append((a_new, 1))
                            
            # Recalculate max
            # This is O(num_areas). num_areas is small.
            if counts:
                new_max = max(counts.values())
            else:
                new_max = 0
                
            if new_max > best_score:
                # Accept
                points[i] = new_p
                occupied.remove(old_p)
                occupied.add(new_p)
                best_score = new_max
                best_points = points[:]
            else:
                # Revert
                for area, delta in changes:
                    counts[area] -= delta
                    if counts[area] == 0:
                        del counts[area]
                        
        return best_points, best_score

    # Initialization Strategies
    initial_configs = []
    
    # 1. 6x5 Grid centered
    xs = [-10, -6, -2, 2, 6, 10]
    ys = [-10, -5, 0, 5, 10]
    initial_configs.append([(x, y) for x in xs for y in ys])
    
    # 2. 5x6 Grid centered
    xs = [-10, -5, 0, 5, 10]
    ys = [-12, -6, 0, 6, 12]
    initial_configs.append([(x, y) for x in xs for y in ys])
    
    # 3. Random
    random.seed(42)
    pts = []
    occupied = set()
    while len(pts) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in occupied:
            pts.append((x, y))
            occupied.add((x, y))
    initial_configs.append(pts)
    
    # 4. Another random
    random.seed(123)
    pts = []
    occupied = set()
    while len(pts) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in occupied:
            pts.append((x, y))
            occupied.add((x, y))
    initial_configs.append(pts)
    
    best_global_points = None
    best_global_score = 0
    
    # Evaluate initial configs to pick a good start
    for conf in initial_configs:
        s = get_score(conf)
        if s > best_global_score:
            best_global_score = s
            best_global_points = conf
    
    # Run local search on the best initial config
    # And maybe on others if time permits
    
    # We will run local search on the best one first
    p, s = local_search(best_global_points, time_limit - 5) # Leave 5s margin
    
    if s > best_global_score:
        best_global_score = s
        best_global_points = p
        
    # If time remains, try local search on other promising starts
    if time.time() < time_limit - 10:
        for conf in initial_configs:
            if conf == best_global_points: continue
            # Check if this config is promising (score close to best)
            # Or just run it
            if time.time() > time_limit - 5: break
            p, s = local_search(conf, time_limit - 2)
            if s > best_global_score:
                best_global_score = s
                best_global_points = p
                
    return best_global_points

# EVOLVE-BLOCK-END
