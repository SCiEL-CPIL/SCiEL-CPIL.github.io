import numpy as np
import itertools
import random
import math
from collections import Counter

def construct():
    # Define the search space
    # Pool of points in [-10, 10] x [-10, 10]
    pool = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]
    pool_list = pool
    
    # Helper to calculate count of triangles with target area in a list of points
    def count_target_area(points, target_area):
        count = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det == target_area:
                        count += 1
        return count

    # Helper to calculate contribution of a point P to target area with set S
    # S is a list of points (size 29)
    def count_involving(P, S, target_area):
        count = 0
        x1, y1 = P
        n = len(S)
        for i in range(n):
            x2, y2 = S[i]
            for j in range(i+1, n):
                x3, y3 = S[j]
                det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if det == target_area:
                    count += 1
        return count

    # Local search for a specific target area
    def local_search_for_target(target_area, initial_points, iterations=2000):
        current_points = set(initial_points)
        current_count = count_target_area(list(current_points), target_area)
        best_points = list(current_points)
        best_count = current_count
        
        T = 0.1
        
        for _ in range(iterations):
            if not current_points:
                break
            out_point = random.choice(list(current_points))
            
            # Pick random in_point not in current_points
            while True:
                in_point = random.choice(pool_list)
                if in_point not in current_points:
                    break
            
            # Calculate delta
            remaining = [p for p in current_points if p != out_point]
            removed_contrib = count_involving(out_point, remaining, target_area)
            added_contrib = count_involving(in_point, remaining, target_area)
            
            delta = added_contrib - removed_contrib
            
            # Accept if delta > 0, or with probability exp(delta/T) if delta < 0
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points.remove(out_point)
                current_points.add(in_point)
                current_count += delta
                
                if current_count > best_count:
                    best_count = current_count
                    best_points = list(current_points)
                
            T *= 0.99
            if T < 0.001:
                break

        return best_points, best_count

    # Initial configurations
    # 1. Grid 6x5
    init_grid = []
    for x in range(-2, 4):
        for y in range(-2, 3):
            init_grid.append((x, y))
            
    # 2. Random set
    init_random = random.sample(pool, 30)
    
    # Target areas to try
    # Small integer areas are more likely to be frequent in small grids
    target_areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    
    best_global_points = None
    best_global_count = 0
    
    # Run local search for each target area
    inits = [init_grid, init_random]
    searches_per_target = 2
    
    for ta in target_areas:
        for init in inits:
            for _ in range(searches_per_target):
                pts, cnt = local_search_for_target(ta, init, iterations=2000)
                if cnt > best_global_count:
                    best_global_count = cnt
                    best_global_points = pts
                    
    # If no points found (shouldn't happen), return grid
    if best_global_points is None:
        return init_grid
        
    return best_global_points
