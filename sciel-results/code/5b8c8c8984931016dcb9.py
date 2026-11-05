import time
import random
import math
from collections import Counter
import numpy as np

def construct():
    deadline = time.time() + 85.0  # safety margin for 90s limit
    random.seed(42)
    
    N = 30
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def full_count(points):
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count
    
    def get_pairs_excluding(idx):
        pairs = []
        for j in range(N):
            if j == idx:
                continue
            for k in range(j+1, N):
                if k == idx:
                    continue
                pairs.append((j, k))
        return pairs
    
    all_pairs = [get_pairs_excluding(i) for i in range(N)]
    
    def count_target(points, target):
        """Count how many triples have doubled area == target."""
        count = 0
        for i in range(N):
            for j, k in all_pairs[i]:
                if doubled_area(points[i], points[j], points[k]) == target:
                    count += 1
        return count

    def generate_double_lattice(shift_x, shift_y):
        """Generate 30 points as union of two 15-point grids."""
        # 15 points in a 3x5 grid centered at 0,0
        grid_A = [(x, y) for x in range(-2, 3) for y in range(-2, 3)]
        # We need exactly 15. 5x3 = 15.
        # Let's use x in [-2, 2], y in [-2, 2] -> 25 points. Take first 15?
        # Better: x in [-2, 2], y in [-1, 1] -> 15 points.
        grid_A = [(x, y) for x in range(-2, 3) for y in range(-1, 2)]
        
        grid_B = [(x + shift_x, y + shift_y) for x, y in grid_A]
        
        points = grid_A + grid_B
        # Check for duplicates and range
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                unique_points.append(p)
        
        # If we have duplicates or out of range, we might have < 30.
        # Pad with random points if needed
        while len(unique_points) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        
        return unique_points[:30]

    def generate_triangular_config():
        """Generate a triangular-ish configuration."""
        points = []
        # Rows of decreasing length
        # 1+2+3+4+5+6+7+8 = 36. We need 30.
        # 1+2+3+4+5+6+7 = 28. Add 2.
        # Let's just take a 6x5 grid and rotate it? No, let's use a hexagon.
        # Hexagon of radius 3 in integer coords is tricky.
        # Let's use a "diamond" |x|+|y| <= K.
        # K=4: 1+3+5+7+9+7+5+3+1 = 41.
        # K=3: 1+3+5+7+5+3+1 = 25.
        # K=3 gives 25 points. Add 5 more from K=4.
        points = [(x, y) for x in range(-4, 5) for y in range(-4, 5) if abs(x) + abs(y) <= 3]
        # 25 points.
        # Add 5 points from the next layer
        extra_candidates = [(x, y) for x in range(-4, 5) for y in range(-4, 5) if abs(x) + abs(y) == 4]
        random.shuffle(extra_candidates)
        points.extend(extra_candidates[:5])
        return points

    # Start with a structured initial configuration
    # Try a few different starting configurations and pick the best
    initial_configs = [
        # Grid 6x5
        [(x, y) for x in range(6) for y in range(5)],
        # Concentric rings
        [(0,0)] + [(r, 0) for r in range(1, 8)] + [(0, r) for r in range(1, 8)] + 
        [(-r, 0) for r in range(1, 7)] + [(0, -r) for r in range(1, 5)],
        # Double lattice with various shifts
        generate_double_lattice(3, 0),
        generate_double_lattice(0, 3),
        generate_double_lattice(2, 2),
        generate_double_lattice(4, 1),
        # Triangular config
        generate_triangular_config(),
    ]
    
    # Filter to valid distinct points in [-30,30]^2
    best_initial = None
    best_initial_count = 0
    
    for config in initial_configs:
        # Ensure 30 distinct points in range
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        # Pad with random points if needed
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]
        
        counter, ba, bc = full_count(filtered)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered
    
    coords = list(best_initial)
    counter, target, current_count = full_count(coords)
    best_overall = current_count
    best_points = list(coords)
    best_target = target
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.99
    
    no_improve = 0
    max_no_improve = 200
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            
            # Re-evaluate current state
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            
            # Switch to the most common area in current configuration
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50
            T = 10.0 # Reset temperature
            
            # Random restart with perturbation
            for _ in range(3):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    current_count = count_target(coords, target)
                    break
            
            # If current count dropped, revert
            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        pairs = all_pairs[idx]
        
        # Try multiple candidate positions
        candidates = []
        for _ in range(15):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.append(new_pt)
        
        best_delta = -999
        best_new_pt = None
        
        # Precompute old matches for this idx
        old_matches = 0
        for j, k in pairs:
            p_j = coords[j]
            p_k = coords[k]
            if doubled_area(old_pt, p_j, p_k) == target:
                old_matches += 1
        
        for new_pt in candidates:
            new_matches = 0
            for j, k in pairs:
                p_j = coords[j]
                p_k = coords[k]
                if doubled_area(new_pt, p_j, p_k) == target:
                    new_matches += 1
            
            delta = new_matches - old_matches
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt
        
        if best_new_pt is not None:
            # Simulated Annealing acceptance
            if best_delta > 0:
                accept = True
            else:
                # Accept with probability exp(delta / T)
                if T > 0.1:
                    accept_prob = math.exp(best_delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False
            
            if accept:
                coords[idx] = best_new_pt
                current_count += best_delta
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
            else:
                no_improve += 1
        
        T *= cooling_rate
    
    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
