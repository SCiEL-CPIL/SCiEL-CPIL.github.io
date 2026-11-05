import itertools
import random
import time
import math
from collections import Counter

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s buffer for safety (90s total limit)
    
    N = 30
    RANGE = 30
    
    def calculate_score(points):
        counts = {}
        p = points
        for i in range(29):
            xi, yi = p[i]
            for j in range(i + 1, 29):
                xj, yj = p[j]
                dx1 = xj - xi
                dy1 = yj - yi
                for k in range(j + 1, 30):
                    xk, yk = p[k]
                    dx2 = xk - xi
                    dy2 = yk - yi
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] = counts.get(det, 0) + 1
        if not counts:
            return 0
        return max(counts.values())
    
    def calculate_score_fast(points):
        """Compute the max multiplicity of any doubled area using vectorized-like approach."""
        n = len(points)
        # Precompute all pair differences
        # For each pair (i,j), compute vector v_ij = p_j - p_i
        # For each triple (i,j,k), det = |v_ij x v_ik|
        # This is O(n^3) but with small constant
        
        # Alternative: for each point i, compute determinants with all pairs (j,k) where j,k > i
        counts = {}
        p = points
        for i in range(n - 2):
            xi, yi = p[i]
            # Collect vectors from i to all other points
            vecs = []
            for j in range(i + 1, n):
                vecs.append((p[j][0] - xi, p[j][1] - yi))
            m = len(vecs)
            for j in range(m - 1):
                ax, ay = vecs[j]
                for k in range(j + 1, m):
                    bx, by = vecs[k]
                    det = abs(ax * by - ay * bx)
                    if det > 0:
                        counts[det] = counts.get(det, 0) + 1
        if not counts:
            return 0
        return max(counts.values())
    
    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts
    
    def gen_lattice_directions():
        """Generate points using a small set of primitive directions with varying magnitudes."""
        # Use directions: (1,0), (0,1), (1,1), (1,-1)
        # Create a configuration that's a subset of a dense lattice
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        # Strategy: place points along lines with these directions
        # Start with a central point and expand
        pts = [(0, 0)]
        
        # Add points along each direction at various distances
        for dx, dy in directions:
            for dist in range(1, 8):
                p = (dist * dx, dist * dy)
                if -RANGE <= p[0] <= RANGE and -RANGE <= p[1] <= RANGE and p not in pts:
                    pts.append(p)
        
        # Add more points to reach 30
        # Use a grid-like pattern in the center
        for x in range(-5, 6):
            for y in range(-5, 6):
                p = (x, y)
                if len(pts) >= N:
                    break
                if p not in pts and -RANGE <= p[0] <= RANGE and -RANGE <= p[1] <= RANGE:
                    pts.append(p)
            if len(pts) >= N:
                break
        
        return pts[:N]
    
    def gen_scaled_lattice(scale=1):
        """Generate a scaled lattice pattern."""
        pts = []
        # Use a hexagonal-like lattice
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * scale + (j * scale // 2)
                y = j * scale * 3 // 2
                p = (x, y)
                if -RANGE <= p[0] <= RANGE and -RANGE <= p[1] <= RANGE:
                    if p not in pts:
                        pts.append(p)
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        return pts[:N]
    
    # Initial configurations
    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    inits.append(gen_grid(15, 2, -6, 0))
    inits.append(gen_grid(2, 15, 0, -6))
    
    # Add lattice-based initializations
    try:
        inits.append(gen_lattice_directions())
    except:
        pass
    try:
        inits.append(gen_scaled_lattice(1))
    except:
        pass
    try:
        inits.append(gen_scaled_lattice(2))
    except:
        pass
    
    # Add random initializations
    for _ in range(5):
        random.seed(random.randint(0, 10000))
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)
    
    best_overall_score = 0
    best_points = inits[0]
    
    for init_pts in inits:
        if time.time() > time_limit - 5.0:
            break
            
        current_points = list(init_pts)
        # Ensure distinct and in bounds
        if len(set(current_points)) != N:
            continue
        valid = True
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue
            
        curr_score = calculate_score(current_points)
        
        if curr_score > best_overall_score:
            best_overall_score = curr_score
            best_points = list(current_points)
            
        # Local Search with fast scoring
        no_improve = 0
        max_no_improve = 2000  # Allow more iterations with fast scoring
        
        while time.time() < time_limit - 2.0 and no_improve < max_no_improve:
            # Pick random point
            idx = random.randint(0, N-1)
            old_pt = current_points[idx]
            
            # Generate candidate moves
            moves = []
            # Local moves: neighbors with larger radius
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0: continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                        if new_pt not in current_points:
                            moves.append(new_pt)
            
            # Add random global moves
            for _ in range(5):
                new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                if new_pt not in current_points and new_pt != old_pt:
                    moves.append(new_pt)
            
            if not moves:
                no_improve += 1
                continue
                
            random.shuffle(moves)
            
            found_improvement = False
            for new_pt in moves:
                current_points[idx] = new_pt
                new_score = calculate_score_fast(current_points)
                
                if new_score > curr_score:
                    curr_score = new_score
                    found_improvement = True
                    no_improve = 0
                    if curr_score > best_overall_score:
                        best_overall_score = curr_score
                        best_points = list(current_points)
                    break
                elif new_score == curr_score:
                    # Accept neutral moves with some probability to wander
                    if random.random() < 0.3:
                        no_improve += 1
                        break
                    else:
                        current_points[idx] = old_pt
                else:
                    current_points[idx] = old_pt
            
            if not found_improvement:
                no_improve += 1
            else:
                no_improve = 0
    
    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)
    
    return best_points
