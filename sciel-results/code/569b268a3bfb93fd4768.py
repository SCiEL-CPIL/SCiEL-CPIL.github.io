"""No-isosceles set: find max points in [0,N)^2 with no 3 forming isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 290  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def try_greedy(order, deadline):
        """Greedy: add points in given order if they don't create isosceles."""
        pts = []
        # For efficiency, track distances from each existing point
        # When adding p, check against all existing pairs
        for p in order:
            if time.time() > deadline:
                break
            valid = True
            k = len(pts)
            for i in range(k):
                di = dist2(p, pts[i])
                if di == 0:
                    valid = False
                    break
                for j in range(i + 1, k):
                    # Check if p, pts[i], pts[j] form isosceles
                    dj = dist2(p, pts[j])
                    dij = dist2(pts[i], pts[j])
                    if di == dj or di == dij or dj == dij:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                pts.append(p)
        return pts
    
    # Multiple strategies with different orderings
    num_attempts = 200
    
    for attempt in range(num_attempts):
        if time.time() > TIME_LIMIT - 5:
            break
        
        # Different ordering strategies
        if attempt % 4 == 0:
            # Random shuffle
            order = all_points[:]
            random.shuffle(order)
        elif attempt % 4 == 1:
            # Sort by distance from center (start near center)
            cx, cy = N / 2, N / 2
            order = sorted(all_points, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
            random.shuffle(order)  # Add some randomness
        elif attempt % 4 == 2:
            # Sort by x then y, with random perturbation
            order = sorted(all_points, key=lambda p: (p[0] + random.random(), p[1] + random.random()))
        else:
            # Spiral-like ordering
            cx, cy = N // 2, N // 2
            order = sorted(all_points, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy) + random.random() * 0.1)
        
        pts = try_greedy(order, TIME_LIMIT - 10)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
    
    # Local improvement: try to add more points or swap
    if len(best_pts) > 0 and time.time() < TIME_LIMIT - 20:
        current = best_pts[:]
        # Try adding random points
        for _ in range(500):
            if time.time() > TIME_LIMIT - 10:
                break
            p = random.choice(all_points)
            if p in current:
                continue
            valid = True
            k = len(current)
            for i in range(k):
                di = dist2(p, current[i])
                for j in range(i + 1, k):
                    dj = dist2(p, current[j])
                    dij = dist2(current[i], current[j])
                    if di == dj or di == dij or dj == dij:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current.append(p)
        
        if len(current) > best_len:
            best_pts = current
    
    return best_pts
# EVOLVE-BLOCK-END
