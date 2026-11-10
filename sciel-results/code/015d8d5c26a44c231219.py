import numpy as np
import time
import random

DEADLINE = 620.0
START_TIME = time.time()

def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    n_points = p * p * p * p  # 2401
    n_dirs = p * p + 1        # 50

    # Point indexing:
    # Point ((xa, xb), (ya, yb)) -> idx
    # idx = ((xa * p + xb) * p + ya) * p + yb
    
    # Precompute field operations
    # alpha^2 = 3
    # (a1 + b1 alpha) * (a2 + b2 alpha) = (a1 a2 + 3 b1 b2) + (a1 b2 + a2 b1) alpha
    
    def field_mul(a1, b1, a2, b2):
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return a3, b3

    def field_sub(a1, b1, a2, b2):
        return (a1 - a2) % p, (b1 - b2) % p

    # Precompute Impact lists
    # Impact[x] is a list of (y, v) encoded as y * n_dirs + v
    # such that x is on the line through y with direction v, x != y.
    
    # First, map point index to coordinates
    coords = []
    for idx in range(n_points):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        coords.append(((xa, xb), (ya, yb)))

    # Directions
    # v=0..48: (1, m) where m = ma + mb alpha
    # v=49: (0, 1)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append(((1, 0), (ma, mb)))
    dirs.append(((0, 1), (0, 0)))

    # For each point x, determine the direction v and the other point y?
    # It's easier to iterate over pairs (x, y) and find the direction.
    # But that's O(N^2) = 5.7M pairs.
    # For each pair (x, y), find v.
    # Then add (x, v) to Impact[y] and (y, v) to Impact[x].
    
    impact = [[] for _ in range(n_points)]
    
    # To speed up direction finding, precompute for each difference d = y - x, the direction.
    # d = (dx, dy).
    # If dx == 0, v = 49.
    # Else, m = dy / dx. v = index of m.
    
    # Precompute inverse of all non-zero elements
    inv = {}
    for i in range(1, p*p):
        # find j such that i*j = 1
        pass # Too slow. Just compute on fly or precompute mult table.
        
    # Let's just compute direction for each pair (x, y) directly.
    # Optimization: Group by x.
    for x in range(n_points):
        (xa, xb), (ya, yb) = coords[x]
        for y in range(n_points):
            if x == y: continue
            (xa2, xb2), (ya2, yb2) = coords[y]
            # dx = x - xa? No, direction is slope of line.
            # Line through x and y.
            # dx = xa2 - xa, etc.
            dx_a = (xa2 - xa) % p
            dx_b = (xb2 - xb) % p
            dy_a = (ya2 - ya) % p
            dy_b = (yb2 - yb) % p
            
            # Check if dx is 0
            if dx_a == 0 and dx_b == 0:
                # Vertical line
                v = 49
            else:
                # m = dy / dx
                # dy = dy_a + dy_b alpha
                # dx = dx_a + dx_b alpha
                # inv_dx = 1 / dx
                # m = dy * inv_dx
                
                # Compute 1/dx
                # (dx_a + dx_b alpha)^-1
                # Norm = dx_a^2 - 3 dx_b^2
                norm = (dx_a * dx_a - k * dx_b * dx_b) % p
                # inv_norm
                inv_norm = pow(norm, p-2, p) # Fermat's little theorem
                # inv_dx = (dx_a - dx_b alpha) / norm
                inv_dx_a = (dx_a * inv_norm) % p
                inv_dx_b = (-dx_b * inv_norm) % p
                
                # m = dy * inv_dx
                ma, mb = field_mul(dy_a, dy_b, inv_dx_a, inv_dx_b)
                
                # Find index of m
                # m = ma + mb alpha
                # Index in dirs is ma * p + mb
                v = ma * p + mb
            
            # Add to impact lists
            # Adding x to S blocks v for y
            impact[y].append(x * n_dirs + v)
            # Adding y to S blocks v for x
            impact[x].append(y * n_dirs + v)

    # Greedy Algorithm with Random Restarts
    
    best_S_size = 0
    best_S_mask = 0
    best_S_points = []
    
    rng = np.random.default_rng(42)
    
    while _time_left() > 10:
        # Initialize A[y] to all directions available
        A = np.full(n_points, (1 << n_dirs) - 1, dtype=np.int64)
        S_mask = 0
        S_points = []
        
        # Order of points
        order = rng.permutation(n_points)
        
        for x in order:
            # Check if x can be added
            # Impact of x: for each (y, v) in impact[x], check A[y]
            
            # We need to know if A[y] becomes 0.
            # A[y] loses bit v.
            # If A[y] == (1 << v), it becomes 0.
            # If A[y] has other bits, it's fine.
            
            valid = True
            updates = []
            
            for item in impact[x]:
                y = item // n_dirs
                v = item % n_dirs
                bit = 1 << v
                ay = A[y]
                
                if ay == bit:
                    valid = False
                    break
                elif ay & bit:
                    updates.append((y, bit))
            
            if valid:
                # Apply updates
                for y, bit in updates:
                    A[y] &= ~bit
                S_mask |= (1 << x)
                S_points.append(x)
        
        if len(S_points) > best_S_size:
            best_S_size = len(S_points)
            best_S_points = S_points[:]
            # print(f"New best: {best_S_size}")

    # Construct result array
    # N = F_49^2 \ S
    # We return points in N.
    
    S_set = set(best_S_points)
    points = []
    for idx in range(n_points):
        if idx not in S_set:
            (xa, xb), (ya, yb) = coords[idx]
            points.append([[xa, xb], [ya, yb]])
    
    arr = np.array(points, dtype=np.int64)
    return arr
