import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3
    
    best_config = None
    best_L = None

    # Try a range of L values
    # Estimated L ~ 5.2 - 5.5
    L_values = np.arange(5.2, 5.8, 0.05)
    
    for L in L_values:
        if time.time() > deadline:
            break
            
        S = L - sqrt3
        if S <= 0:
            continue
            
        # Initialize points randomly inside
        # Use a fixed seed for reproducibility? No, random is fine, but let's try a few restarts
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize points
            # Try to initialize in a grid-like fashion to help convergence
            # Or just random
            points = np.array([get_random_point_in_hexagon(S) for _ in range(N)])
            
            # Relaxation
            # Use a cooling schedule for step size
            step_size = 0.5
            for it in range(2000):
                if time.time() > deadline:
                    break
                
                forces = np.zeros_like(points)
                # Repulsion
                for i in range(N):
                    for j in range(i+1, N):
                        d_vec = points[i] - points[j]
                        d = np.linalg.norm(d_vec)
                        if d < min_dist:
                            if d > 1e-6:
                                f = (min_dist - d) / d
                                forces[i] += f * d_vec
                                forces[j] -= f * d_vec
                            else:
                                forces[i] += np.random.randn(2) * 0.5
                                forces[j] -= np.random.randn(2) * 0.5
                            
                # Boundary constraint
                for i in range(N):
                    p = points[i]
                    if not is_in_hexagon(p[0], p[1], S):
                        # Calculate gradient of the constraint violation
                        # Simpler: move towards center
                        forces[i] += -p * 0.5
                        # Also add a small random jitter to escape local minima
                        forces[i] += np.random.randn(2) * 0.01
                
                points += forces * step_size
                step_size *= 0.999 # Slow cooling
                
                # Clamp to region
                for i in range(N):
                    if not is_in_hexagon(points[i][0], points[i][1], S):
                        # Move towards center until inside
                        # Simple bisection towards origin
                        # Or just scale down
                        # Let's find the closest point on the boundary
                        # For now, just move towards origin
                        points[i] = points[i] * 0.95
                        
                # Check validity every 100 steps
                if it % 100 == 0:
                    valid = True
                    for i in range(N):
                        if not is_in_hexagon(points[i][0], points[i][1], S):
                            valid = False
                            break
                        for j in range(i+1, N):
                            d = np.linalg.norm(points[i] - points[j])
                            if d < min_dist - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        found = True
                        break
                        
            if found:
                inner = [[points[i][0], points[i][1], 0.0] for i in range(N)]
                return inner, (0.0, 0.0), L, 0.0

    # Fallback: Return a large safe configuration
    # Grid of 12 hexagons
    # 4x3 grid
    inner = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 1.5
            y = (r - 1.0) * sqrt3
            inner.append([x, y, 0.0])
            
    # Calculate required L for this grid
    # Max x: 1.5 * 1.5 = 2.25. Min x: -2.25.
    # Max y: 1.0 * sqrt3 = 1.732. Min y: -1.732.
    # This is a rectangular-ish block.
    # Bounding hexagon side L must cover this.
    # Distance from origin to (2.25, 1.732) is sqrt(2.25^2 + 1.732^2) = sqrt(5.06 + 3.0) = sqrt(8.06) ~ 2.84.
    # This is the circumradius of the point.
    # The hexagon must contain the unit hexagon centered there.
    # So L must be at least 2.84 + something.
    # Let's just use L = 8.0 to be safe.
    return inner, (0.0, 0.0), 8.0, 0.0

def is_in_hexagon(x, y, S):
    # Hexagon side S, centered at 0,0
    # Vertices at angles 0, 60, ...
    # Inequalities:
    # |x| <= S
    # |y| <= S * sqrt(3) / 2
    # |x/2 + y*sqrt(3)/2| <= S * sqrt(3) / 2
    # |x/2 - y*sqrt(3)/2| <= S * sqrt(3) / 2
    
    sqrt3 = np.sqrt(3)
    if abs(x) > S:
        return False
    if abs(y) > S * sqrt3 / 2:
        return False
    if abs(x/2 + y*sqrt3/2) > S * sqrt3 / 2:
        return False
    if abs(x/2 - y*sqrt3/2) > S * sqrt3 / 2:
        return False
    return True

def get_random_point_in_hexagon(S):
    # Rejection sampling
    while True:
        x = np.random.uniform(-S, S)
        y = np.random.uniform(-S*1.5, S*1.5)
        if is_in_hexagon(x, y, S):
            return np.array([x, y])

# EVOLVE-BLOCK-END
