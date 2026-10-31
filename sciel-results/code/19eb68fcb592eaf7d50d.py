import numpy as np
import time
import random

N = 11
# Side length of equilateral triangle with area 1
# Area = (sqrt(3)/4) * s^2 = 1 => s = sqrt(4/sqrt(3))
S = np.sqrt(4.0 / np.sqrt(3.0))
H = S * np.sqrt(3.0) / 2.0

# Precompute indices for all triangles
triangle_indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            triangle_indices.append((i, j, k))
triangle_indices = np.array(triangle_indices)

def bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Triangle vertices: A(0,0), B(S,0), C(S/2, H)
    P = u*A + v*B + w*C where w = 1-u-v
    P = v*(S,0) + (1-u-v)*(S/2, H)
    P_x = v*S + (1-u-v)*S/2 = S/2 + v*S/2 - u*S/2 = S/2 * (1 + v - u)
    P_y = (1-u-v)*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = S / 2.0 * (1.0 + v - u)
    y = H * (1.0 - u - v)
    return np.column_stack((x, y))

def min_triangle_area(bary_points):
    """
    Calculate the minimum area of all triangles formed by the points.
    """
    cart = bary_to_cart(bary_points)
    # Get coordinates for all triangles
    # Shape: (165, 3, 2)
    tri_pts = cart[triangle_indices]
    
    # Calculate area using cross product
    # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
    x1, y1 = tri_pts[:, 0, 0], tri_pts[:, 0, 1]
    x2, y2 = tri_pts[:, 1, 0], tri_pts[:, 1, 1]
    x3, y3 = tri_pts[:, 2, 0], tri_pts[:, 2, 1]
    
    areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    return np.min(areas)

def random_bary_points():
    """
    Generate N random points in barycentric coordinates inside the triangle.
    """
    pts = []
    for _ in range(N):
        u = random.random()
        v = random.random() * (1 - u)
        pts.append([u, v])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_bary = None
    best_score = -1.0
    
    # Initial guess: Grid-like or Symmetric placement might be a good start
    # Let's start with a few random restarts
    
    rng = np.random.default_rng(42)
    
    # Run multiple independent SA runs
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Initialize with random points
        current_bary = random_bary_points()
        # Ensure points are strictly inside (avoid boundary issues in calculation if any, though area is 0 on boundary)
        # Add small jitter
        current_bary[:, 0] += rng.uniform(0.001, 0.005, N)
        current_bary[:, 1] += rng.uniform(0.001, 0.005, N)
        # Re-normalize to be inside
        sum_uv = current_bary[:, 0] + current_bary[:, 1]
        # If sum > 1, scale down
        mask = sum_uv > 1.0
        if np.any(mask):
            current_bary[mask] /= sum_uv[mask][:, np.newaxis] * 1.01
        
        current_score = min_triangle_area(current_bary)
        
        # Simulated Annealing
        T = 0.05
        T_min = 1e-6
        alpha = 0.99
        steps_per_run = 5000
        
        for step in range(steps_per_run):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = rng.integers(0, N)
            old_u, old_v = current_bary[idx]
            
            # Random move in barycentric space
            # Step size decreases slightly or stays constant? 
            # Let's use a step size proportional to T or just fixed small step
            step_size = 0.1 * T + 0.01
            
            new_u = old_u + rng.normal(0, step_size)
            new_v = old_v + rng.normal(0, step_size)
            
            # Constraints: u >= 0, v >= 0, u+v <= 1
            # Simple reflection or clamping
            if new_u < 0:
                new_u = -new_u
            if new_v < 0:
                new_v = -new_v
            if new_u + new_v > 1:
                # Reflect or scale
                # Scale to boundary
                s = 1.0 / (new_u + new_v)
                new_u *= s
                new_v *= s
            
            # Apply move
            old_row = current_bary[idx].copy()
            current_bary[idx] = [new_u, new_v]
            
            new_score = min_triangle_area(current_bary)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_score = new_score
            else:
                # Revert
                current_bary[idx] = old_row
            
            T *= alpha
            if T < T_min:
                T = T_min
            
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Optional: Restart with a cooled schedule or just continue
        # The loop above handles restarts implicitly by starting a new SA run
        
    # Final Polish: Local optimization on the best found
    if best_bary is not None:
        # Do a quick local search with very small steps
        T = 1e-4
        for _ in range(2000):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            old_u, old_v = best_bary[idx]
            step_size = 0.005
            new_u = old_u + rng.normal(0, step_size)
            new_v = old_v + rng.normal(0, step_size)
            
            if new_u < 0: new_u = 0.001
            if new_v < 0: new_v = 0.001
            if new_u + new_v > 1:
                s = 1.0 / (new_u + new_v)
                new_u *= s
                new_v *= s
            
            old_row = best_bary[idx].copy()
            best_bary[idx] = [new_u, new_v]
            new_score = min_triangle_area(best_bary)
            
            if new_score > best_score:
                best_score = new_score
            else:
                best_bary[idx] = old_row
            T *= 0.95

    # Return Cartesian coordinates
    return bary_to_cart(best_bary)

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(min_triangle_area(np.array([[0,0],[1,0],[0,1]]))) # dummy check
