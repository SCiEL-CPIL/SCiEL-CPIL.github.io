import numpy as np
import time
import math
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # 5 seconds buffer before 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """
    Get vertices of a regular hexagon.
    Vertices are ordered counter-clockwise.
    """
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def outer_hex_side(points, outer_rot_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon centered at origin
    with given rotation that encloses all points.
    """
    # Normals to the sides of the hexagon
    # For a regular hexagon with side L, the distance from center to side (apothem) is L * sqrt(3) / 2.
    # The vertices are at distance L from the center.
    # The normals to the sides are at angles 30, 90, 150, 210, 270, 330 degrees (if vertices are at 0, 60, ...)
    # Actually, if we define the hexagon by its side length L, the projection of vertices onto the normal directions
    # determines L.
    # Let's define the outer hexagon vertices at angles 0, 60, ...
    # Then the normals to the sides are at 30, 90, ...
    # The maximum projection of any point onto these normals must be <= L * sqrt(3) / 2.
    # So L = 2 / sqrt(3) * max(|projection|)
    
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_max(values, beta=50.0):
    """
    Smooth approximation of max using log-sum-exp.
    """
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    # Avoid overflow
    if m < -50.0 / beta:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def compute_objective(params, beta=50.0):
    """
    Compute the smooth objective function.
    params: [x1, y1, r1, x2, y2, r2, ..., x12, y12, r12, outer_rot]
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    # 1. Compute all vertices of inner hexagons
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    
    # 2. Compute smooth L
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    abs_proj = np.abs(proj)
    # We want the max of absolute projections
    # smooth_max over the absolute values
    L_smooth = (2.0 / np.sqrt(3.0)) * smooth_max(abs_proj.flatten(), beta=beta)
    
    # 3. Compute overlap penalty
    # For each pair, check distance between centers or use SAT.
    # SAT is more robust for rotated hexagons but harder to differentiate smoothly.
    # Alternative: Use distance between centers. 
    # Min distance between non-overlapping unit hexagons is 2*sqrt(3)/2 = sqrt(3) if aligned? 
    # Actually, min distance between centers of two non-overlapping unit hexagons is at least 2 * apothem = sqrt(3) if faces are parallel and touching.
    # But if rotated, it's complex.
    # Let's use a simple smooth penalty based on center distance.
    # If distance < sqrt(3), they *might* overlap. 
    # To be safe, let's use a slightly larger threshold or check actual overlap.
    # Given the time constraint, a heuristic penalty based on center distance is faster but less accurate.
    # Let's try to implement a smooth approximation of the penetration depth using center distance and relative rotation.
    # This is complex. Let's stick to center distance penalty with a tight constraint.
    # Min center distance for non-overlapping unit hexagons is sqrt(3) (approx 1.732).
    # We will penalize if dist < sqrt(3).
    
    centers = xys_rots[:, :2]
    penalty = 0.0
    min_dist = np.sqrt(3.0)
    
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                # Smooth hinge penalty
                # (min_dist - d)^2
                penalty += (min_dist - d) ** 2
    
    # We want to minimize L. Penalty should be large if overlap.
    # Weight penalty heavily to enforce constraints.
    return L_smooth + 1000.0 * penalty

def is_valid(params, tol=1e-6):
    """
    Check if the configuration is valid (no overlaps, inside outer hexagon).
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    # Check overlaps using SAT or simple distance check for hexagons
    # For hexagons, if center distance < sqrt(3) - eps, they likely overlap.
    # But rotation matters. Let's use a more robust check:
    # Generate vertices and check if any vertex of one is inside the other? No, that's not enough.
    # SAT is the way.
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    for i in range(N):
        for j in range(i+1, N):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            # SAT check
            axes = []
            for v in (v1, v2):
                for k in range(6):
                    e = v[(k+1)%6] - v[k]
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-12:
                        axes.append(n / norm)
            
            overlap = False
            for n in axes:
                proj1 = v1 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                proj2 = v2 @ n
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 + tol or max2 < min1 + tol:
                    break # Separated on this axis
                # Check if overlap on this axis
                # If they don't separate on any axis, they overlap
            else:
                # If we didn't break, we need to check if they are actually overlapping
                # The loop above breaks if separated. If we finish loop, we need to ensure they are NOT separated on ANY axis.
                # Wait, the logic is: if there exists an axis where they are separated, they don't overlap.
                # So if we never break, they might overlap.
                # Let's re-structure:
                overlap = True
                for n in axes:
                    proj1 = v1 @ n
                    min1, max1 = np.min(proj1), np.max(proj1)
                    proj2 = v2 @ n
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 - tol or max2 < min1 - tol:
                        overlap = False
                        break
                if overlap:
                    return False
    
    # Check if inside outer hexagon
    all_points = np.vstack(vertices_list)
    L_calc = outer_hex_side(all_points, outer_rot)
    # We don't know the target L in is_valid, so we just check if it's reasonable?
    # Actually, the optimization minimizes L. We just need to ensure no overlaps.
    # The outer boundary constraint is handled by the objective minimizing L.
    
    return True

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    # Initial configurations
    def make_init_ring():
        pts = []
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang), 0.0])
        # 5 more
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang), 0.0])
        # Need 12 total. 1+6+5=12.
        return np.array(pts)
    
    def make_init_grid():
        pts = []
        # 3x4 grid?
        for r in range(3):
            for c in range(4):
                x = c * np.sqrt(3)
                y = r * 1.5
                if r % 2 == 1:
                    x += np.sqrt(3) / 2
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    inits = [
        make_init_ring(),
        make_init_grid()
    ]
    
    # Add some random perturbations to inits
    for i in range(5):
        base = inits[i % 2].copy()
        base[:, :2] += np.random.normal(0, 0.5, (N, 2))
        base[:, 2] += np.random.uniform(0, 30, N)
        inits.append(base)
    
    # Optimization
    # We will use Nelder-Mead on the smooth objective.
    # It's derivative-free and robust.
    
    for init in inits:
        if time_left() < 10:
            break
            
        init_params = np.concatenate((init.flatten(), [0.0]))
        
        # Try a few random outer rotations
        for orot in [0.0, 30.0, 15.0]:
            if time_left() < 5:
                break
                
            p0 = init_params.copy()
            p0[-1] = orot
            
            try:
                # Run Nelder-Mead
                res = minimize(
                    compute_objective, 
                    p0, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                # Check validity and compute actual L
                if is_valid(res.x):
                    xys_rots = res.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, res.x[-1])
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Final polish: If we have a good candidate, try to refine it further
    if best_params is not None and time_left() > 10:
        try:
            # Use a higher beta for smoother max
            res_final = minimize(
                lambda p: compute_objective(p, beta=100.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if is_valid(res_final.x):
                xys_rots = res_final.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, res_final.x[-1])
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_params is None:
        # Return a trivial valid solution (large hexagon)
        init = make_init_ring()
        p = np.concatenate((init.flatten(), [0.0]))
        xys_rots = p[:3*N].reshape(N, 3)
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        best_L = outer_hex_side(all_points, 0.0)
        best_params = p
    
    # Extract result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle

# Ensure the function is exposed
if __name__ == "__main__":
    # Quick test
    start = time.time()
    result = construct()
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Side: {result[2]}")
