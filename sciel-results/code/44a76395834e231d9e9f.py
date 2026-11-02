import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices_vectorized(centers, angles_deg, side=1.0):
    """
    Centers: (N, 2)
    Angles: (N,)
    Returns: (N, 6, 2)
    """
    angles_rad = np.radians(angles_deg)
    # Vertex angles relative to center: 0, 60, ..., 300
    # This corresponds to a hexagon with a vertex pointing right (along +x) when angle=0
    # Wait, standard math: cos(0)=1, sin(0)=0. Vertex at (1,0).
    # Next: cos(60)=0.5, sin(60)=sqrt(3)/2.
    # This is a "pointy-side" hexagon if x is horizontal? 
    # Let's stick to the definition: vertices at angles theta + k*pi/3.
    
    theta = angles_rad[:, np.newaxis] + np.linspace(0, 5*np.pi/3, 6)[np.newaxis, :]
    x = centers[:, 0:1] + side * np.cos(theta)
    y = centers[:, 1:2] + side * np.sin(theta)
    
    verts = np.stack((x, y), axis=-1)
    return verts

def _hex_normals_vectorized(angles_deg):
    """
    Returns (N, 3, 2) normals for the 3 pairs of parallel sides.
    For a hexagon with vertices at angles a + k*60, the side normals are at a + 30 + k*60.
    """
    angles_rad = np.radians(angles_deg)
    theta = angles_rad[:, np.newaxis] + np.radians(30 + np.array([0, 60, 120])[np.newaxis, :])
    nx = np.cos(theta)
    ny = np.sin(theta)
    normals = np.stack((nx, ny), axis=-1)
    return normals

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, os, oang]
    """
    n_inner = N
    # Extract inner parameters
    inner_centers = x[:n_inner*3:3]
    inner_ys = x[1:n_inner*3:3]
    inner_angles = x[2:n_inner*3:3]
    
    # Reconstruct centers array
    centers = np.column_stack((inner_centers, inner_ys))
    
    # Extract outer parameters
    ocx = x[n_inner*3]
    ocy = x[n_inner*3+1]
    os = x[n_inner*3+2]
    oang = x[n_inner*3+3]
    
    # 1. Containment Penalty
    # Vertices of inner hexagons must be inside outer hexagon.
    # It is sufficient to check vertices for convex polygons? 
    # If all vertices of Convex A are inside Convex B, then A is inside B.
    # Yes.
    inner_verts = _hex_vertices_vectorized(centers, inner_angles, side=1.0)
    # inner_verts shape: (N, 6, 2)
    
    # Outer hexagon parameters
    outer_center = np.array([[ocx, ocy]])
    outer_angle = np.array([oang])
    outer_normals = _hex_normals_vectorized(outer_angle) # (1, 3, 2)
    apothem_outer = os * SQRT3 / 2.0
    
    # Calculate distance of each inner vertex to outer hexagon
    # Relative position of inner verts w.r.t outer center
    rel_verts = inner_verts - outer_center # (N, 6, 2)
    
    # Project onto outer normals
    # rel_verts: (N, 6, 2)
    # outer_normals: (1, 3, 2) -> broadcast to (N, 6, 3)
    # proj = rel_verts @ outer_normals.T -> (N, 6, 3)
    proj = np.einsum('ijk,kj->ij', rel_verts, outer_normals[0].T)
    
    # Violation if proj > apothem or proj < -apothem
    # Since we want min side length, we expect most verts to be close to boundary.
    # We can just penalize max(0, proj - apothem) and max(0, -proj - apothem)
    
    viol_pos = np.maximum(0, proj - apothem_outer)
    viol_neg = np.maximum(0, -proj - apothem_outer)
    
    containment_penalty = np.sum(viol_pos**2) + np.sum(viol_neg**2)
    
    # 2. Pairwise Overlap Penalty
    # Use SAT (Separating Axis Theorem) for Hexagons.
    # Two convex polygons do not overlap if there is a separating axis.
    # For regular hexagons, the potential separating axes are the normals of the sides of both hexagons.
    # So we check 6 normals from H1 and 6 normals from H2 (total 12, but 3 are redundant per hex due to symmetry, so 6 unique directions per hex? 
    # Actually, for two hexagons, we need to check normals of both. 3+3=6 unique directions if aligned, but they are rotated differently.
    # So we check 3 normals from H1 and 3 from H2. Total 6 axes.
    
    # Precompute normals for all inner hexagons
    inner_normals = _hex_normals_vectorized(inner_angles) # (N, 3, 2)
    apothem_inner = 1.0 * SQRT3 / 2.0
    
    overlap_penalty = 0.0
    
    # Vectorize pairwise checks? 
    # N=11, so 55 pairs. 
    # Doing it in a loop is fine if the inner part is fast.
    
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            # Quick reject: distance between centers
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.0: # (2*side)^2
                continue
            
            # Get normals for i and j
            ni = inner_normals[i] # (3, 2)
            nj = inner_normals[j] # (3, 2)
            
            # All candidate axes
            axes = np.vstack((ni, nj)) # (6, 2)
            
            # Project both hexagons onto these axes
            # We need the extent of each hexagon on each axis.
            # For a regular hexagon centered at (0,0) with side 1, the projection onto a normal vector n (unit) is [-apothem, apothem].
            # Wait, this is only true if the axis is one of the hexagon's own normals.
            # If the axis is NOT a normal of the hexagon, the extent is different.
            # Specifically, for a hexagon, the support function h(u) = max(v . u) for v in vertices.
            # For a unit normal n_k of the hexagon, h(n_k) = apothem.
            # For other directions, it's larger.
            
            # Let's compute the actual extent for each axis.
            # Vertices of i: vi (6, 2)
            # Vertices of j: vj (6, 2)
            # We can get vertices from centers and angles, but we already have them?
            # No, we only computed inner_verts for all i.
            # vi = inner_verts[i]
            # vj = inner_verts[j]
            
            vi = inner_verts[i]
            vj = inner_verts[j]
            
            # Project vi onto axes
            # proj_i = vi @ axes.T -> (6, 6)
            # min_i = min(proj_i, axis=0)
            # max_i = max(proj_i, axis=0)
            
            # This is 6x6 dot product.
            proj_i = vi @ axes.T
            min_i = np.min(proj_i, axis=0)
            max_i = np.max(proj_i, axis=0)
            
            proj_j = vj @ axes.T
            min_j = np.min(proj_j, axis=0)
            max_j = np.max(proj_j, axis=0)
            
            # Shift by center difference?
            # The projections are in global coordinates.
            # So we just check if intervals [min_i, max_i] and [min_j, max_j] overlap for all axes.
            # If for any axis, max_i < min_j or max_j < min_i, they are separated.
            
            # Overlap on axis k: min(max_i[k], max_j[k]) - max(min_i[k], min_j[k])
            # If this is < 0 for any k, no overlap.
            
            # We want a smooth penalty.
            # If they overlap on all axes, penalty > 0.
            # A common penalty is the minimum overlap depth.
            # But we want to push them apart.
            # Let's use the sum of squared penetrations? Or just min penetration?
            # Min penetration is non-differentiable.
            # Sum of squared penetrations is differentiable.
            
            # Calculate overlap depth for each axis
            # depth_k = min(max_i[k], max_j[k]) - max(min_i[k], min_j[k])
            
            min_max = np.minimum(max_i, max_j)
            max_min = np.maximum(min_i, min_j)
            depths = min_max - max_min
            
            # If depth < 0, they are separated on that axis.
            # If all depths > 0, they overlap.
            # The "penetration" is the minimum depth.
            # To make it smooth, we can use a soft min or just sum of squares of negative depths?
            # No, if they overlap, depths are positive.
            # We want to penalize positive depths.
            # But if they are separated, depth is negative.
            
            # Actually, if they overlap, the minimum depth is the penetration distance.
            # If we minimize the sum of squares of the minimum depth, it might work.
            # But min is non-diff.
            
            # Alternative: Use the sum of squares of all positive depths?
            # This penalizes overlap heavily.
            # Or, simpler: If they overlap, add a penalty proportional to the square of the minimum depth.
            # To avoid min, we can use a differentiable approximation or just check if min_depth > 0.
            
            # For DE, non-differentiability is okay, but smoothness helps.
            # Let's use: penalty = max(0, min_depths)^2
            # min_depths = np.min(depths)
            # If min_depths > 0, they overlap.
            
            min_depth = np.min(depths)
            if min_depth > 0:
                overlap_penalty += min_depth * min_depth
                
    # Total objective
    # We want to minimize os.
    # If there is any violation (containment or overlap), we add a large penalty.
    # The penalty should dominate os.
    
    total_penalty = containment_penalty + overlap_penalty
    
    # If penalty is 0, we return os.
    # If penalty > 0, we return os + large * penalty.
    # This encourages valid solutions, and among valid ones, minimizes os.
    
    return os + 1000.0 * total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Lattice packing
    init_inner = []
    # Center
    init_inner.append((0.0, 0.0, 0.0))
    # Ring 1
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
    # Ring 2 (4 more to make 11)
    # Just pick 4 random-ish spots far enough
    init_inner.append((3.0, 0.0, 0.0))
    init_inner.append((2.25, 1.9, 0.0))
    init_inner.append((0.75, 2.6, 0.0))
    init_inner.append((-1.5, 2.6, 0.0))
    
    x0 = []
    for p in init_inner:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
        
    # Outer init: large enough to contain
    x0.extend([0.0, 0.0, 5.0, 0.0])
    
    # Bounds
    # Inner centers: [-10, 10]
    # Angles: [0, 360]
    # Outer center: [-10, 10]
    # Outer side: [3.0, 10.0] # Min possible is roughly 2.5?
    # Outer angle: [0, 360]
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
        
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((3.0, 10.0))
    bounds.append((0, 360))
    
    best_x = None
    best_val = float('inf')
    
    # Run DE
    # N=11, 3*11 + 4 = 37 variables.
    # Vectorized objective should be fast.
    # Let's try to run for a few seconds.
    
    try:
        # popsize=15, maxiter=50 might be too slow if objective is not fast enough.
        # Let's start with popsize=10, maxiter=30.
        res = differential_evolution(_objective, bounds, maxiter=30, popsize=10, seed=42, polish=False)
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Check time
    if time.time() > deadline:
        if best_x is not None:
            return _format_result(best_x)
        else:
            return _format_result(x0)
            
    # Local optimization (Nelder-Mead) to polish
    if best_x is not None:
        try:
            from scipy.optimize import minimize
            res2 = minimize(_objective, best_x, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception as e:
            print(f"Minimize failed: {e}")
            
    return _format_result(best_x if best_x is not None else x0)

def _format_result(x):
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner.append([cx, cy, a])
        
    ocx = x[N*3]
    ocy = x[N*3+1]
    os = x[N*3+2]
    oang = x[N*3+3]
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
