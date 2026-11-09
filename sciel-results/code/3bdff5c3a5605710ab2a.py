# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 14 points in 3D by exploiting D3h symmetry.
    
    Strategy:
    1. Assume the optimal configuration has D3h symmetry: 3 points in a triangle at z=h1,
       3 points in a triangle at z=h2 (rotated by 60 degrees), 3 points at z=h3,
       3 points at z=h4, and 2 points on the z-axis (poles).
       Wait, 3+3+3+3+2 = 14.
       Let's refine the symmetry assumption based on inspiration 1 (score 51.8):
       It mentions D3h or C3v related to triangular prisms.
       
       A common high-symmetry packing for N=14 in 3D often involves layers.
       Let's try a configuration with 3-fold rotational symmetry and a mirror plane.
       
       Candidate Symmetry Structure:
       - Layer 1 (z = z1): 3 points at radius r1, angles 0, 120, 240.
       - Layer 2 (z = z2): 3 points at radius r2, angles 60, 180, 300.
       - Layer 3 (z = z3): 3 points at radius r3, angles 0, 120, 240.
       - Layer 4 (z = z4): 3 points at radius r4, angles 60, 180, 300.
       - Points 5 & 6: On the z-axis at z = z5 and z = z6.
       
       Total points: 3+3+3+3+2 = 14.
       Parameters: r1, r2, r3, r4, z1, z2, z3, z4, z5, z6.
       Constraints: Symmetry reduces dimensionality significantly.
       
       Actually, let's look at a simpler, very strong candidate:
       Two interpenetrating triangular prisms? Or a hexagonal bipyramid + 8 points?
       
       Let's use a flexible parameterization that allows for several symmetric layers.
       We will optimize 10 parameters: 4 radii and 6 z-heights for 6 layers? No, 14 points.
       
       Let's stick to the D3h symmetric layers:
       Layer 1: 3 pts, radius r1, z = z1, angles [0, 120, 240]
       Layer 2: 3 pts, radius r2, z = z2, angles [60, 180, 300]
       Layer 3: 3 pts, radius r3, z = z3, angles [0, 120, 240]
       Layer 4: 3 pts, radius r4, z = z4, angles [60, 180, 300]
       Point 13: (0, 0, z5)
       Point 14: (0, 0, z6)
       
       This covers 14 points.
       We will search over these 10 parameters.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 50s margin

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        r1, r2, r3, r4, z1, z2, z3, z4, z5, z6 = params
        
        # Ensure radii are non-negative
        r1 = abs(r1)
        r2 = abs(r2)
        r3 = abs(r3)
        r4 = abs(r4)
        
        pts = np.zeros((14, 3))
        
        # Layer 1
        angles1 = np.array([0, 2*np.pi/3, 4*np.pi/3])
        pts[0,0] = r1 * np.cos(angles1[0]); pts[0,1] = r1 * np.sin(angles1[0]); pts[0,2] = z1
        pts[1,0] = r1 * np.cos(angles1[1]); pts[1,1] = r1 * np.sin(angles1[1]); pts[1,2] = z1
        pts[2,0] = r1 * np.cos(angles1[2]); pts[2,1] = r1 * np.sin(angles1[2]); pts[2,2] = z1
        
        # Layer 2 (rotated 60 deg)
        angles2 = np.array([np.pi/3, np.pi, 5*np.pi/3])
        pts[3,0] = r2 * np.cos(angles2[0]); pts[3,1] = r2 * np.sin(angles2[0]); pts[3,2] = z2
        pts[4,0] = r2 * np.cos(angles2[1]); pts[4,1] = r2 * np.sin(angles2[1]); pts[4,2] = z2
        pts[5,0] = r2 * np.cos(angles2[2]); pts[5,1] = r2 * np.sin(angles2[2]); pts[5,2] = z2
        
        # Layer 3
        pts[6,0] = r3 * np.cos(angles1[0]); pts[6,1] = r3 * np.sin(angles1[0]); pts[6,2] = z3
        pts[7,0] = r3 * np.cos(angles1[1]); pts[7,1] = r3 * np.sin(angles1[1]); pts[7,2] = z3
        pts[8,0] = r3 * np.cos(angles1[2]); pts[8,1] = r3 * np.sin(angles1[2]); pts[8,2] = z3
        
        # Layer 4 (rotated 60 deg)
        pts[9,0] = r4 * np.cos(angles2[0]); pts[9,1] = r4 * np.sin(angles2[0]); pts[9,2] = z4
        pts[10,0] = r4 * np.cos(angles2[1]); pts[10,1] = r4 * np.sin(angles2[1]); pts[10,2] = z4
        pts[11,0] = r4 * np.cos(angles2[2]); pts[11,1] = r4 * np.sin(angles2[2]); pts[11,2] = z4
        
        # Poles
        pts[12,0] = 0; pts[12,1] = 0; pts[12,2] = z5
        pts[13,0] = 0; pts[13,1] = 0; pts[13,2] = z6
        
        # Center of mass
        pts -= np.mean(pts, axis=0)
        
        dists = pdist(pts)
        if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
            return 1e10
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        return -ratio * ratio

    best_score = -np.inf
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Initial guesses based on geometric intuition
    # Try a few structured starting points
    starts = []
    
    # 1. Symmetric distribution
    # z: -1, -0.5, 0, 0.5, 1...
    starts.append([0.5, 0.5, 0.5, 0.5, -1.0, -0.5, 0.5, 1.0, -1.5, 1.5])
    starts.append([0.8, 0.8, 0.8, 0.8, -0.8, -0.4, 0.4, 0.8, -1.2, 1.2])
    starts.append([0.3, 0.6, 0.6, 0.3, -1.0, -0.3, 0.3, 1.0, -1.5, 1.5])
    
    # 2. Random starts
    for _ in range(20):
        r = rng.uniform(0.2, 1.0, 4)
        z = rng.uniform(-1.5, 1.5, 6)
        # Sort z to maintain some ordering for stability
        z = np.sort(z)
        starts.append(np.concatenate([r, z]))
        
    # Grid search over a coarse parameter space to find good basins
    # We will perform a coarse grid search on the z-heights and radii
    # To save time, we'll do a randomized search with local refinement
    
    num_starts = len(starts)
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        x0 = np.array(starts[i])
        
        try:
            # Use L-BFGS-B for speed
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < -1e9:
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
                    
        except Exception:
            continue
            
    # If we haven't found a good solution, try a second pass with more random starts
    # or refine the best solution found
    
    if best_params is None:
        # Fallback to icosahedron + poles if symmetry search fails completely
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts = ico_verts / norms[:, np.newaxis]
        points = np.zeros((14, 3))
        points[:12] = ico_verts
        points[12] = [0, 0, 1]
        points[13] = [0, 0, -1]
        points -= np.mean(points, axis=0)
        return points

    # Reconstruct points from best_params
    r1, r2, r3, r4, z1, z2, z3, z4, z5, z6 = best_params
    r1 = abs(r1); r2 = abs(r2); r3 = abs(r3); r4 = abs(r4)
    
    pts = np.zeros((14, 3))
    angles1 = np.array([0, 2*np.pi/3, 4*np.pi/3])
    angles2 = np.array([np.pi/3, np.pi, 5*np.pi/3])
    
    pts[0,0] = r1 * np.cos(angles1[0]); pts[0,1] = r1 * np.sin(angles1[0]); pts[0,2] = z1
    pts[1,0] = r1 * np.cos(angles1[1]); pts[1,1] = r1 * np.sin(angles1[1]); pts[1,2] = z1
    pts[2,0] = r1 * np.cos(angles1[2]); pts[2,1] = r1 * np.sin(angles1[2]); pts[2,2] = z1
    
    pts[3,0] = r2 * np.cos(angles2[0]); pts[3,1] = r2 * np.sin(angles2[0]); pts[3,2] = z2
    pts[4,0] = r2 * np.cos(angles2[1]); pts[4,1] = r2 * np.sin(angles2[1]); pts[4,2] = z2
    pts[5,0] = r2 * np.cos(angles2[2]); pts[5,1] = r2 * np.sin(angles2[2]); pts[5,2] = z2
    
    pts[6,0] = r3 * np.cos(angles1[0]); pts[6,1] = r3 * np.sin(angles1[0]); pts[6,2] = z3
    pts[7,0] = r3 * np.cos(angles1[1]); pts[7,1] = r3 * np.sin(angles1[1]); pts[7,2] = z3
    pts[8,0] = r3 * np.cos(angles1[2]); pts[8,1] = r3 * np.sin(angles1[2]); pts[8,2] = z3
    
    pts[9,0] = r4 * np.cos(angles2[0]); pts[9,1] = r4 * np.sin(angles2[0]); pts[9,2] = z4
    pts[10,0] = r4 * np.cos(angles2[1]); pts[10,1] = r4 * np.sin(angles2[1]); pts[10,2] = z4
    pts[11,0] = r4 * np.cos(angles2[2]); pts[11,1] = r4 * np.sin(angles2[2]); pts[11,2] = z4
    
    pts[12,0] = 0; pts[12,1] = 0; pts[12,2] = z5
    pts[13,0] = 0; pts[13,1] = 0; pts[13,2] = z6
    
    # Center
    pts -= np.mean(pts, axis=0)
    
    # Scale to unit max distance
    dists = pdist(pts)
    d_max = np.max(dists)
    if d_max > 1e-12:
        pts = pts / d_max
        
    # Final check
    if not np.all(np.isfinite(pts)):
        # Fallback
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts = ico_verts / norms[:, np.newaxis]
        pts = np.zeros((14, 3))
        pts[:12] = ico_verts
        pts[12] = [0, 0, 1]
        pts[13] = [0, 0, -1]
        pts -= np.mean(pts, axis=0)
        
    return pts

# EVOLVE-BLOCK-END
