# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Use a highly symmetric initial configuration: a regular tetrahedron scaled to unit circumradius,
       with 3 additional points at edge midpoints and 1 point at the centroid.
       Wait, that's 4 + 3 + 1 = 8 points. We need 14.
       
       Let's rethink the "tetrahedron expansion" idea to get 14 points.
       A regular tetrahedron has 4 vertices, 6 edges, 4 faces, 1 centroid.
       Total "feature points" = 4 + 6 + 4 + 1 = 15.
       If we omit one face centroid, we get 14 points.
       
       Alternatively, the prompt suggests: "regular tetrahedron scaled to unit circumradius, with three 
       additional points placed at the midpoints of each edge of the tetrahedron and one point at the 
       centroid". 
       4 (verts) + 3 (midpoints? no, there are 6 edges) + 1 (centroid) = 8 points.
       The prompt's description is a bit ambiguous on how to get 14. 
       "three additional points placed at the midpoints of each edge" - maybe it means 3 specific edges?
       Or maybe it implies a different structure.
       
       Let's look at the actual instruction: "Represent the 14 points as a regular tetrahedron scaled to 
       unit circumradius, with three additional points placed at the midpoints of each edge of the 
       tetrahedron and one point at the centroid, creating a highly symmetric configuration with only 4 
       free scaling parameters that can be optimized via grid search to minimize the distance variance."
       
       This description sums to 4+3+1=8 points. This is likely a truncated description of a more complex 
       symmetric set, or perhaps it refers to a specific subset. However, 14 points is the requirement.
       
       Let's assume the strategy is to use a symmetric configuration that can be parameterized by a few 
       variables. A common high-symmetry set for 14 points might be related to the icosahedron (12) + 2 
       poles, which was in the previous program. But the directive says to use the tetrahedron-based one.
       
       Perhaps the "14 points" in the directive is a mistake and it meant a smaller number, OR it implies 
       a more complex construction. Let's try to construct a 14-point set with high symmetry.
       
       Actually, let's look at the "4 free scaling parameters" part.
       If we take the 4 vertices of a tetrahedron, their positions are fixed by the circumradius.
       If we add points, we need to scale them.
       
       Let's try a different interpretation: Maybe the "14 points" are distributed on the symmetry elements 
       of the tetrahedron.
       Vertices: 4
       Edge midpoints: 6
       Face centroids: 4
       Body centroid: 1
       Total: 15.
       
       If we remove the body centroid, we have 14 points: 4 vertices + 6 edge midpoints + 4 face centroids.
       This set has Td symmetry (or a subset if scaled differently).
       
       The "4 free scaling parameters" could be:
       1. Scale of vertices (fixed to 1 if circumradius is 1)
       2. Scale of edge midpoints (distance from center)
       3. Scale of face centroids (distance from center)
       
       Wait, the distance of an edge midpoint from the center of a regular tetrahedron with circumradius R is 
       R/2.
       The distance of a face centroid from the center is R/3.
       
       So if we allow these distances to vary, we have 2 free parameters (scale_mid, scale_face) plus the 
       overall scale (which doesn't affect the ratio) and maybe a global rotation (which doesn't affect 
       distances).
       
       Let's parameterize:
       - 4 vertices at radius R_v
       - 6 edge midpoints at radius R_m
       - 4 face centroids at radius R_f
       
       Since the metric is scale-invariant, we can fix R_v = 1.
       Then we optimize R_m and R_f.
       This gives 2 free parameters. The prompt says 4. Maybe it allows for different scaling of different 
       types of midpoints? No, symmetry suggests they are all equal.
       
       Perhaps the "4 free scaling parameters" refers to something else.
       Let's stick to the 14-point set: 4 Vertices, 6 Edge Midpoints, 4 Face Centroids.
       
       Initial values:
       R_v = 1
       R_m = 0.5 (for a regular tetrahedron, edge midpoint is at half the radius? No.
       Let's calculate.
       Tetrahedron vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
       Norm = sqrt(3).
       Unit vertices: v1 = (1,1,1)/sqrt(3), etc.
       Midpoint of v1, v2: ((1,1,1) + (1,-1,-1))/2 = (1,0,0). Norm = 1.
       Wait. The midpoint of (1,1,1) and (1,-1,-1) is (1,0,0). The norm is 1.
       So for a tetrahedron with circumradius 1, the edge midpoints are also at radius 1?
       Let's check.
       v1 = (1,1,1)/sqrt(3). v2 = (1,-1,-1)/sqrt(3).
       mid = (2,0,0)/(2*sqrt(3)) = (1,0,0)/sqrt(3).
       Norm of mid = 1/sqrt(3).
       Ah, I messed up the arithmetic.
       v1 = (1,1,1)/sqrt(3).
       v2 = (1,-1,-1)/sqrt(3).
       mid = ((1+1)/2, (1-1)/2, (1-1)/2) / sqrt(3) = (1, 0, 0) / sqrt(3).
       Norm = 1/sqrt(3).
       
       So for a regular tetrahedron with circumradius 1:
       Vertex radius = 1
       Edge midpoint radius = 1/sqrt(3) ≈ 0.577
       Face centroid:
       Face 1: v1, v2, v3.
       Centroid = (v1+v2+v3)/3.
       v1=(1,1,1), v2=(1,-1,-1), v3=(-1,1,-1) (unscaled)
       Sum = (1,1,-1).
       Centroid = (1,1,-1)/3.
       Norm of (1,1,-1) is sqrt(3).
       So centroid radius (unscaled) = sqrt(3)/3 = 1/sqrt(3).
       Scaled by 1/sqrt(3) for circumradius 1:
       Radius = (1/sqrt(3)) * (1/sqrt(3)) = 1/3.
       
       So:
       R_v = 1
       R_m = 1/sqrt(3)
       R_f = 1/3
       
       We can optimize R_m and R_f.
       Let's define the points based on these parameters.
       """
    start_time = time.time()
    deadline = start_time + 590

    # Define the base tetrahedron vertices (unscaled)
    t_verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    
    # Normalize to unit circumradius
    t_verts = t_verts / np.linalg.norm(t_verts, axis=1)[:, np.newaxis]
    
    # Calculate edge midpoints (unscaled relative to unit circumradius)
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    t_mids = np.array([(t_verts[i] + t_verts[j]) / 2.0 for i, j in edges])
    # t_mids are already at radius 1/sqrt(3) if t_verts are at radius 1.
    # Let's verify.
    # t_verts[0] = (1,1,1)/sqrt(3). t_verts[1] = (1,-1,-1)/sqrt(3).
    # mid = (2,0,0)/(2*sqrt(3)) = (1,0,0)/sqrt(3). Norm = 1/sqrt(3). Correct.
    
    # Calculate face centroids (unscaled relative to unit circumradius)
    # Faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    t_faces = np.array([(t_verts[i] + t_verts[j] + t_verts[k]) / 3.0 for i, j, k in faces])
    # t_faces are at radius 1/3.
    
    def build_points(params):
        """
        params: [scale_mids, scale_faces]
        Returns array of 14 points.
        """
        sm, sf = params
        # Vertices are fixed at scale 1
        pts_v = t_verts.copy()
        
        # Scale midpoints
        # Base radius is 1/sqrt(3). We want to scale the position vector.
        # pos = base_pos * (sm / (1/sqrt(3))) ? 
        # Or just pos = base_pos * sm?
        # If sm = 1, we get the regular tetrahedron midpoints.
        # Let's define sm as the multiplier on the base position vector.
        pts_m = t_mids * sm
        
        # Scale faces
        pts_f = t_faces * sf
        
        points = np.vstack([pts_v, pts_m, pts_f])
        return points

    def evaluate(points):
        """
        Returns (d_min/d_max)^2
        """
        if not np.all(np.isfinite(points)):
            return 0.0
        pts = points.copy()
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def objective(params):
        """
        Objective for optimization (minimize negative metric).
        """
        pts = build_points(params)
        metric = evaluate(pts)
        return -metric

    # Initial guess: Regular tetrahedron geometry
    # sm = 1.0, sf = 1.0 corresponds to the regular configuration.
    # Let's try a grid search or differential evolution on sm, sf.
    
    # Bounds for sm, sf
    # They shouldn't be negative.
    bounds = [(0.1, 2.0), (0.1, 2.0)]
    
    best_score = 0.0
    best_params = [1.0, 1.0]
    best_points = build_points(best_params)
    
    # Evaluate initial
    init_score = evaluate(best_points)
    if init_score > best_score:
        best_score = init_score

    # Use Differential Evolution for global search in 2D parameter space
    # This is very fast for 2D.
    try:
        de_res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8, 
            polish=False # We will polish manually
        )
        if de_res.fun < -best_score:
            best_params = de_res.x
            best_points = build_points(best_params)
            best_score = -de_res.fun
    except Exception:
        pass

    # Refine with Nelder-Mead
    try:
        nm_res = minimize(
            objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000}
        )
        if nm_res.fun < -best_score:
            best_params = nm_res.x
            best_points = build_points(best_params)
            best_score = -nm_res.fun
    except Exception:
        pass

    # Ensure points are finite and valid
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular tetrahedron set
        best_points = build_points([1.0, 1.0])
        
    return best_points

# EVOLVE-BLOCK-END
