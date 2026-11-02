import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11

def hex_corners(cx, cy, side, angle_deg):
    """Get 6 corners of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    corners = []
    for i in range(6):
        a = angle + i * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        corners.append([x, y])
    return np.array(corners)

def min_enclosing_hex_side(points, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with given rotation) 
    that contains all points. The hexagon center is free to move.
    
    For a regular hexagon with side s and rotation theta, the 6 face normals are
    at angles theta + i*pi/3. The constraint is that for each face, the projection
    of all points onto the face normal must span at most s*sqrt(3).
    """
    angle = np.radians(outer_angle_deg)
    normals = np.array([[np.cos(angle + i * np.pi / 3), 
                         np.sin(angle + i * np.pi / 3)] for i in range(6)])
    
    projections = points @ normals.T  # (n_points, 6)
    ranges = projections.max(axis=0) - projections.min(axis=0)
    return np.max(ranges) / np.sqrt(3)

def evaluate_packing(inner_positions, inner_angles, outer_angle_deg):
    """Compute minimal outer hexagon side length."""
    all_points = []
    for i in range(len(inner_positions)):
        cx, cy = inner_positions[i]
        corners = hex_corners(cx, cy, 1.0, inner_angles[i])
        all_points.append(corners)
    all_points = np.vstack(all_points)
    return min_enclosing_hex_side(all_points, outer_angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_side = float('inf')
    best_result = None
    
    d = np.sqrt(3)  # center-to-center distance for adjacent hexagons
    
    # Generate hexagonal lattice positions
    # Axial coordinates to cartesian
    def axial_to_cartesian(q, r, size=1):
        x = size * (3.0/2.0 * q)
        y = size * (np.sqrt(3)/2.0 * q + np.sqrt(3) * r)
        return [x, y]
    
    # Generate candidate packings from hexagonal lattice subsets
    # For 11 hexagons, try various cluster shapes
    
    candidates = []
    
    # Hexagonal cluster: center + ring1 (6) + ring2 partial (4)
    # Ring 1 in axial coords: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: positions at distance 2
    
    ring0 = [(0, 0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), 
             (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)]
    
    # Try center + all ring1 + 4 from ring2
    for subset in itertools.combinations(range(len(ring2)), 4):
        axials = ring0 + ring1 + [ring2[i] for i in subset]
        positions = np.array([axial_to_cartesian(q, r, d) for q, r in axials])
        candidates.append(positions)
    
    # Also try other patterns
    # 2-3-3-3 or similar
    # Let's also add some random candidates
    np.random.seed(123)
    for _ in range(100):
        # Random positions in a reasonable area
        positions = np.random.uniform(-3*d, 3*d, size=(11, 2))
        candidates.append(positions)
    
    # Evaluate each candidate with optimized outer angle
    for idx, candidate in enumerate(candidates):
        if time.time() > deadline - 15:
            break
        
        # Quick evaluation with a few outer angles
        best_candidate_side = float('inf')
        best_outer_angle = 0.0
        
        for outer_angle in [0, 15, 30, 45, 60]:
            side = evaluate_packing(candidate, np.zeros(11), float(outer_angle))
            if side < best_candidate_side:
                best_candidate_side = side
                best_outer_angle = float(outer_angle)
        
        if best_candidate_side < best_side:
            # Refine outer angle
            def obj(angle):
                return evaluate_packing(candidate, np.zeros(11), float(angle[0]))
            
            res = minimize(obj, [best_outer_angle], method='Nelder-Mead',
                          options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500})
            if res.fun < best_candidate_side:
                best_candidate_side = res.fun
                best_outer_angle = float(res.x[0])
            
            best_side = best_candidate_side
            best_result = (candidate, np.zeros(11), best_outer_angle)
    
    if best_result is None:
        # Fallback
        positions = np.array([[i * d, 0] for i in range(11)])
        best_result = (positions, np.zeros(11), 0.0)
        best_side = evaluate_packing(positions, np.zeros(11), 0.0)
    
    # Now optimize positions and angles
    if time.time() < deadline - 20:
        init_positions, init_angles, init_outer_angle = best_result
        
        def full_objective(params):
            positions = params[:22].reshape(11, 2)
            angles = params[22:33]
            return evaluate_packing(positions, angles, init_outer_angle)
        
        x0 = np.concatenate([init_positions.flatten(), init_angles])
        
        # Use differential evolution
        bounds = [(-8, 8)] * 22 + [(-90, 90)] * 11
        
        try:
            result = differential_evolution(full_objective, bounds, seed=42,
                                            maxiter=30, popsize=12, tol=1e-4,
                                            polish=True)
            if result.fun < best_side:
                best_side = result.fun
                best_positions = result.x[:22].reshape(11, 2)
                best_angles = result.x[22:33]
            else:
                best_positions = init_positions
                best_angles = init_angles
        except Exception:
            best_positions = init_positions
            best_angles = init_angles
        
        # Also try optimizing outer angle simultaneously
        if time.time() < deadline - 10:
            def full_objective2(params):
                positions = params[:22].reshape(11, 2)
                angles = params[22:33]
                outer_angle = params[33]
                return evaluate_packing(positions, angles, float(outer_angle))
            
            x0 = np.concatenate([best_positions.flatten(), best_angles, [init_outer_angle]])
            bounds2 = [(-8, 8)] * 22 + [(-90, 90)] * 11 + [(-30, 30)]
            
            try:
                result2 = differential_evolution(full_objective2, bounds2, seed=42,
                                                 maxiter=20, popsize=10, tol=1e-4,
                                                 polish=True)
                if result2.fun < best_side:
                    best_side = result2.fun
                    best_positions = result2.x[:22].reshape(11, 2)
                    best_angles = result2.x[22:33]
                    init_outer_angle = float(result2.x[33])
            except Exception:
                pass
    
    # Center the packing
    center = best_positions.mean(axis=0)
    centered_positions = best_positions - center
    
    # Format output
    inner = []
    for i in range(11):
        x = float(centered_positions[i][0])
        y = float(centered_positions[i][1])
        angle = float(best_angles[i])
        inner.append([x, y, angle])
    
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_angle = float(init_outer_angle)
    
    return inner, outer_center, outer_side, outer_angle

