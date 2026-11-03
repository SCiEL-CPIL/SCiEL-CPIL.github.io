"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = np.deg2rad(angle_deg)
    angles = theta + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_containment(vertices, ocx, ocy, oside, oangle_deg):
    """
    For a set of vertices, compute the minimum outer side length such that all vertices
    are inside the outer hexagon centered at (ocx,ocy) with rotation oangle_deg.
    We use the support function: for each vertex, compute the maximum over the 6 face normals
    of the dot product, which gives the required side length for that vertex.
    The overall required side length is the max over all vertices.
    """
    theta = np.deg2rad(oangle_deg)
    # Face normals of a regular hexagon: directions at angles theta + k*pi/3 for k=0..5
    # Actually, the outward normals to the faces of a regular hexagon point at angles:
    # theta + k * pi/3 for k = 0, 1, ..., 5
    # A point p is inside the hexagon (centered at origin, side s) iff for all k:
    #   dot(p, n_k) <= s * cos(pi/6) = s * sqrt(3)/2
    # where n_k are unit outward normals.
    # So the required s for point p is: max_k dot(p - c, n_k) / (sqrt(3)/2)
    
    normals = np.column_stack([np.cos(theta + np.pi * np.arange(6) / 3.0),
                               np.sin(theta + np.pi * np.arange(6) / 3.0)])
    # vertices: (M, 2), ocx, ocy: center of outer hex
    rel = vertices - np.array([ocx, ocy])  # (M, 2)
    # dots: (M, 6)
    dots = rel @ normals.T
    # required side for each vertex: max over k of dots[m,k] / (sqrt(3)/2)
    max_dots = np.max(dots, axis=1)  # (M,)
    required = max_dots / (np.sqrt(3.0) / 2.0)
    return np.max(required)

def _smooth_outer_side(vertices, ocx, ocy, oside_init, oangle_deg, beta=50.0):
    """
    Smooth surrogate for the outer side length using log-sum-exp.
    Returns a differentiable approximation of the max required side.
    """
    theta = np.deg2rad(oangle_deg)
    normals = np.column_stack([np.cos(theta + np.pi * np.arange(6) / 3.0),
                               np.sin(theta + np.pi * np.arange(6) / 3.0)])
    rel = vertices - np.array([ocx, ocy])
    dots = rel @ normals.T  # (M, 6)
    # For each vertex, the required contribution is max over normals
    # We want a smooth max over all (vertex, normal) pairs of dot/(sqrt(3)/2)
    all_dots = dots.flatten()  # (M*6,)
    # Smooth max via log-sum-exp
    max_val = np.max(all_dots)
    if max_val == 0:
        smooth_max = 0.0
    else:
        # log-sum-exp for stability
        shifted = all_dots - max_val
        lse = max_val + np.log(np.sum(np.exp(shifted * beta)) / (len(all_dots) * beta))
        # Actually, the standard smooth max is (1/beta)*log(sum(exp(beta*x_i)))
        # Let me redo:
        lse = (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted))) + max_val
    return lse / (np.sqrt(3.0) / 2.0)

def _objective(params, N_inner=12, oangle_deg=0.0, beta=50.0):
    """
    Objective: smooth outer side length.
    params: array of length 3*N_inner, each triplet is (x, y, angle_deg) for one inner hex.
    """
    ocx, ocy = 0.0, 0.0  # We can fix outer center at origin by symmetry, or optimize it
    # Actually, let's allow outer center to be optimized too. But for now, fix at 0.
    # Let's include outer center in params? No, let's keep it simple: fix outer at origin.
    
    all_vertices = []
    for i in range(N_inner):
        x, y, ang = params[3*i:3*i+3]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # (72, 2)
    
    side = _smooth_outer_side(all_vertices, ocx, ocy, 0.0, oangle_deg, beta)
    return side

def _objective_with_center(params, N_inner=12, oangle_deg=0.0, beta=50.0):
    """
    Objective with outer center as part of params.
    params: [ocx, ocy, x1, y1, a1, x2, y2, a2, ...]
    """
    ocx, ocy = params[0], params[1]
    rest = params[2:]
    all_vertices = []
    for i in range(N_inner):
        x, y, ang = rest[3*i:3*i+3]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    side = _smooth_outer_side(all_vertices, ocx, ocy, 0.0, oangle_deg, beta)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_score = float('inf')
    best_params = None
    
    # Strategy: 
    # 1. Try several initial configurations using grid/random
    # 2. For each, run L-BFGS to polish
    # 3. Keep track of best
    
    N_inner = N
    oangle_deg = 0.0  # Fix outer rotation at 0 for simplicity
    
    # Initial configurations:
    # Try placing hexagons in a hexagonal lattice pattern
    # Also try random configurations
    
    rng = np.random.RandomState(42)
    
    candidates = []
    
    # Configuration 1: Hexagonal lattice
    # Place 12 hexagons in a hexagonal arrangement
    # Center at origin, then 6 around, then 5 more?
    # Let's try a 2x6 grid rotated
    for angle_offset in [0.0, 30.0, 60.0, 90.0]:
        for spacing in [1.8, 2.0, 2.2, 2.5]:
            # Arrange in a hexagonal pattern
            positions = []
            # 7 hexagons in a hex pattern (1 center + 6 around)
            positions.append((0.0, 0.0))
            for k in range(6):
                ang = np.deg2rad(60.0 * k + angle_offset)
                positions.append((spacing * np.cos(ang), spacing * np.sin(ang)))
            # Add 5 more around the perimeter
            for k in range(5):
                ang = np.deg2rad(60.0 * k + 30.0 + angle_offset)
                r = spacing * np.sqrt(3.0)
                positions.append((r * np.cos(ang), r * np.sin(ang)))
            
            params = np.zeros(3 * N_inner)
            for i, (px, py) in enumerate(positions[:N_inner]):
                params[3*i] = px
                params[3*i+1] = py
                params[3*i+2] = angle_offset  # Inner hex angle
            candidates.append(params)
    
    # Configuration 2: Random
    for _ in range(20):
        params = np.zeros(3 * N_inner)
        for i in range(N_inner):
            params[3*i] = rng.uniform(-5, 5)
            params[3*i+1] = rng.uniform(-5, 5)
            params[3*i+2] = rng.uniform(0, 60)
        candidates.append(params)
    
    # Configuration 3: Circular arrangement
    for radius in [2.0, 2.5, 3.0]:
        for angle_offset in [0.0, 30.0]:
            params = np.zeros(3 * N_inner)
            for i in range(N_inner):
                ang = np.deg2rad(360.0 * i / N_inner + angle_offset)
                params[3*i] = radius * np.cos(ang)
                params[3*i+1] = radius * np.sin(ang)
                params[3*i+2] = angle_offset
            candidates.append(params)
    
    # Evaluate each candidate and polish with L-BFGS
    for idx, params in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Evaluate raw objective
        obj = _objective(params, N_inner, oangle_deg, beta=50.0)
        if obj < best_score:
            best_score = obj
            best_params = params.copy()
        
        # Polish with L-BFGS
        try:
            result = minimize(_objective, params, args=(N_inner, oangle_deg, 50.0),
                              method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            if result.success or result.fun < best_score:
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
        except Exception:
            pass
        
        # Also try with different beta for smoothing
        if time.time() > deadline:
            break
        for beta in [20.0, 100.0]:
            try:
                result = minimize(_objective, best_params, args=(N_inner, oangle_deg, beta),
                                  method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
    
    # Final: compute the true (non-smooth) outer side length for the best params
    if best_params is not None:
        ocx, ocy = 0.0, 0.0
        all_vertices = []
        inner = []
        for i in range(N_inner):
            x, y, ang = best_params[3*i:3*i+3]
            verts = _hex_vertices(x, y, 1.0, ang)
            all_vertices.append(verts)
            inner.append([float(x), float(y), float(ang)])
        all_vertices = np.vstack(all_vertices)
        true_side = _outer_containment(all_vertices, ocx, ocy, 0.0, oangle_deg)
    else:
        inner = [[0.0, 0.0, 0.0] for _ in range(N_inner)]
        true_side = 40.0
    
    return inner, (0.0, 0.0), float(true_side), 0.0

# EVOLVE-BLOCK-END
