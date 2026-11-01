import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of unit hexagon centered at 0
    # Side 1. Vertices at distance 1 from center.
    # Angles: 0, 60, 120, 180, 240, 300
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    offsets = base_angles + angle_rad
    xs = np.cos(offsets)
    ys = np.sin(offsets)
    return np.column_stack((xs + cx, ys + cy))

def point_in_hex(v, L, outer_angle_deg):
    # Check if point v is inside hex of side L, rotated by outer_angle_deg
    # Hex normals are at angles: outer_angle, outer_angle+60, outer_angle+120
    # Inradius r = sqrt(3)/2 * L
    # Condition: |v . n_k| <= r for k=0,1,2
    outer_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_rad), np.sin(outer_rad)],
        [np.cos(outer_rad + np.pi/3), np.sin(outer_rad + np.pi/3)],
        [np.cos(outer_rad + 2*np.pi/3), np.sin(outer_rad + 2*np.pi/3)]
    ])
    r = np.sqrt(3) / 2 * L
    # v is (x,y)
    dots = np.abs(np.dot(normals, v))
    return np.all(dots <= r)

def hexes_overlap(v1, v2):
    # SAT check for two convex polygons
    # v1, v2 are (6,2) arrays
    def get_axes(v):
        axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            # Normal
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
            
        # Calculate overlap depth for penalty
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > 0:
            return overlap
            
    return 0.0

def objective(params):
    # params: x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, outer_angle
    # 11 * 3 + 1 = 34 vars
    inner_params = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    # Calculate vertices for all hexes
    verts = []
    for i in range(11):
        x, y, a = inner_params[i]
        verts.append(hex_vertices(x, y, a))
        
    # Calculate minimal L
    # L = max over all vertices v of (2/sqrt(3)) * max_k |v . n_k|
    outer_rad = np.deg2rad(outer_angle)
    normals = np.array([
        [np.cos(outer_rad), np.sin(outer_rad)],
        [np.cos(outer_rad + np.pi/3), np.sin(outer_rad + np.pi/3)],
        [np.cos(outer_rad + 2*np.pi/3), np.sin(outer_rad + 2*np.pi/3)]
    ])
    
    max_proj = 0.0
    for v in verts:
        dots = np.abs(np.dot(normals, v.T)) # (3, 6)
        max_proj = max(max_proj, np.max(dots))
        
    L_min = (2.0 / np.sqrt(3)) * max_proj
    
    # Check overlaps
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            ov = hexes_overlap(verts[i], verts[j])
            if ov > 0:
                penalty += ov * 100.0 # Heavy penalty
                
    return L_min + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit
    
    # Initial guess
    # Lattice
    u = np.array([2.0, 0.0])
    v = np.array([1.0, np.sqrt(3)])
    
    centers = []
    # Center
    centers.append([0.0, 0.0, 0.0])
    # Ring 1
    ring1_offsets = [u, v, -u+v, -u, -v, u-v]
    for off in ring1_offsets:
        centers.append([off[0], off[1], 0.0])
        
    # Ring 2 (pick 4)
    # Candidates: 2u, u+2v, 2v, -u+2v, -2u, ...
    # Let's pick 2u, -2u, u+2v, -u-2v? No, symmetry.
    # Pick (4,0), (-4,0), and two others.
    # (3, sqrt(3)) and (-3, -sqrt(3))?
    # Let's just add 4 more from the next ring arbitrarily.
    ring2_candidates = [
        2*u, u+2*v, 2*v, -u+2*v, -2*u, -u-2*v, -2*v, u-2*v, 2*u-v, -2*u+v, -u-2*v, -2*u-v
    ]
    # Note: some duplicates in manual list, let's generate properly.
    # Actually, just adding 4 specific points.
    centers.append([4.0, 0.0, 0.0])
    centers.append([-4.0, 0.0, 0.0])
    centers.append([2.0, 2*np.sqrt(3), 0.0])
    centers.append([-2.0, -2*np.sqrt(3), 0.0])
    
    init_params = np.array(centers).flatten()
    init_params = np.append(init_params, 0.0) # outer angle
    
    best_params = init_params
    best_score = objective(best_params)
    
    # Optimization
    # Powell is good for this
    try:
        res = minimize(objective, init_params, method='Powell', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        # Check time
        if time.time() > deadline:
            pass # Use best so far? minimize doesn't easily yield intermediate bests in this API without callback.
            # We'll rely on the result or the init if it timed out.
            # Actually, if it times out, it might raise or return partial.
            # Better to run in a loop with smaller maxiter and check time.
            
        # Let's do a manual loop to be safe with time.
    except:
        pass

    # Manual loop for time safety
    current_params = init_params
    current_score = objective(current_params)
    
    # Simple local search / Powell steps
    # Since Powell is not easily interruptible in a simple loop without re-implementing,
    # we can use Nelder-Mead which is slower but we can check time?
    # Or just run minimize with a timeout wrapper?
    
    # Let's try a few restarts with different inits if time permits.
    
    # For now, just one run.
    # To ensure we don't exceed time, we can limit iterations.
    
    # Re-run minimize but we can't easily kill it mid-flight in standard scipy.
    # However, N=11 is small. Powell should be fast.
    
    res = minimize(objective, init_params, method='Powell', 
                   options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3})
    
    if res.success or res.fun < best_score:
        best_params = res.x
        best_score = res.fun
        
    # Check time
    if time.time() < deadline - 10:
        # Try a second run with different initial angle?
        init_params2 = init_params.copy()
        init_params2[33] = 30.0
        res2 = minimize(objective, init_params2, method='Powell', 
                   options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
        if res2.fun < best_score:
            best_params = res2.x
            best_score = res2.fun

    # Extract results
    inner_params = best_params[:33].reshape(11, 3)
    outer_angle = best_params[33]
    
    inner = []
    for i in range(11):
        x, y, a = inner_params[i]
        inner.append([x, y, a])
        
    # Calculate final L
    # We need the actual L, not the penalized score.
    # Re-calculate L_min from best_params
    verts = []
    for i in range(11):
        x, y, a = inner_params[i]
        verts.append(hex_vertices(x, y, a))
        
    outer_rad = np.deg2rad(outer_angle)
    normals = np.array([
        [np.cos(outer_rad), np.sin(outer_rad)],
        [np.cos(outer_rad + np.pi/3), np.sin(outer_rad + np.pi/3)],
        [np.cos(outer_rad + 2*np.pi/3), np.sin(outer_rad + 2*np.pi/3)]
    ])
    
    max_proj = 0.0
    for v in verts:
        dots = np.abs(np.dot(normals, v.T))
        max_proj = max(max_proj, np.max(dots))
        
    L_final = (2.0 / np.sqrt(3)) * max_proj
    
    return inner, (0.0, 0.0), L_final, outer_angle
