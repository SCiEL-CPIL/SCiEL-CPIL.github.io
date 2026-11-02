"""Packing 12 unit hexagons in a regular hexagon - vectorized SAT, multi-param SA, epigraph polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 144.0

    def hex_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        thetas = angle + np.arange(6) * (np.pi / 3)
        return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

    def get_all_verts(params):
        """Get vertices for all N hexagons from params vector."""
        verts = np.empty((N, 6, 2))
        for i in range(N):
            verts[i] = hex_vertices(params[3*i], params[3*i+1], 1.0, params[3*i+2])
        return verts

    def pair_disjoint(verts):
        """Vectorized SAT check. Returns boolean array of shape (N, N) where True means disjoint."""
        # Use all 12 edge normals from each hex (6 from hex i, 6 from hex j)
        # Edge normals for a hexagon with angle a: directions a+30, a+90, a+150, a+210, a+270, a+330 degrees
        # Actually for a regular hexagon, the 6 outward face normals are at angles a+30+k*60 for k=0..5
        # But we need edge normals. Let me think...
        # A regular hexagon with vertex at angle a has edges between consecutive vertices.
        # Edge i goes from vertex i to vertex i+1. The outward normal of this edge is at angle a + (i+0.5)*60 + 90 = a + (i+1.5)*60
        # Wait, let me be more careful.
        # Vertex i is at angle a + i*60. Edge from vertex i to vertex i+1.
        # Midpoint of edge is at angle a + (i+0.5)*60, at distance (side/2)*sqrt(3) from center (apothem).
        # The outward normal direction is a + (i+0.5)*60.
        # Actually the normal to the edge from v_i to v_{i+1} points in the direction of the midpoint from center.
        # The midpoint angle is a + (i+0.5)*pi/3.
        
        # For SAT between two convex polygons, we check all edge normals from both polygons.
        # For two hexagons, that's 12 normals total.
        # But we can be smarter: for each hex, precompute its 6 face normals.
        
        # Let me use a different approach: for each pair, check separation along 6 directions
        # derived from the relative orientation.
        
        # Actually, the simplest correct approach: for each pair (i,j), check all 6 normals from hex i
        # and all 6 normals from hex j. If any normal separates them, they're disjoint.
        
        # Let me precompute normals for each hexagon.
        normals = np.empty((N, 6, 2))
        for i in range(N):
            angle = np.radians(params[3*i+2])
            for k in range(6):
                na = angle + (k + 0.5) * np.pi / 3
                normals[i, k, 0] = np.cos(na)
                normals[i, k, 1] = np.sin(na)
        
        disjoint = np.ones((N, N), dtype=bool)
        
        for i in range(N):
            for j in range(i+1, N):
                vi = verts[i]  # (6,2)
                vj = verts[j]  # (6,2)
                
                separated = False
                # Check normals from hex i
                for k in range(6):
                    n = normals[i, k]
                    proj_i = vi @ n
                    proj_j = vj @ n
                    if proj_i.max() < proj_j.min() or proj_j.max() < proj_i.min():
                        separated = True
                        break
                if not separated:
                    # Check normals from hex j
                    for k in range(6):
                        n = normals[j, k]
                        proj_i = vi @ n
                        proj_j = vj @ n
                        if proj_i.max() < proj_j.min() or proj_j.max() < proj_i.min():
                            separated = True
                            break
                
                if separated:
                    disjoint[i, j] = True
                    disjoint[j, i] = True
                else:
                    disjoint[i, j] = False
                    disjoint[j, i] = False
        
        return disjoint

    def is_valid(params):
        """Check if all hexagons are pairwise disjoint."""
        verts = get_all_verts(params)
        # Quick distance check first
        centers = params[:3*N].reshape(N, 3)[:, :2]
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                if dx*dx + dy*dy > 16.0:  # distance > 4, definitely disjoint
                    continue
                # Need SAT check
                vi = verts[i]
                vj = verts[j]
                angle_i = np.radians(params[3*i+2])
                angle_j = np.radians(params[3*j+2])
                
                separated = False
                # Normals from hex i
                for k in range(6):
                    na = angle_i + (k + 0.5) * np.pi / 3
                    nx = np.cos(na)
                    ny = np.sin(na)
                    proj_i = vi[:,0]*nx + vi[:,1]*ny
                    proj_j = vj[:,0]*nx + vj[:,1]*ny
                    if proj_i.max() < proj_j.min() or proj_j.max() < proj_i.min():
                        separated = True
                        break
                if not separated:
                    for k in range(6):
                        na = angle_j + (k + 0.5) * np.pi / 3
                        nx = np.cos(na)
                        ny = np.sin(na)
                        proj_i = vi[:,0]*nx + vi[:,1]*ny
                        proj_j = vj[:,0]*nx + vj[:,1]*ny
                        if proj_i.max() < proj_j.min() or proj_j.max() < proj_i.min():
                            separated = True
                            break
                if not separated:
                    return False
        return True

    def compute_side(params):
        """Compute minimum side length of enclosing regular hexagon."""
        inner_verts = get_all_verts(params)
        center = np.array([params[3*N], params[3*N+1]])
        angle = params[3*N+3]
        
        all_verts = inner_verts.reshape(-1, 2)
        offsets = all_verts - center
        
        max_extent = 0.0
        for k in range(6):
            na = np.radians(angle) + np.radians(30) + k * np.pi / 3
            nx = np.cos(na)
            ny = np.sin(na)
            extents = offsets[:,0]*nx + offsets[:,1]*ny
            max_e = extents.max()
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / SQRT3

    def hard_objective(params):
        """Return side length if valid, else large penalty."""
        if not is_valid(params):
            return 1e8
        return compute_side(params)

    def smooth_objective(params, beta=50.0):
        """Smooth surrogate: soft containment + separation penalty."""
        inner_verts = get_all_verts(params)
        center = np.array([params[3*N], params[3*N+1]])
        angle = params[3*N+3]
        
        all_verts = inner_verts.reshape(-1, 2)
        offsets = all_verts - center
        
        max_extent = 0.0
        for k in range(6):
            na = np.radians(angle) + np.radians(30) + k * np.pi / 3
            nx = np.cos(na)
            ny = np.sin(na)
            extents = offsets[:,0]*nx + offsets[:,1]*ny
            # Soft-max
            m = extents.max()
            if m > 1e6:
                max_e = m
            else:
                max_e = m + np.log(np.sum(np.exp(extents - m))) / beta
            if max_e > max_extent:
                max_extent = max_e
        
        side = max_extent * 2.0 / SQRT3
        
        # Separation penalty
        centers = params[:3*N].reshape(N, 3)[:, :2]
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist_sq = dx*dx + dy*dy
                # Minimum center distance for non-overlapping unit hexagons is sqrt(3) ≈ 1.732
                # But with rotation, it can be as low as... well, let's use 1.5 as soft threshold
                threshold = 2.2
                if dist_sq < threshold:
                    sep_penalty += (threshold - dist_sq) ** 2
        
        return side + 3.0 * sep_penalty

    def initial_honeycomb():
        """Compact honeycomb-like initial configuration."""
        params = np.zeros(3*N + 4)
        positions = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((SQRT3 * np.cos(theta), SQRT3 * np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/10
            positions.append((3.0*np.cos(theta), 3.0*np.sin(theta), 0))
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+3] = 0.0
        return params

    def initial_random():
        params = np.zeros(3*N + 4)
        for i in range(N):
            params[3*i] = random.uniform(-2.5, 2.5)
            params[3*i+1] = random.uniform(-2.5, 2.5)
            params[3*i+2] = random.uniform(0, 60)
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+3] = random.uniform(0, 60)
        return params

    # Multi-start SA with diverse initializations
    best_params = None
    best_score = float('inf')
    
    inits = [initial_honeycomb()]
    for _ in range(4):
        inits.append(initial_random())
    
    for p in inits:
        s = hard_objective(p)
        if s < best_score:
            best_score = s
            best_params = p.copy()
    
    # Main SA loop with multi-parameter perturbations
    current_params = best_params.copy()
    current_score = smooth_objective(current_params, 50.0)
    
    T = 1.5
    T_min = 0.0002
    cooling_rate = 0.9997
    max_steps = 80000
    
    step_count = 0
    last_improve_step = 0
    patience = 8000
    
    while time.time() < deadline - 20.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb 2-4 random parameters simultaneously
        n_perturb = random.choice([2, 3, 4])
        idxs = random.sample(range(3*N + 4), n_perturb)
        for idx in idxs:
            if idx == 3*N + 2:  # skip center_x (use symmetry)
                idx = random.randint(0, 3*N + 1)
            step_size = 0.25 * (0.2 + 0.8 * T / T_min)
            new_params[idx] += random.gauss(0, step_size)
        
        new_score = smooth_objective(new_params, 50.0)
        
        accept = new_score < current_score
        if not accept:
            delta = new_score - current_score
            if delta < 0:
                accept = True
            elif random.random() < np.exp(-delta / max(T, 1e-10)):
                accept = True
        
        if accept:
            current_params = new_params
            current_score = new_score
            
            if step_count % 50 == 0 or current_score < best_score - 0.005:
                h = hard_objective(current_params)
                if h < best_score:
                    best_score = h
                    best_params = current_params.copy()
                    last_improve_step = step_count
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        if step_count - last_improve_step > patience and T < 0.005:
            break
    
    # L-BFGS-B polish with epigraph-style objective
    if time.time() < deadline - 8.0 and best_score < 1e6:
        for attempt in range(4):
            if time.time() >= deadline - 8.0:
                break
            start_p = best_params.copy()
            if attempt > 0:
                start_p = start_p + np.random.normal(0, 0.03, start_p.shape)
            
            try:
                res = minimize(
                    smooth_objective,
                    start_p,
                    method='L-BFGS-B',
                    args=(50.0,),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                if res.fun < 1e6:
                    hs = hard_objective(res.x)
                    if hs < best_score:
                        best_score = hs
                        best_params = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision local search
    if time.time() < deadline - 3.0 and best_score < 1e6:
        for _ in range(80):
            if time.time() >= deadline - 3.0:
                break
            p = best_params.copy()
            # Perturb 1-2 params with very small steps
            n_p = random.choice([1, 2])
            idxs = random.sample(range(3*N + 4), n_p)
            for idx in idxs:
                p[idx] += random.gauss(0, 0.005)
            s = hard_objective(p)
            if s < best_score:
                best_score = s
                best_params = p.copy()
    
    # Extract result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+3])
    side = compute_side(best_params)
    
    # Validate
    valid = is_valid(best_params)
    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
