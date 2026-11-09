"""Program using Differential Evolution for hexagon packing."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    SQRT3 = math.sqrt(3)
    R = 1.0  # Circumradius
    INR = SQRT3 / 2.0 # Inradius

    # Precompute normal vectors for the 6 faces of a unit hexagon
    # Normal angles are 30, 90, 150, 210, 270, 330 degrees
    hex_norms = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))]
        for i in range(6)
    ])
    
    # Precompute vertex offsets relative to center (unit hexagon)
    hex_verts = np.array([
        [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))]
        for i in range(6)
    ])

    def objective(params):
        # params: 3*N array [x1, y1, a1, x2, y2, a2, ...]
        # a in radians
        penalty = 0.0
        
        # 1. Compute outer side length (cost to minimize)
        # Outer hexagon is centered at (0,0) with rotation 0 (we fix this, or optimize later)
        # Actually, let's optimize the outer rotation too? 
        # The task allows outer rotation. 
        # However, the objective function is easier if we just compute the bounding box.
        # Let's stick to outer rotation = 0 for the main search to keep dimensionality manageable,
        # then refine.
        
        # Calculate max projection of all inner hexagon vertices onto the 6 outer normals
        max_proj = 0.0
        
        # Unpack parameters
        centers = params[0:3*N:3]
        ys = params[1:3*N:3]
        angles = params[2:3*N:3]
        
        # For each inner hexagon
        for i in range(N):
            cx, cy, ang = centers[i], ys[i], angles[i]
            
            # Rotate vertices
            c_a = math.cos(ang)
            s_a = math.sin(ang)
            
            # Vertices of this hexagon
            # v_rot = v * R * [cos(a), -sin(a); sin(a), cos(a)]
            # Since hex_verts are unit, multiply by R=1
            vx = hex_verts[:, 0] * c_a - hex_verts[:, 1] * s_a + cx
            vy = hex_verts[:, 0] * s_a + hex_verts[:, 1] * c_a + cy
            
            # Project onto 6 outer normals
            # outer_norms are fixed at 30, 90...
            # proj = v . n
            # We want the max absolute projection? No, the hexagon is centered at 0, so we look at max(v.n) and min(v.n)
            # The side length is related to max(v.n) - min(v.n)
            # Specifically, side = (max_proj - min_proj) / sqrt(3) ?
            # For a regular hexagon of side S, the distance between parallel sides is S*sqrt(3).
            # So S = (max_proj - min_proj) / sqrt(3).
            
            projs = vx * hex_norms[:, 0] + vy * hex_norms[:, 1]
            current_max = np.max(projs)
            current_min = np.min(projs)
            
            if current_max > max_proj:
                max_proj = current_max
            # We need global min too
            # Let's track global min separately
            
            # Optimization: We can just compute all vertices at once for speed?
            # For now, loop is fine for N=11.

        # We need global min as well
        # Let's re-calculate global min in the same loop or separately
        # To avoid double loop, let's just do it in one pass
        
        # Re-do logic to track global min
        global_max = 0.0
        global_min = 0.0
        first = True
        
        for i in range(N):
            cx, cy, ang = centers[i], ys[i], angles[i]
            c_a = math.cos(ang)
            s_a = math.sin(ang)
            
            vx = hex_verts[:, 0] * c_a - hex_verts[:, 1] * s_a + cx
            vy = hex_verts[:, 0] * s_a + hex_verts[:, 1] * c_a + cy
            
            projs = vx * hex_norms[:, 0] + vy * hex_norms[:, 1]
            
            if first:
                global_max = np.max(projs)
                global_min = np.min(projs)
                first = False
            else:
                global_max = max(global_max, np.max(projs))
                global_min = min(global_min, np.min(projs))
        
        outer_side = (global_max - global_min) / SQRT3
        
        # 2. Overlap Penalty
        # For each pair (i, j), check if they overlap.
        # Use Separating Axis Theorem (SAT) logic.
        # If for any normal direction (from i or j), the projections are disjoint, they are disjoint.
        # If they overlap in all 12 directions, they intersect.
        # To make it smooth for DE, we use a penalty based on the "depth" of overlap in the *most overlapping* direction?
        # Or simpler: sum of squared overlaps?
        
        # A robust smooth penalty:
        # For each pair, find the minimum separation distance.
        # If separation < 0, penalty += separation^2.
        
        for i in range(N):
            cx_i, cy_i, ang_i = centers[i], ys[i], angles[i]
            for j in range(i + 1, N):
                cx_j, cy_j, ang_j = centers[j], ys[j], angles[j]
                
                # Normals for i and j
                c_i = math.cos(ang_i); s_i = math.sin(ang_i)
                c_j = math.cos(ang_j); s_j = math.sin(ang_j)
                
                # Normals of i rotated
                ni = hex_norms[:, 0] * c_i - hex_norms[:, 1] * s_i
                niy = hex_norms[:, 0] * s_i + hex_norms[:, 1] * c_i
                
                # Normals of j rotated
                nj = hex_norms[:, 0] * c_j - hex_norms[:, 1] * s_j
                njy = hex_norms[:, 0] * s_j + hex_norms[:, 1] * c_j
                
                # All 12 candidate normals
                all_nx = np.concatenate([ni, nj])
                all_ny = np.concatenate([niy, njy])
                
                # Project vertices of i and j onto these normals
                # Vertices of i
                vxi = hex_verts[:, 0] * c_i - hex_verts[:, 1] * s_i + cx_i
                vyi = hex_verts[:, 0] * s_i + hex_verts[:, 1] * c_i + cy_i
                
                # Vertices of j
                vxj = hex_verts[:, 0] * c_j - hex_verts[:, 1] * s_j + cx_j
                vyj = hex_verts[:, 0] * s_j + hex_verts[:, 1] * c_j + cy_j
                
                # Projections
                # pi = vxi * nx + vyi * ny
                # pj = vxj * nx + vyj * ny
                
                # Vectorized projection
                # pi is (6, 12) -> (12, 6)? No.
                # We want for each of 12 normals, the range of i and range of j.
                
                # proj_i[k, m] = v_i[m] . n[k]
                # v_i is 6x2, n is 12x2
                # proj_i = v_i @ n.T  -> 6x12
                # We need min/max over axis 0 (vertices) for each normal k.
                
                proj_i = np.dot(hex_verts, all_nx) + cx_i * all_nx + np.dot(hex_verts[:, 1], all_ny) + cy_i * all_ny
                # Wait, hex_verts is 6x2. all_nx is 12x1? No, all_nx is 12x1 array?
                # Let's use matrix multiplication properly.
                # v_i (6x2) @ n_k (2x1) -> 6x1
                # Let's do:
                # proj_i = (vxi * all_nx[None, :]) + (vyi * all_ny[None, :])
                # vxi is 6x1, all_nx is 12x1 -> broadcast to 6x12
                proj_i = vxi[:, None] * all_nx[None, :] + vyi[:, None] * all_ny[None, :]
                min_i = np.min(proj_i, axis=0) # 12x1
                max_i = np.max(proj_i, axis=0)
                
                proj_j = vxj[:, None] * all_nx[None, :] + vyj[:, None] * all_ny[None, :]
                min_j = np.min(proj_j, axis=0)
                max_j = np.max(proj_j, axis=0)
                
                # Overlap in direction k:
                # overlap = min(max_i, max_j) - max(min_i, min_j)
                # If overlap > 0, they intersect in this direction.
                # If overlap <= 0 for ANY k, they are disjoint.
                
                overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                
                # If max(overlaps) > 0, they overlap.
                # The "depth" of overlap is max(overlaps).
                # Penalty: square of the positive part.
                max_overlap = np.max(overlaps)
                if max_overlap > 0:
                    penalty += max_overlap * max_overlap
        
        return outer_side + 10.0 * penalty

    # Bounds for DE
    # x, y in [-5, 5]
    # angle in [0, 2*pi]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 2.0 * math.pi))
    
    # Initial population
    # Use a mix of random and structured starts?
    # DE generates its own population.
    
    # Run DE
    # maxiter: 1000? 
    # tol: 1e-4
    # popsize: 20
    
    best_result = None
    best_val = float('inf')
    
    # We might need to run multiple DE instances if time permits, or just one big one.
    # One big one is usually better for DE.
    
    try:
        # Use differential_evolution
        # We need to be careful with time.
        # Let's estimate time per function evaluation.
        # N=11, pairs=55. 12 normals per pair.
        # ~55 * 12 * 6 ops ~ 4000 ops. Very fast.
        # 1000 iters * 20 popsize * 11 (generations) ? 
        # DE complexity is O(popsize * N_vars * maxiter).
        # 20 * 33 * 1000 = 660,000 evaluations.
        # If each eval is 1ms, that's 660s. Too slow.
        # If each eval is 0.1ms, that's 66s. Feasible.
        
        # Let's try maxiter=200, popsize=15.
        # 15 * 33 * 200 = 99,000 evals.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            polish=False, # We'll do our own polish
            init='sobol' # Better initial distribution
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    if best_result is None:
        # Fallback
        return ([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                 [-1.0, 1.5, 0.0], [1.0, 1.5, 0.0], [0.0, 3.0, 0.0],
                 [-1.0, -1.5, 0.0], [1.0, -1.5, 0.0], [0.0, -3.0, 0.0],
                 [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], (0.0, 0.0), 5.0, 0.0)
        
    # Post-processing: Refine with Nelder-Mead or simple gradient descent?
    # DE finds a good basin. Let's do a local refinement using Nelder-Mead on the same objective.
    
    from scipy.optimize import minimize
    
    try:
        # Refine
        res_refined = minimize(
            objective, 
            best_result, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if res_refined.fun < best_val:
            best_val = res_refined.fun
            best_result = res_refined.x
    except Exception as e:
        print(f"Refinement failed: {e}")
        
    # Extract best parameters
    centers = best_result[0:3*N:3]
    ys = best_result[1:3*N:3]
    angles_rad = best_result[2:3*N:3]
    
    # Calculate actual outer side and rotation
    # The objective assumed outer rotation 0.
    # We should sweep outer rotation to find the true minimum bounding hexagon.
    
    # Reconstruct vertices
    all_verts = []
    for i in range(N):
        cx, cy, ang = centers[i], ys[i], angles_rad[i]
        c_a = math.cos(ang)
        s_a = math.sin(ang)
        vx = hex_verts[:, 0] * c_a - hex_verts[:, 1] * s_a + cx
        vy = hex_verts[:, 0] * s_a + hex_verts[:, 1] * c_a + cy
        all_verts.extend(zip(vx, vy))
        
    # Sweep outer rotation
    best_side = float('inf')
    best_oa = 0.0
    
    # The outer hexagon normals are rotated by oa.
    # side(oa) = max_{v in V, n in N(oa)} (v.n) - min_{v in V, n in N(oa)} (v.n) ) / sqrt(3)
    # Wait, the bounding hexagon is the smallest regular hexagon containing all points.
    # For a fixed orientation of the outer hexagon, the side length is determined by the max projection.
    
    for oa_deg in range(0, 60, 1):
        oa = math.radians(oa_deg)
        # Outer normals
        onx = np.array([math.cos(oa + math.radians(30 + i * 60)) for i in range(6)])
        ony = np.array([math.sin(oa + math.radians(30 + i * 60)) for i in range(6)])
        
        max_p = -float('inf')
        min_p = float('inf')
        
        for vx, vy in all_verts:
            for k in range(6):
                p = vx * onx[k] + vy * ony[k]
                if p > max_p: max_p = p
                if p < min_p: min_p = p
                
        side = (max_p - min_p) / SQRT3
        if side < best_side:
            best_side = side
            best_oa = oa_deg
            
    # Fine sweep
    for oa_deg in np.arange(best_oa - 1, best_oa + 1, 0.1):
        oa = math.radians(oa_deg)
        onx = np.array([math.cos(oa + math.radians(30 + i * 60)) for i in range(6)])
        ony = np.array([math.sin(oa + math.radians(30 + i * 60)) for i in range(6)])
        
        max_p = -float('inf')
        min_p = float('inf')
        
        for vx, vy in all_verts:
            for k in range(6):
                p = vx * onx[k] + vy * ony[k]
                if p > max_p: max_p = p
                if p < min_p: min_p = p
                
        side = (max_p - min_p) / SQRT3
        if side < best_side:
            best_side = side
            best_oa = oa_deg
            
    # Center the configuration?
    # The problem doesn't strictly require the inner hexagons to be centered,
    # but the outer one is at (0,0).
    # Our optimization kept them roughly centered due to the bounding box symmetry,
    # but let's explicitly center the centroid of the inner hexagons to (0,0)
    # to ensure the bounding box is minimal?
    # Actually, shifting the inner hexagons doesn't change their relative positions,
    # but it shifts the bounding box.
    # The bounding box is centered at (0,0) in our calculation?
    # No, the bounding box is the hexagon centered at (0,0).
    # If the inner hexagons are shifted, the bounding box size increases.
    # So the optimizer should have centered them.
    
    # Construct output
    inner_list = []
    for i in range(N):
        inner_list.append([centers[i], ys[i], math.degrees(angles_rad[i]) % 360.0])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
