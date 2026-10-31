import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
# We use barycentric coordinates (u, v) where w = 1 - u - v
# Point P = u*A + v*B + w*C
# A = (0,0)
# B = (1,0)
# C = (0.5, sqrt(3)/2)
# P_x = u*0 + v*1 + w*0.5 = v + 0.5*(1-u-v) = 0.5 - 0.5*u + 0.5*v
# P_y = u*0 + v*0 + w*sqrt(3)/2 = sqrt(3)/2 * (1-u-v)

def bary_uv_to_cart(uv_flat):
    """
    Convert flat array of (u, v) for N points to Cartesian coordinates.
    uv_flat shape: (2*N,)
    Returns shape: (N, 2)
    """
    uv = uv_flat.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Safety clip to ensure valid barycentric coords (though optimizer should handle bounds)
    # u, v, w >= 0
    # x = 0.5 - 0.5*u + 0.5*v
    # y = sqrt(3)/2 * w
    x = 0.5 - 0.5 * u + 0.5 * v
    y = (np.sqrt(3) / 2.0) * w
    
    return np.column_stack([x, y])

def min_area_from_cart(cart):
    """
    Compute minimum triangle area from Cartesian coordinates.
    cart shape: (N, 2)
    """
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def objective_hard(uv_flat):
    """
    Objective function for minimization (negative of min area).
    """
    cart = bary_uv_to_cart(uv_flat)
    return -min_area_from_cart(cart)

def soft_min_area(uv_flat, beta=50.0):
    """
    Smooth surrogate for min area.
    """
    cart = bary_uv_to_cart(uv_flat)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp trick for numerical stability
    # We want min(areas).
    # -1/beta * log(sum(exp(-beta * areas))) approximates min(areas)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def objective_soft(uv_flat, beta=50.0):
    """
    Objective function for minimization using soft min.
    """
    return -soft_min_area(uv_flat, beta)

def generate_hex_init():
    """
    Generate a hexagonal lattice initialization.
    """
    # Hexagonal packing in a triangle.
    # We can place points on a triangular grid.
    # For N=11, a 4x4 grid has 16 points, we need 11.
    # Let's use a specific pattern.
    # Rows in barycentric (u, v):
    # Row k (distance from A): points (i/k, (k-i)/k) for i=0..k
    # Total points in triangle of side L is (L+1)(L+2)/2.
    # L=3 -> 10 points. L=4 -> 15 points.
    # We need 11. Take the 10 points from L=3 and add one?
    # Or just take the first 11 points from a fine grid.
    
    # Let's generate a grid of side 4 (15 points) and select 11.
    # Or better, use a known good configuration for 11 points?
    # Let's stick to a grid and let the optimizer move them.
    
    points = []
    # Grid side 3 gives 10 points.
    # (0,0), (1/3, 0), (2/3, 0), (1, 0)
    # (0, 1/3), (1/3, 1/3), (2/3, 1/3)
    # (0, 2/3), (1/3, 2/3)
    # (0, 1)
    # This is 10 points.
    # Add the centroid (1/3, 1/3)? No, that's already there.
    # Add a point near the center?
    
    # Let's just use the 10 points and add one random point near the center.
    for k in range(4): # k=0,1,2,3
        for i in range(k + 1):
            u = i / 3.0
            v = (k - i) / 3.0
            points.append([u, v])
            
    # We have 10 points. Add 1 more.
    # The "hole" in the hex packing is at the center of the hexagons.
    # For a 4x4 triangular grid, the center is roughly (1/2, 1/2)?
    # Let's add (0.5, 0.5) -> w=0.
    points.append([0.5, 0.5])
    
    uv = np.array(points)
    return uv.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Bounds for u and v: [0, 1]
    # Constraint: u + v <= 1 is handled by the geometry if we clip, 
    # but DE needs explicit bounds. 
    # If u,v in [0,1], w=1-u-v can be negative.
    # We must enforce u>=0, v>=0, u+v<=1.
    # Differential Evolution doesn't support inequality constraints directly.
    # We can use a penalty function or map from a hypercube to the triangle.
    
    # Mapping: 
    # Let r1, r2 be uniform in [0,1].
    # u = sqrt(r1) * r2
    # v = sqrt(r1) * (1 - r2)
    # w = 1 - sqrt(r1)
    # This maps the unit square to the triangle uniformly.
    # However, for optimization, it's better to work in the constrained space if the optimizer supports it.
    # SLSQP supports constraints. DE does not.
    
    # Strategy:
    # 1. Use DE in the (r1, r2) space? No, the objective is non-smooth in that space too.
    # 2. Use DE with a penalty for invalid points.
    # 3. Use SLSQP with constraints.
    
    # Let's use SLSQP primarily, with multiple restarts.
    # And use DE if time permits, with a penalty.
    
    def penalty(uv_flat):
        uv = uv_flat.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1 - u - v
        # Penalty for u<0, v<0, w<0
        pen = 0
        pen += np.sum(np.minimum(u, 0)**2)
        pen += np.sum(np.minimum(v, 0)**2)
        pen += np.sum(np.minimum(w, 0)**2)
        return pen * 100.0

    def objective_de(uv_flat):
        val = objective_hard(uv_flat)
        pen = penalty(uv_flat)
        return val + pen

    # Bounds for DE: u, v in [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    best_uv = None
    best_val = -np.inf
    
    # Phase 1: Differential Evolution (Global Search)
    # DE is slow, so limit population size and max iterations.
    if time.time() < deadline - 60:
        try:
            # Use a smaller population to fit in time
            de_result = differential_evolution(
                objective_de,
                bounds=bounds,
                seed=42,
                maxiter=50, # Keep it low
                popsize=10,  # Keep it low
                tol=1e-5,
                polish=False, # We will polish manually
                init='sobol',
                updating='deferred'
            )
            if de_result.success or True: # DE often returns 'not success' but good point
                uv_de = de_result.x
                # Clean up invalid points by projecting to nearest valid point
                # Simple projection: clip u,v to [0,1] and if u+v>1, scale down
                uv_de = uv_de.reshape(N, 2)
                u = uv_de[:, 0]
                v = uv_de[:, 1]
                w = 1 - u - v
                invalid = (u < 0) | (v < 0) | (w < 0)
                if np.any(invalid):
                    # Project to barycentric simplex
                    # This is a simple heuristic, SLSQP will fix it properly
                    uv_de[invalid, 0] = np.clip(u[invalid], 0, 1)
                    uv_de[invalid, 1] = np.clip(v[invalid], 0, 1)
                    # Ensure sum <= 1
                    sums = uv_de.sum(axis=1)
                    over = sums > 1
                    if np.any(over):
                        uv_de[over] = uv_de[over] / sums[over][:, None]
                
                val_de = -objective_hard(uv_de.flatten())
                if val_de > best_val:
                    best_val = val_de
                    best_uv = uv_de.flatten().copy()
        except Exception as e:
            pass

    # Phase 2: Local Polish with SLSQP
    # Start from best_uv or hex_init if DE failed
    if best_uv is None:
        best_uv = generate_hex_init()
        best_val = -objective_hard(best_uv)
        
    # Define constraints for SLSQP
    # u >= 0, v >= 0, u + v <= 1 for each point
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]}) # u >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]}) # v >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*i+1]}) # u+v <= 1
        
    # Run SLSQP from multiple starting points
    starts = [best_uv]
    # Add hex init
    starts.append(generate_hex_init())
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random valid points
        r1 = rng.random(N)
        r2 = rng.random(N)
        u = np.sqrt(r1) * r2
        v = np.sqrt(r1) * (1 - r2)
        starts.append(np.column_stack([u, v]).flatten())
        
    for s in starts:
        if time.time() > deadline - 10:
            break
        try:
            # Use soft objective for smoother gradients initially, or hard?
            # SLSQP can handle non-smooth, but soft is faster.
            # Let's try hard first, it's the true objective.
            res = minimize(
                objective_hard,
                s,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_uv = res.x.copy()
        except Exception:
            pass

    # Phase 3: High-precision Polish
    # Use a very high beta soft min to refine
    if time.time() < deadline - 5 and best_uv is not None:
        try:
            res = minimize(
                lambda x: objective_soft(x, beta=200.0),
                best_uv,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            # Check if the soft solution is better in hard metric
            val_soft = -res.fun
            val_hard = -objective_hard(res.x)
            if val_hard > best_val:
                best_val = val_hard
                best_uv = res.x.copy()
        except Exception:
            pass
            
    # Final check with Nelder-Mead on hard objective (no constraints needed if we clip?)
    # NM doesn't handle constraints well. Stick to SLSQP.
    
    if best_uv is None:
        # Fallback
        best_uv = generate_hex_init()
        
    # Convert to Cartesian
    cart = bary_uv_to_cart(best_uv)
    return cart
