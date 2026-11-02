import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
CIRCUMRADIUS = 1.0 # Unit hexagon circumradius
# Minkowski sum of two unit hexagons (same orientation) is a hexagon with side 2, circumradius 2.
# If rotated, the distance required to avoid overlap depends on relative angle.
# For simplicity and differentiability, we can use a conservative bound or a smooth approximation.
# The minimal distance between centers of two non-overlapping unit hexagons with relative angle delta
# is related to the support function of the Minkowski sum.
# However, a simpler robust penalty is based on the fact that if distance < 2*CIRCUMRADIUS, they *might* overlap.
# Actually, for two regular hexagons of side 1, the "collision" distance varies with relative rotation.
# Max distance needed is 2 (when aligned). Min distance is sqrt(3) (when one is rotated 30 deg? No, that's not right).
# Let's use a soft penalty that grows as distance d < d_min(angle_diff).
# d_min for two unit hexagons with relative angle alpha:
# The Minkowski sum is a convex polygon. The distance from origin to boundary is the support function h(u).
# This is complex to compute analytically in a fast vectorized way for the gradient.
# Alternative: Use a simpler penalty: sum of max(0, 2 - dist)^2. This assumes worst-case overlap distance 2.
# This is conservative but ensures no overlap if penalty is strong enough and optimizer converges.
# To be more precise, we can precompute the minimum distance for a set of relative angles or use a lookup.
# Given time constraints, let's use a slightly relaxed constraint: d > 1.99 for aligned, but we want smoothness.
# Let's stick to d > 2.0 as a hard-ish constraint via penalty, or use the exact support function if possible.
# Actually, the Minkowski sum of two regular hexagons with relative rotation theta is a regular 12-gon if theta=pi/6, 
# but for general theta it's a 12-gon (or fewer if symmetric).
# The minimum distance to avoid overlap is the minimum support function value.
# Let's approximate the non-overlap constraint as: dist > 2 * cos(min(angle_diff, pi/6 - angle_diff)/2)? No.
# Let's use a very safe bound: dist > 2.0. This might prevent optimal packing if rotation allows closer packing.
# Wait, if I rotate one 30 degrees, the "width" decreases? No, the circumradius is constant.
# The apothem is sqrt(3)/2.
# If I align them, min dist is 2.
# If I rotate one by 30 deg, the vertices of one point to the edges of the other.
# The distance from center to vertex is 1. Distance from center to edge is sqrt(3)/2.
# The minimum distance to avoid overlap is determined by the Minkowski sum.
# For two unit hexagons, the Minkowski sum's circumradius is 2 (when aligned).
# When rotated by 30 deg, the Minkowski sum is a regular dodecagon? 
# Side length of Minkowski sum: if aligned, sides are 2. 
# If rotated by 30 deg, the vertices of the sum are sums of vertices.
# Let's just use d > 2.0 as the constraint. It's conservative. If the optimizer can't get them closer, it will stay at 2.0.
# But maybe we can do better. The minimum distance between two non-overlapping unit hexagons is actually slightly less than 2 if rotated?
# No, the circumradius is 1. The distance between centers must be at least the distance from origin to the boundary of the Minkowski sum in the direction of the center vector.
# The Minkowski sum of two unit hexagons is always contained in a circle of radius 2.
# So d >= 2 is a sufficient condition for non-overlap for ANY rotation.
# Is it necessary? No. But it's a valid constraint.
# To allow tighter packing, we should use the exact minimum distance for the given relative rotation.
# Let's implement a function that returns the minimum safe distance for a given relative angle.
# Actually, for the penalty, we can just use d > 2.0 and see if it works. If the score is bad, we refine.
# However, the task allows rotation. If we fix relative rotations to 0, we are back to lattice.
# The variables include individual angles.
# Let's define the penalty as sum over pairs of max(0, d_min - dist)^2.
# d_min for relative angle alpha:
# The support function of a regular hexagon with circumradius R=1 is h(u) = R * max_{v in vertices} u.dot(v).
# The Minkowski sum has support h_sum(u) = h_hex(u) + h_hex(R_alpha^T u).
# The minimum distance is min_u h_sum(u) / ||u||? No, the distance is the minimum of h_sum(u) for unit u.
# This is a 1D minimization for each pair, which is expensive inside the objective.
# Given 11 hexagons, there are 55 pairs. 
# Let's approximate d_min(alpha) with a lookup table or a simple formula.
# For small alpha, d_min is close to 2.
# For alpha = pi/6, the hexagons are "interlocked" in a sense.
# Let's use d_min = 2.0 * (1 - 0.05 * (1 - cos(6*alpha)))? Just a guess.
# Better: Use d_min = 2.0. It's safe. The optimizer will try to minimize the outer size. If it's constrained by d>=2, it will find the best packing with d>=2.
# This is a valid packing. It might not be the global optimum if tighter packing is possible with rotation, but it's a good start.
# Actually, I recall that for circles, it's 2R. For hexagons, the "effective radius" varies.
# Let's try to implement the exact d_min.
# h_hex(u) for unit hexagon: vertices at angles 0, 60, ... 300.
# h(u) = max(cos(theta_u - 0), cos(theta_u - 60), ...).
# This is a piecewise linear function in u.
# The minimum of h_sum(u) over unit circle u.
# This is a convex optimization problem itself? No, h is convex, sum is convex. Min of convex function is easy.
# But doing this 55 times per objective evaluation is too slow.
# Let's stick to d_min = 2.0 for now, or a slightly relaxed value if we are confident.
# Actually, if I use d_min = 2.0, I am essentially forcing the hexagons to be at least 2 units apart.
# This is the same as packing circles of radius 1.
# The optimal packing of 11 circles in a hexagon is known.
# But we are packing hexagons. Hexagons can pack tighter than circles if rotated?
# No, hexagons tile the plane. Circles do not.
# So hexagons can pack tighter.
# If I force d >= 2, I am preventing them from tiling.
# I must use the correct d_min.
# Let's precompute d_min for relative angles 0 to 30 degrees (due to symmetry).
# We can sample the angle 0 to pi/6 in small steps and find the min support function.

def compute_d_min_table(num_steps=100):
    angles = np.linspace(0, math.pi/6, num_steps)
    d_mins = np.zeros(num_steps)
    # Vertices of unit hexagon
    v_angles = np.arange(6) * math.pi / 3
    v_x = np.cos(v_angles)
    v_y = np.sin(v_angles)
    
    for i, alpha in enumerate(angles):
        # Minimize h_sum(u) = h_hex(u) + h_hex(R_alpha^T u) over unit circle u
        # u is (cos(theta), sin(theta))
        # h_hex(u) = max_k (u dot v_k)
        # This is a 1D problem in theta.
        # We can sample theta from 0 to 2*pi.
        thetas = np.linspace(0, 2*math.pi, 720)
        u_x = np.cos(thetas)
        u_y = np.sin(thetas)
        
        # h1
        dots1 = u_x[:, None] * v_x[None, :] + u_y[:, None] * v_y[None, :]
        h1 = np.max(dots1, axis=1)
        
        # h2: rotate u by -alpha, then dot with v
        # R_alpha^T u = (cos(alpha)u_x + sin(alpha)u_y, -sin(alpha)u_x + cos(alpha)u_y)
        c_a = math.cos(alpha)
        s_a = math.sin(alpha)
        u2_x = c_a * u_x + s_a * u_y
        u2_y = -s_a * u_x + c_a * u_y
        
        dots2 = u2_x[:, None] * v_x[None, :] + u2_y[:, None] * v_y[None, :]
        h2 = np.max(dots2, axis=1)
        
        h_sum = h1 + h2
        d_mins[i] = np.min(h_sum)
        
    return angles, d_mins

D_MIN_ANGLES, D_MIN_VALUES = compute_d_min_table()

def get_d_min(alpha):
    # alpha is relative angle
    # Normalize to [0, pi/6]
    alpha = abs(alpha) % (math.pi/3)
    if alpha > math.pi/6:
        alpha = math.pi/3 - alpha
    # Interpolate
    idx = np.interp(alpha, D_MIN_ANGLES, D_MIN_VALUES)
    return idx

START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Variables:
    # 0..10: x coords of 11 hexagons
    # 11..21: y coords of 11 hexagons
    # 22..32: angles of 11 hexagons (radians)
    # 33: outer center x
    # 34: outer center y
    # 35: outer rotation (beta)
    
    # Initial guess: Lattice packing
    # Use a known good lattice configuration as seed
    # e.g. 1 in center, 6 around, 4 more
    # Let's use the first candidate from the previous program as a seed
    seed_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    seed_pts = []
    for q, r in seed_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        seed_pts.append((x, y))
    
    x0 = []
    for i in range(N):
        x0.append(seed_pts[i][0])
        x0.append(seed_pts[i][1])
        x0.append(0.0) # angle
    x0.append(0.0) # outer cx
    x0.append(0.0) # outer cy
    x0.append(0.0) # outer beta
    
    x0 = np.array(x0)
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        xs = params[0:N]
        ys = params[N:2*N]
        angles = params[2*N:3*N]
        cx = params[3*N]
        cy = params[3*N+1]
        beta = params[3*N+2]
        
        # 1. Calculate enclosing hexagon size
        # Normals of outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0
        for i in range(N):
            dx = xs[i] - cx
            dy = ys[i] - cy
            for nx, ny in normals:
                proj = abs(dx * nx + dy * ny)
                if proj > max_proj:
                    max_proj = proj
        
        S = (max_proj + APOTHEM) / APOTHEM
        
        # 2. Calculate overlap penalty
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                rel_angle = angles[i] - angles[j]
                d_min = get_d_min(rel_angle)
                if dist < d_min:
                    # Penalty for violation
                    # Use quadratic penalty
                    penalty += (d_min - dist) ** 2
        
        # Return S + penalty * weight
        # Weight should be high enough to prevent overlap
        return S + penalty * 100.0

    # Optimization
    best_S = float('inf')
    best_params = None
    
    # Try a few random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Perturb seed
        if r == 0:
            x_init = x0
        else:
            x_init = x0 + np.random.normal(0, 0.1, size=len(x0))
            # Ensure angles are reasonable
            x_init[2*N:3*N] = np.random.uniform(-math.pi/6, math.pi/6, N)
            
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to seed
        best_params = x0
        best_S = objective(best_params)
        
    # Extract results
    xs = best_params[0:N]
    ys = best_params[N:2*N]
    angles = best_params[2*N:3*N]
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    beta = best_params[3*N+2]
    
    # Calculate final S
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
        
    max_proj = 0
    for i in range(N):
        dx = xs[i] - cx
        dy = ys[i] - cy
        for nx, ny in normals:
            proj = abs(dx * nx + dy * ny)
            if proj > max_proj:
                max_proj = proj
                
    final_S = (max_proj + APOTHEM) / APOTHEM
    
    # Check for overlaps (sanity check)
    # If overlap exists, the score might be invalid or penalized by evaluator?
    # The task says "disjoint". If they overlap, it's invalid.
    # We must ensure no overlap.
    # If the penalty method didn't work, we might have overlaps.
    # Let's add a final check and if overlaps exist, try to fix by scaling or rejecting.
    # But we must return a valid result.
    # If overlaps exist, we can try to increase S to accommodate? No, S is determined by positions.
    # If overlaps exist, the configuration is invalid.
    # We should probably fall back to the lattice solution if this fails.
    
    overlap_found = False
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            rel_angle = angles[i] - angles[j]
            d_min = get_d_min(rel_angle)
            if dist < d_min - 1e-9:
                overlap_found = True
                break
        if overlap_found:
            break
            
    if overlap_found:
        # Fallback to lattice
        print("Overlap found, falling back to lattice")
        # Use the seed lattice
        inner = []
        for i in range(N):
            inner.append([float(seed_pts[i][0]), float(seed_pts[i][1]), 0.0])
        center = [0.0, 0.0]
        # Calculate S for lattice
        max_proj = 0
        for i in range(N):
            dx = seed_pts[i][0]
            dy = seed_pts[i][1]
            for k in range(6):
                ang = math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                proj = abs(dx * nx + dy * ny)
                if proj > max_proj:
                    max_proj = proj
        side = (max_proj + APOTHEM) / APOTHEM
        rotation = 0.0
        return inner, center, side, rotation

    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), float(math.degrees(angles[i]))])
        
    center = [float(cx), float(cy)]
    side = float(final_S)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation
