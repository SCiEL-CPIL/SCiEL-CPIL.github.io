"""Combined program: Hexagonal lattice initialization + SLSQP optimization for hexagon packing."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    # Helper functions
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """
        Compute the minimum side length of a regular hexagon centered at (ocx, ocy)
        with rotation otheta_deg that contains all vertices of the inner hexagons.
        """
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        # The 6 outward normals of a regular hexagon are at angles otheta + 30 + 60*k
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = math.radians(angles[i])
                # Check all 6 vertices of the i-th inner hexagon
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # The apothem (inradius) of the outer hexagon must be >= max_proj
        # Apothem = side * sqrt(3) / 2
        # So side = 2 * apothem / sqrt(3)
        if max_proj <= 0:
            return 0.0
        return 2.0 * max_proj / math.sqrt(3)

    def check_disjoint(centers, min_dist_sq):
        """Check if all hexagons are disjoint based on center distance."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx*dx + dy*dy < min_dist_sq:
                    return False
        return True

    # Initialization: Hexagonal lattice cluster
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We use a slightly larger distance to allow the optimizer to find exact touching or slightly separated
    d = math.sqrt(3) 
    # Lattice vectors
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # 1 center
    centers = [np.array([0.0, 0.0])]
    angles = [0.0]
    
    # 6 in first ring
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    for vec in ring1_vectors:
        centers.append(vec)
        angles.append(0.0)
        
    # 5 in second ring (to make 12 total)
    # We pick 5 from the available positions in the second ring
    # Second ring positions are at distance 2*d or d*sqrt(7) etc.
    # Let's just generate a few candidates and pick 5 that are far enough from existing
    # Or simpler: use the lattice points at (2u, 2v, 2v-2u, -2u, -2v, 2u-2v)
    # But we only need 5.
    
    # Let's define the second ring vectors explicitly
    # The neighbors of the 6 ring1 hexagons are at:
    # u+u=2u, u+v, u+v-u=v (already there? no, v is ring1)
    # Let's just use a set of candidate vectors for the second ring
    second_ring_candidates = [
        2*u, 2*v, 2*(v-u), -2*u, -2*v, 2*(u-v),
        u+v, v+v-u, v-u+u, -u-v, -v-(v-u), -(v-u)-u,
        u-(v-u), v-(v-u), (v-u)-u, -u+(v-u), -v+(v-u), -(v-u)+(v-u) # messy
    ]
    
    # Better: Generate all lattice points in a radius and filter
    # Lattice points are i*u + j*v
    # We want 5 more points.
    # Let's just hardcode a good set of 5 that forms a compact shape
    # Or use the 6th ring point and rotate?
    
    # Let's use the 6 vectors: 2u, 2v, 2(v-u), -2u, -2(v-u), -2v ?
    # Actually, the most compact is to fill the gaps.
    # Let's just take 5 of the 6 "corner" points of the hexagon formed by the first 7.
    # The 7 hexagons form a hexagon of side 2 (in lattice units).
    # The next layer adds 6 more.
    # Let's pick 5 of these 6.
    
    # Vectors for the 6th ring (distance from center is 2*d? No, lattice step is d)
    # The 6 hexagons in ring 1 are at distance d.
    # The hexagons in ring 2 that touch ring 1 are at distance 2d? No.
    # In a hex lattice, the distance from center to ring 2 hexagons is 2d?
    # No, the ring 2 hexagons are at positions like u+v, v+v-u, etc.
    # Distance of u+v is d*sqrt(3) ?
    # |u+v|^2 = |u|^2 + |v|^2 + 2 u.v = d^2 + d^2 + 2 d^2 cos(60) = 2d^2 + d^2 = 3d^2.
    # So distance is d*sqrt(3) = sqrt(3)*sqrt(3) = 3.
    # Wait, d = sqrt(3). So distance is 3.
    
    # Let's just use the 6 vectors:
    # p1 = 2u
    # p2 = u+v
    # p3 = v+v-u
    # p4 = -u+v  (wait, this is ring 1? no, v-u is ring 1. -u+v is ring 1? no. v-u is ring 1. -u+v is same as v-u.
    # Let's list ring 1 again: u, v, v-u, -u, -v, u-v.
    
    # Ring 2 candidates (touching ring 1):
    # u+u = 2u
    # u+v
    # v+(v-u) = 2v-u
    # (v-u)+(-u) = v-2u
    # (-u)+(-v) = -u-v
    # (-v)+(u-v) = u-2v
    # (u-v)+u = 2u-v
    # u+(-v) ? No, u and -v are not neighbors?
    
    # Let's just pick 5 of these 6 "corner" type vectors to form a pentagon-like shape or just 5 of the 6.
    # For symmetry, maybe we should optimize the choice?
    # Let's just pick 5: 2u, u+v, 2v-u, v-2u, -u-v.
    
    ring2_vectors = [
        2*u,
        u+v,
        2*v - u,
        v - 2*u,
        -u - v
    ]
    
    for vec in ring2_vectors:
        centers.append(vec)
        angles.append(0.0)
        
    # Ensure we have 12
    assert len(centers) == N, f"Expected 12 centers, got {len(centers)}"
    
    # Convert to list of floats for scipy
    x0 = []
    for i in range(N):
        x0.extend([centers[i][0], centers[i][1], angles[i]])
    # Outer hexagon parameters: center x, y, angle
    x0.extend([0.0, 0.0, 0.0])
    
    def objective(x):
        if time.time() > deadline - 10:
            return 1e10
        c = [[x[3*i], x[3*i+1]] for i in range(N)]
        a = [x[3*i+2] for i in range(N)]
        ocx, ocy, otheta = x[3*N], x[3*N+1], x[3*N+2]
        return compute_outer_side(c, a, ocx, ocy, otheta)
    
    # Constraints: pairwise distance >= sqrt(3) - eps
    min_dist = math.sqrt(3) - 0.001
    min_dist_sq = min_dist * min_dist
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            def dist_constraint(x, i=i, j=j):
                xi = x[3*i]
                yi = x[3*i+1]
                xj = x[3*j]
                yj = x[3*j+1]
                return (xi - xj)**2 + (yi - yj)**2 - min_dist_sq
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.extend([(-10, 10), (-10, 10), (0, 60)])
    bounds.extend([(-10, 10), (-10, 10), (0, 60)])
    
    best_side = float('inf')
    best_x = x0[:]
    
    try:
        # Run SLSQP
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
        
        if res.success or (hasattr(res, 'fun') and res.fun < best_side):
            if hasattr(res, 'fun') and res.fun < best_side:
                best_side = res.fun
                best_x = res.x
    
    except Exception as e:
        pass

    # Extract best result
    c = [[best_x[3*i], best_x[3*i+1]] for i in range(N)]
    a = [best_x[3*i+2] for i in range(N)]
    ocx, ocy, otheta = best_x[3*N], best_x[3*N+1], best_x[3*N+2]
    
    # Final polish: try to adjust outer hexagon slightly better with local search
    # This can sometimes fix small numerical issues
    def polish(centers, angles, ocx, ocy, otheta, steps=50):
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        for _ in range(steps):
            if time.time() > deadline - 5:
                break
            improved = False
            # Small perturbations of outer parameters
            for dx in [-0.01, 0.0, 0.01]:
                for dy in [-0.01, 0.0, 0.01]:
                    for da in [-1.0, 0.0, 1.0]:
                        new_side = compute_outer_side(centers, angles, ocx+dx, ocy+dy, otheta+da)
                        if new_side < best_side:
                            best_side = new_side
                            ocx += dx
                            ocy += dy
                            otheta += da
                            improved = True
            if not improved:
                break
        return ocx, ocy, otheta, best_side

    ocx, ocy, otheta, best_side = polish(c, a, ocx, ocy, otheta)
    
    # Construct return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for i in range(N):
        inner.append([c[i][0], c[i][1], a[i]])
        
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
