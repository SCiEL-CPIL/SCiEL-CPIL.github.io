import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0 # Safety margin for 150s limit
    random.seed(42)
    np.random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        """Check if two hexagons are disjoint using SAT."""
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Calculate the minimum side length of the outer hexagon containing all inner hexagons."""
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def flatten(centers, angles, ocx, ocy, otheta):
        v = []
        for i in range(N):
            v.extend([centers[i][0], centers[i][1], angles[i]])
        v.extend([ocx, ocy, otheta])
        return v

    def unflatten(v):
        c = []
        a = []
        for i in range(N):
            c.append([v[3*i], v[3*i+1]])
            a.append(v[3*i+2])
        ocx = v[3*N]
        ocy = v[3*N+1]
        otheta = v[3*N+2]
        return c, a, ocx, ocy, otheta

    def objective(v):
        c, a, ocx, ocy, otheta = unflatten(v)
        # If any angle is out of bounds or invalid, return large value
        # But SLSQP handles bounds. 
        return compute_outer_side(c, a, ocx, ocy, otheta)

    def generate_lattice_solution():
        """Generate a hexagonal close packing configuration."""
        # Minimum distance between centers is sqrt(3)
        d = math.sqrt(3)
        # Basis vectors for hexagonal lattice
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [[0.0, 0.0]]
        # Ring 1: 6 hexagons
        ring1_vectors = [u, v, v-u, -u, -v, u-v]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
            
        # Ring 2: We need 5 more. 
        # Candidates are at distance sqrt(3)*sqrt(3) = 3? No.
        # Distance from center to ring 2 is d*sqrt(3) = 3?
        # Let's just pick 5 from the next nearest neighbors.
        # Neighbors of Ring 1 are at distance d from Ring 1.
        # Ring 2 positions are typically at 2u, 2v, etc. or u+v.
        # u+v has length d. (u+v) is a ring 1 vector? No, u+v is not in ring1.
        # Wait, u and v are 60 deg apart. |u+v| = d.
        # So u+v is at distance d from origin. It is a ring 1 position.
        # Let's list all lattice points within radius 2*d.
        
        points = set()
        for i in range(-2, 3):
            for j in range(-2, 3):
                p = i * u + j * v
                # Check if it's one of the first 7
                if i == 0 and j == 0: continue
                # Check if it's in ring 1
                is_ring1 = False
                for rv in ring1_vectors:
                    if np.allclose(p, rv, atol=1e-9):
                        is_ring1 = True
                        break
                if is_ring1: continue
                
                dist = np.linalg.norm(p)
                # We want the closest 5 to ring 1
                points.add((round(p[0], 9), round(p[1], 9)))
        
        # Convert to list and sort by distance
        pts_list = [np.array([x, y]) for x, y in points]
        pts_list.sort(key=lambda p: np.linalg.norm(p))
        
        # Add the first 5
        for p in pts_list[:5]:
            centers.append([p[0], p[1]])
            
        # All angles 0 for simplicity in lattice, or 30? 
        # Orientation doesn't matter for distance constraint, but for outer side it does.
        # 0 is fine.
        angles = [0.0] * 12
        
        # Center of mass for outer hex
        all_verts = []
        for i in range(12):
            verts = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        ocx = np.mean(all_verts[:, 0])
        ocy = np.mean(all_verts[:, 1])
        otheta = 0.0
        
        return centers, angles, ocx, ocy, otheta

    def solve_packing(centers, angles, ocx, ocy, otheta, max_time=20.0):
        start_time = time.time()
        
        x0 = flatten(centers, angles, ocx, ocy, otheta)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        # Constraints: Pairwise distance squared >= 3 (minus small epsilon for numerical stability)
        # d^2 >= 3 - 1e-6
        eps = 1e-6
        min_d2 = 3.0 - eps
        
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    xj = x[3*j]
                    yj = x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - min_d2
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        def obj_wrapper(x):
            if time.time() > start_time + max_time or time.time() > deadline - 5:
                return 1e10
            return objective(x)
        
        try:
            result = minimize(obj_wrapper, x0, method='SLSQP', bounds=bounds, 
                             constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            
            if result.fun < 1e9:
                c, a, ocx, ocy, otheta = unflatten(result.x)
                # Verify disjointness with SAT just in case
                verts = []
                for i in range(N):
                    v = get_hex_vertices(c[i][0], c[i][1], 1.0, a[i])
                    verts.append(v)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not are_disjoint_sat(verts[i], verts[j]):
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return c, a, ocx, ocy, otheta, result.fun
        except Exception:
            pass
        return None

    # 1. Generate initial lattice solution
    centers, angles, ocx, ocy, otheta = generate_lattice_solution()
    initial_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_side = initial_side
    best_config = (centers, angles, ocx, ocy, otheta)
    
    # 2. Optimize from lattice
    if time.time() < deadline - 20:
        res = solve_packing(centers, angles, ocx, ocy, otheta, max_time=min(30, deadline - time.time() - 10))
        if res:
            c, a, ocx, ocy, otheta, side = res
            if side < best_side:
                best_side = side
                best_config = (c, a, ocx, ocy, otheta)

    # 3. Try a few random restarts with SLSQP if time permits
    # Random restarts might find better global minima if lattice is local
    for r in range(3):
        if time.time() > deadline - 15:
            break
            
        # Perturb the best config
        c_init = [[c[0] + random.gauss(0, 0.2), c[1] + random.gauss(0, 0.2)] for c in best_config[0]]
        a_init = [(a + random.gauss(0, 5.0)) % 60 for a in best_config[1]]
        ocx_init = best_config[2] + random.gauss(0, 0.1)
        ocy_init = best_config[3] + random.gauss(0, 0.1)
        otheta_init = (best_config[4] + random.gauss(0, 5.0)) % 60
        
        res = solve_packing(c_init, a_init, ocx_init, ocy_init, otheta_init, max_time=min(20, deadline - time.time() - 5))
        if res:
            c, a, ocx, ocy, otheta, side = res
            if side < best_side:
                best_side = side
                best_config = (c, a, ocx, ocy, otheta)

    # 4. Final Polish: Use Nelder-Mead on the best config found, ignoring constraints 
    #    but with a heavy penalty if they violate, to break out of SLSQP's local optima?
    #    Actually, SLSQP is good. Let's just do a high-precision SLSQP run on the best config.
    if time.time() < deadline - 10:
        c, a, ocx, ocy, otheta = best_config
        # Tighten constraints? No, just run SLSQP again with more iterations
        start_time = time.time()
        x0 = flatten(c, a, ocx, ocy, otheta)
        
        bounds = []
        for i in range(N):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        eps = 1e-8
        min_d2 = 3.0 - eps
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    xj = x[3*j]
                    yj = x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - min_d2
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                             constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if result.fun < best_side - 1e-10:
                c, a, ocx, ocy, otheta = unflatten(result.x)
                # Verify
                verts = []
                for i in range(N):
                    v = get_hex_vertices(c[i][0], c[i][1], 1.0, a[i])
                    verts.append(v)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not are_disjoint_sat(verts[i], verts[j]):
                            valid = False
                            break
                    if not valid: break
                if valid:
                    best_side = result.fun
                    best_config = (c, a, ocx, ocy, otheta)
        except Exception:
            pass

    c, a, ocx, ocy, otheta = best_config
    
    # Final verification
    verts = []
    for i in range(N):
        v = get_hex_vertices(c[i][0], c[i][1], 1.0, a[i])
        verts.append(v)
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not are_disjoint_sat(verts[i], verts[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to lattice
        c, a, ocx, ocy, otheta = generate_lattice_solution()
        best_side = compute_outer_side(c, a, ocx, ocy, otheta)

    inner = [[c[i][0], c[i][1], a[i]] for i in range(N)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
