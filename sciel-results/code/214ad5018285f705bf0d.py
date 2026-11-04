import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side(pos, ang, outer_center, outer_angle):
        """Calculate outer hexagon side length given inner positions, angles, and outer params."""
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        oc = np.array(outer_center)
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i])
            verts_rel = verts - oc
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / SQRT3

    def sat_gap(p1, p2):
        """Returns minimum separation between two hexagons. Negative if overlapping."""
        min_sep = float('inf')
        axes = []
        for poly in [p1, p2]:
            for k in range(6):
                edge = poly[(k+1)%6] - poly[k]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    axes.append(n/norm)
        
        for axis in axes:
            proj1 = np.dot(p1, axis)
            proj2 = np.dot(p2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            sep = max(min1, min2) - min(max1, max2)
            if sep < min_sep:
                min_sep = sep
            if min_sep > 0:
                break
        
        return min_sep

    def is_valid(pos, ang, tol=1e-6):
        """Check if configuration is valid (no overlaps)."""
        polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective(params):
        """
        Objective function for Nelder-Mead.
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, c_angle]
        Returns outer side length if valid, else inf.
        """
        pos = np.array([[params[3*i], params[3*i+1]] for i in range(N)])
        ang = np.array([params[3*i+2] for i in range(N)])
        oc = np.array([params[-3], params[-2]])
        oa = params[-1]
        
        if not is_valid(pos, ang, tol=1e-5):
            return float('inf')
            
        return calc_outer_side(pos, ang, oc, oa)

    def optimize_config(init_pos, init_ang):
        """
        Run Nelder-Mead optimization starting from a valid configuration.
        """
        # Flatten parameters
        init_params = []
        for i in range(N):
            init_params.extend([init_pos[i][0], init_pos[i][1], init_ang[i]])
        # Initial outer center and angle
        centroid = np.mean(init_pos, axis=0)
        init_params.extend([centroid[0], centroid[1], 0.0])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            # Check if result is valid
            final_params = res.x
            final_pos = np.array([[final_params[3*i], final_params[3*i+1]] for i in range(N)])
            final_ang = np.array([final_params[3*i+2] for i in range(N)])
            final_oc = np.array([final_params[-3], final_params[-2]])
            final_oa = final_params[-1]
            
            if is_valid(final_pos, final_ang, tol=1e-5):
                side = calc_outer_side(final_pos, final_ang, final_oc, final_oa)
                return final_pos, final_ang, final_oc, final_oa, side
        except Exception:
            pass
            
        # If optimization failed or resulted in invalid config, return initial (which was valid)
        side = calc_outer_side(init_pos, init_ang, centroid, 0.0)
        return init_pos, init_ang, centroid, 0.0, side

    # Generate initial configurations using lattice-based approach
    inits = []
    
    # 1. Standard honeycomb lattice (guaranteed valid)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take several different subsets of 12 points from the lattice to provide diverse starts
    # We want compact clusters. The 12 closest points to origin form a valid hexagon-like shape.
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append((init_pos, init_ang))
    
    # 2. Slightly perturbed lattice to break symmetry
    np.random.seed(42)
    for k in range(4):
        pos = init_pos.copy()
        # Small random perturbations that keep validity (lattice has slack)
        for i in range(N):
            pos[i] += np.random.uniform(-0.1, 0.1, 2)
        ang = np.random.uniform(0, 2*math.pi, N) * 0.1 # Small angle perturbations
        # Ensure validity
        if is_valid(pos, ang):
            inits.append((pos, ang))
            
    # 3. Ring configuration (check validity)
    ring_pos = []
    ring_ang = []
    r = 2.5
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    if is_valid(ring_pos, np.array(ring_ang)):
        inits.append((ring_pos, np.array(ring_ang)))

    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_oc = np.array([0.0, 0.0])
    best_oa = 0.0
    
    # Process each initial configuration
    for idx, (init_pos, init_ang) in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Optimize
        opt_pos, opt_ang, opt_oc, opt_oa, opt_side = optimize_config(init_pos, init_ang)
        
        if opt_side < best_side:
            best_side = opt_side
            best_pos = opt_pos
            best_ang = opt_ang
            best_oc = opt_oc
            best_oa = opt_oa
            
    # Fallback if no valid configuration found (should not happen with lattice start)
    if best_pos is None:
        best_pos = np.array(lattice_points[:12])
        best_pos -= np.mean(best_pos, axis=0)
        best_ang = np.zeros(N)
        best_oc, best_oa = np.mean(best_pos, axis=0), 0.0
        best_side = calc_outer_side(best_pos, best_ang, best_oc, best_oa)
    
    # Construct return values
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], np.degrees(best_ang[i])])
    
    return inner, tuple(best_oc), best_side, best_oa
