import math
import time
import random
import numpy as np
from scipy.optimize import linprog, differential_evolution, minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin (evaluator limit 150s)
    
    # Unit hexagon vertices (flat-top, radius 1)
    base_verts = np.array([(math.cos(math.radians(a)), math.sin(math.radians(a))) 
                           for a in range(0, 360, 60)])
    # Normal directions (outward) for each edge: 30, 90, 150, 210, 270, 330
    base_normals = np.array([(math.cos(math.radians(a)), math.sin(math.radians(a))) 
                             for a in range(30, 390, 60)])
    
    def rot_matrix(deg):
        r = math.radians(deg)
        return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    
    def get_rotated_verts(deg):
        return base_verts @ rot_matrix(deg).T
    
    def get_rotated_normals(deg):
        return base_normals @ rot_matrix(deg).T
    
    def compute_side(centers, angles, outer_deg):
        """Compute minimum outer side length for given centers/angles."""
        # Center the cluster
        c = np.array(centers)
        c = c - c.mean(axis=0)
        
        outer_normals = get_rotated_normals(outer_deg)
        max_proj = 0.0
        for i in range(N):
            verts = get_rotated_verts(angles[i]) + c[i]
            projs = verts @ outer_normals.T  # (6, 6)
            max_proj = max(max_proj, float(np.max(np.abs(projs))))
        return 2.0 * max_proj / SQRT3
    
    def check_overlap(centers, angles, eps=1e-9):
        """Check if any two hexagons overlap using SAT."""
        for i in range(N):
            for j in range(i+1, N):
                vi = get_rotated_verts(angles[i]) + centers[i]
                vj = get_rotated_verts(angles[j]) + centers[j]
                ni = get_rotated_normals(angles[i])
                nj = get_rotated_normals(angles[j])
                for axis in list(ni) + list(nj):
                    pi = vi @ axis
                    pj = vj @ axis
                    if pi.max() < pj.min() - eps or pj.max() < pi.min() - eps:
                        break
                else:
                    return False
        return True
    
    def solve_fixed_orientation(angles, max_time=8.0):
        """Solve for optimal centers given fixed orientations using LP."""
        # Variables: x0..x10 (centers), x11 (S - outer side length)
        # Objective: minimize S (x11)
        # Constraints: for each hexagon i, each of 6 sides:
        #   (normal_k . center_i) + (normal_k . max vertex offset) <= S * SQRT3 / 2
        #   -(normal_k . center_i) - (normal_k . max vertex offset) <= S * SQRT3 / 2
        #   => normal_k . center_i + S * SQRT3/2 >= max_offset_k  (for +)
        #      -normal_k . center_i + S * SQRT3/2 >= max_offset_k (for -)
        
        # Precompute max vertex offset for each hexagon orientation and each outer normal
        outer_normals = get_rotated_normals(0)  # We'll use outer angle 0 for the LP, then optimize separately
        # Actually, let's include outer angle as a parameter
        # For now, fix outer angle to 0 and optimize it separately
        # Better: include outer angle in the LP by trying multiple outer angles
        
        # Let's do it properly: outer angle is a parameter, we'll try several
        best_result = None
        best_side = float('inf')
        
        # Try outer angles at 0, 5, 10, ..., 55 degrees
        for oa in range(0, 60, 5):
            outer_normals = get_rotated_normals(oa)
            
            c = np.zeros((N, 2))
            # Build constraints
            # For each hexagon i, each outer normal k:
            #   max over vertices v of (outer_normal_k . v) <= S * SQRT3 / 2
            #   vertex = center_i + rotated_base_vert
            #   outer_normal_k . (center_i + rot_base_vert) <= S * SQRT3 / 2
            #   outer_normal_k . center_i + outer_normal_k . rot_base_vert <= S * SQRT3 / 2
            #   outer_normal_k . center_i - S * SQRT3 / 2 <= -outer_normal_k . rot_base_vert
            # For the max: we need this for all 6 vertices, but the binding one is the max
            # Actually for LP we need: for each vertex v of hexagon i:
            #   outer_normal_k . (center_i + v) <= S * SQRT3 / 2
            #   -outer_normal_k . (center_i + v) <= S * SQRT3 / 2
            # That's 6 vertices * 6 normals * 2 = 72 constraints per hexagon
            
            A_ub = []
            b_ub = []
            
            apothem_coeff = SQRT3 / 2.0  # S * SQRT3/2 is the apothem
            
            for i in range(N):
                verts_i = get_rotated_verts(angles[i])
                for k in range(6):
                    on = outer_normals[k]
                    for v in verts_i:
                        # on . (center_i + v) <= S * SQRT3/2
                        # on[0]*cx + on[1]*cy - apothem_coeff*S <= -on[0]*vx - on[1]*vy
                        row = np.zeros(2*N + 1)
                        row[i*2] = on[0]
                        row[i*2+1] = on[1]
                        row[2*N] = -apothem_coeff
                        A_ub.append(row)
                        b_ub.append(-on[0]*v[0] - on[1]*v[1])
                        
                        # -on . (center_i + v) <= S * SQRT3/2
                        # -on[0]*cx - on[1]*cy - apothem_coeff*S <= on[0]*vx + on[1]*vy
                        row = np.zeros(2*N + 1)
                        row[i*2] = -on[0]
                        row[i*2+1] = -on[1]
                        row[2*N] = -apothem_coeff
                        A_ub.append(row)
                        b_ub.append(on[0]*v[0] + on[1]*v[1])
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Objective: minimize S (x11)
            c_obj = np.zeros(2*N + 1)
            c_obj[2*N] = 1.0
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)]*(2*N) + [(0, None)],
                              method='highs')
                if res.success:
                    side = res.x[2*N]
                    centers = res.x[:2*N].reshape(N, 2)
                    if side < best_side:
                        best_side = side
                        best_result = (centers, oa, side)
            except Exception:
                pass
            
            if time.time() > start_time + max_time:
                break
        
        return best_result
    
    def full_eval(centers, angles, outer_deg):
        """Full evaluation: check overlap and compute side."""
        if not check_overlap(centers, angles):
            return float('inf')
        return compute_side(centers, angles, outer_deg)
    
    def objective_de(x):
        """Objective for differential evolution over angles and outer angle."""
        if time.time() > deadline:
            return 1e10
        angles = x[:N]
        outer_deg = x[N] % 60.0
        
        # Solve LP for centers
        result = solve_fixed_orientation(angles, max_time=3.0)
        if result is None:
            return 1e10
        centers, oa, side = result
        
        # Verify no overlap
        if not check_overlap(centers, angles):
            return side + 100.0  # Penalty
        
        # Refine outer angle
        best_s = side
        for oa2 in np.arange(outer_deg - 5, outer_deg + 5, 0.5):
            s = compute_side(centers, angles, oa2 % 60.0)
            if s < best_s:
                best_s = s
        
        return best_s
    
    # Differential evolution over orientations
    best_side = float('inf')
    best_solution = None
    
    # Initial guess: all angles 0
    x0 = np.zeros(N + 1)
    
    try:
        result = differential_evolution(
            objective_de,
            bounds=[(0, 60)] * (N + 1),
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=True,
            init='sobol',
            update_interval=1,
            mutation=(0.5, 1.0),
            recombination=0.7,
            x0=x0,
            seed=42
        )
        if result.fun < best_side:
            best_side = result.fun
            best_solution = result.x
    except Exception:
        pass
    
    # If DE found something, refine it
    if best_solution is not None:
        angles = best_solution[:N]
        outer_deg = best_solution[N] % 60.0
        
        result = solve_fixed_orientation(angles, max_time=5.0)
        if result is not None:
            centers, oa, side = result
            if check_overlap(centers, angles):
                # Fine-tune outer angle
                for oa2 in np.arange(0, 60, 0.1):
                    s = compute_side(centers, angles, oa2)
                    if s < best_side:
                        best_side = s
                        oa = oa2
                best_solution = np.concatenate([angles, [oa]])
    
    # Fallback: honeycomb arrangement if DE failed
    if best_solution is None or best_side > 100:
        # Standard honeycomb: center + 6 around + 4 more
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        centers = []
        for q, r in axial:
            x = SQRT3 * (q + r/2.0)
            y = 1.5 * r
            centers.append([x, y])
        centers = np.array(centers)
        angles = np.zeros(N)
        outer_deg = 0.0
        side = compute_side(centers, angles, outer_deg)
        best_side = side
        best_solution = np.concatenate([angles, [outer_deg]])
    
    # Final refinement: local optimization of angles
    def local_obj(x):
        if time.time() > deadline:
            return 1e10
        angles = x[:N]
        outer_deg = x[N] % 60.0
        result = solve_fixed_orientation(angles, max_time=2.0)
        if result is None:
            return 1e10
        centers, oa, side = result
        if not check_overlap(centers, angles):
            return side + 50.0
        return compute_side(centers, angles, oa)
    
    try:
        res = minimize(local_obj, best_solution, method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 0.1, 'fatol': 0.001})
        if res.fun < best_side:
            best_side = res.fun
            best_solution = res.x
    except Exception:
        pass
    
    # Extract final solution
    angles = best_solution[:N]
    outer_deg = best_solution[N] % 60.0
    
    result = solve_fixed_orientation(angles, max_time=3.0)
    if result is not None:
        centers, oa, side = result
        if check_overlap(centers, angles):
            # Final fine-tune of outer angle
            for oa2 in np.arange(0, 60, 0.05):
                s = compute_side(centers, angles, oa2)
                if s < best_side:
                    best_side = s
                    oa = oa2
            # Center the configuration
            c = np.array(centers)
            c = c - c.mean(axis=0)
            inner_list = []
            for i in range(N):
                inner_list.append([float(c[i][0]), float(c[i][1]), float(angles[i])])
            return (inner_list, (0.0, 0.0), float(best_side), float(oa))
    
    # Ultimate fallback
    axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    centers = []
    for q, r in axial:
        x = SQRT3 * (q + r/2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    centers = centers - centers.mean(axis=0)
    angles = np.zeros(N)
    side = compute_side(centers, angles, 0.0)
    inner_list = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
    return (inner_list, (0.0, 0.0), float(side), 0.0)

# EVOLVE-BLOCK-END
