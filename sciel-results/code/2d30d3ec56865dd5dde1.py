import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Unit hexagon vertices (flat-top)
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    def get_hex_verts(pos, rot_deg):
        """Get vertices of a unit hexagon at pos, rotated by rot_deg degrees."""
        rad = math.radians(rot_deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[c, -s], [s, c]])
        return pos + unit_verts @ R.T

    def polygons_overlap_sat(verts1, verts2, eps=1e-9):
        """Check if two convex polygons overlap using SAT."""
        # Collect all edges from both polygons
        edges = []
        for v in (verts1, verts2):
            for i in range(len(v)):
                j = (i + 1) % len(v)
                edge = v[j] - v[i]
                # Normal perpendicular to edge
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    n = n / norm
                    edges.append(n)
        
        for n in edges:
            # Project both polygons onto normal
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for v in verts1:
                p = np.dot(v, n)
                min1 = min(min1, p)
                max1 = max(max1, p)
            for v in verts2:
                p = np.dot(v, n)
                min2 = min(min2, p)
                max2 = max(max2, p)
            
            # Check for gap
            if max1 + eps < min2 or max2 + eps < min1:
                return False
        return True

    def calc_S_general(positions, rotations, beta_deg):
        """
        Calculate outer side length for given positions, rotations, and outer rotation.
        positions: list of (x,y)
        rotations: list of rotation angles (degrees) for each inner hex
        beta_deg: outer hexagon rotation
        """
        beta_rad = math.radians(beta_deg)
        
        # Outer hexagon normals
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            max_inner_proj = 0.0
            for i in range(len(positions)):
                x, y = positions[i]
                proj_center = x * n[0] + y * n[1]
                
                # Support of this inner hex in direction n
                # Rotate n by -rotation_i to get support of unit hex
                rot_rad = math.radians(rotations[i])
                c = math.cos(-rot_rad)
                s = math.sin(-rot_rad)
                n_rot = np.array([c*n[0] - s*n[1], s*n[0] + c*n[1]])
                
                h_val = np.max(unit_verts @ n_rot)
                
                total_proj = proj_center + h_val
                if total_proj > max_inner_proj:
                    max_inner_proj = total_proj
            
            if max_inner_proj > max_proj:
                max_proj = max_inner_proj
                
        return max_proj * 2.0 / SQRT3

    def check_no_overlap(positions, rotations, eps=1e-8):
        """Check if any two hexagons overlap."""
        verts = [get_hex_verts(np.array(pos), rot) for pos, rot in zip(positions, rotations)]
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if polygons_overlap_sat(verts[i], verts[j], eps):
                    return False
        return True

    def objective(params):
        """Optimization objective: minimize S with overlap penalty."""
        # params: x1,y1,r1, x2,y2,r2, ..., x12,y12,r12, beta
        positions = [(params[i*3], params[i*3+1]) for i in range(12)]
        rotations = [params[i*3+2] for i in range(12)]
        beta = params[-1]
        
        S = calc_S_general(positions, rotations, beta)
        
        # Check overlaps and add penalty
        verts = [get_hex_verts(np.array(pos), rot) for pos, rot in zip(positions, rotations)]
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                if polygons_overlap_sat(verts[i], verts[j], -1e-10):  # eps negative for strict overlap
                    # Estimate overlap penalty based on center distance
                    d = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                    penalty += max(0, 2.0 - d) ** 2 * 1000
        
        return S + penalty

    # Generate initial topologies
    def get_hex_positions(topo, scale=2.0):
        pts = []
        for q, r in topo:
            x = scale * (q + r/2.0)
            y = scale * (r * SQRT3/2.0)
            pts.append((x, y))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return [(x-cx, y-cy) for x, y in pts]

    topologies = []
    
    # 3x4 grid
    t1 = [(q, r) for q in range(3) for r in range(4)]
    topologies.append(t1)
    
    # 4x3 grid
    t2 = [(q, r) for q in range(4) for r in range(3)]
    topologies.append(t2)
    
    # 2x6 grid
    t3 = [(q, r) for q in range(2) for r in range(6)]
    topologies.append(t3)
    
    # 6x2 grid
    t4 = [(q, r) for q in range(6) for r in range(2)]
    topologies.append(t4)
    
    # Hexagonal cluster: center + ring 1 (6) + 5 from ring 2
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    ]
    if len(set(t5)) == 12:
        topologies.append(t5)
    
    # Another cluster variation
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (-2,1), (2,-1)
    ]
    # Check uniqueness
    if len(set(t6)) == 12 and all(c in t5 for c in t6) == False:
        topologies.append(t6)
    
    # Linear arrangement
    t7 = [(q, 0) for q in range(12)]
    topologies.append(t7)

    best_S = float('inf')
    best_params = None

    # Phase 1: Grid search over topologies with uniform rotation
    for topo in topologies:
        if time.time() > deadline:
            break
        
        positions = get_hex_positions(topo, scale=2.0)
        
        # Try a few rotation angles
        for alpha in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline:
                break
            for beta in [0, 15, 30, 45, 60]:
                rotations = [alpha] * 12
                S = calc_S_general(positions, rotations, beta)
                
                if check_no_overlap(positions, rotations):
                    if S < best_S:
                        best_S = S
                        best_params = (positions, rotations, beta)
    
    # Phase 2: Local refinement using Nelder-Mead
    if best_params is not None and time.time() < deadline - 20:
        positions, rotations, beta = best_params
        
        def refine_objective(params):
            # params: beta, then small perturbations to each position and rotation
            b = params[0]
            # For simplicity, just optimize beta and keep positions fixed
            S = calc_S_general(positions, rotations, b)
            return S
        
        try:
            res = minimize(refine_objective, [beta], method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = (positions, rotations, res.x[0])
        except:
            pass
    
    # Phase 3: Try differential evolution on a simplified parameter space
    if time.time() < deadline - 30:
        # Use a representative topology
        positions = get_hex_positions(topologies[0], scale=2.0)
        
        def de_objective(params):
            # params: 12 rotations + 1 beta
            rotations = params[:12]
            beta = params[-1]
            
            S = calc_S_general(positions, rotations, beta)
            
            # Check overlap
            verts = [get_hex_verts(np.array(pos), rot) for pos, rot in zip(positions, rotations)]
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    if polygons_overlap_sat(verts[i], verts[j], -1e-10):
                        d = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                        penalty += max(0, 2.0 - d) ** 2 * 1000
            return S + penalty
        
        bounds = [(0, 360)] * 12 + [(0, 60)]
        
        try:
            res = differential_evolution(de_objective, bounds, maxiter=50, 
                                        tol=1e-6, seed=42, polish=False,
                                        update_interval=1)
            if res.fun < best_S:
                rotations = res.x[:12]
                beta = res.x[-1]
                # Verify no overlap
                verts = [get_hex_verts(np.array(pos), rot) for pos, rot in zip(positions, rotations)]
                if check_no_overlap(positions, rotations):
                    best_S = res.fun
                    best_params = (positions, rotations, beta)
        except:
            pass

    # Final fallback
    if best_params is None:
        positions = get_hex_positions(topologies[0], scale=2.0)
        rotations = [0] * 12
        beta = 0
        best_S = calc_S_general(positions, rotations, beta)
        best_params = (positions, rotations, beta)
    
    positions, rotations, beta = best_params
    
    inner = []
    for (x, y), rot in zip(positions, rotations):
        inner.append([x, y, rot])
    
    return inner, (0.0, 0.0), best_S, beta
