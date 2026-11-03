import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Hexagon vertices for a unit hexagon at origin with given angle
    def hex_vertices(angle):
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3.0
            verts.append((math.cos(a), math.sin(a)))
        return np.array(verts)
    
    # Check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hex_overlap(v1, v2, angle1, angle2):
        verts1 = v1 + hex_vertices(angle1)
        verts2 = v2 + hex_vertices(angle2)
        
        # Get all edges from both hexagons
        edges = []
        for i in range(6):
            e1 = verts1[(i+1)%6] - verts1[i]
            edges.append(np.array([-e1[1], e1[0]]))
            e2 = verts2[(i+1)%6] - verts2[i]
            edges.append(np.array([-e2[1], e2[0]]))
        
        for axis in edges:
            axis = axis / (np.linalg.norm(axis) + EPS)
            proj1 = verts1 @ axis
            proj2 = verts2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - EPS or max2 < min1 - EPS:
                return False
        return True
    
    # Compute the minimum enclosing hexagon size for a set of points
    def compute_enclosing_hex(pts, angle):
        # For a regular hexagon with apothem D, the constraint is:
        # |dot(p - center, n_k)| <= D for all 6 normals n_k
        # We want to find center and D that minimize D
        # This is a linear programming problem
        
        normals = []
        for k in range(6):
            a = angle + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(a), math.sin(a)))
        
        # Use scipy's linprog
        from scipy.optimize import linprog
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S
            else:
                return float('inf')
        except Exception:
            return float('inf')
    
    # Fast overlap check using distance approximation
    def fast_overlap_check(positions, angles):
        """Quick check if any two hexagons overlap"""
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance between centers of non-overlapping unit hexagons is 2*APOTHEM = sqrt(3)
                if dist < 2*APOTHEM - 0.1:
                    # More precise check
                    if hex_overlap(np.array(positions[i]), np.array(positions[j]), angles[i], angles[j]):
                        return True
        return False
    
    # Soft penalty for overlaps
    def soft_overlap_penalty(positions, angles, temperature):
        penalty = 0.0
        n = len(positions)
        min_dist = 2*APOTHEM
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    # Quadratic penalty that becomes very large when overlapping
                    violation = min_dist - dist
                    penalty += violation * violation / (temperature + EPS)
        return penalty
    
    # Simulated annealing optimizer
    def simulated_annealing():
        # Initial configuration: hexagonal lattice
        positions = []
        angles = []
        
        # Start with a dense hexagonal arrangement
        lattice_positions = [
            (0, 0),
            (2*APOTHEM, 0), (-2*APOTHEM, 0),
            (APOTHEM, 1.5), (-APOTHEM, 1.5),
            (APOTHEM, -1.5), (-APOTHEM, -1.5),
            (3*APOTHEM, 0), (-3*APOTHEM, 0),
            (0, 3), (0, -3),
            (2*APOTHEM, 3), (-2*APOTHEM, 3)
        ]
        
        # Select 11 best positions
        random.shuffle(lattice_positions)
        for pos in lattice_positions[:N]:
            positions.append(pos)
            angles.append(0.0)
        
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
        best_S = float('inf')
        
        # Compute initial S
        pts = np.array(positions)
        for beta in np.linspace(0, math.pi/6, 7):
            S = compute_enclosing_hex(pts, beta)
            if S < best_S:
                best_S = S
                best_positions = [p[:] for p in positions]
                best_angles = angles[:]
        
        # Simulated annealing
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        max_iter = 30000
        iter_count = 0
        
        current_positions = [p[:] for p in positions]
        current_angles = angles[:]
        
        while T > T_min and iter_count < max_iter:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            iter_count += 1
            
            # Generate a new candidate
            new_positions = [p[:] for p in current_positions]
            new_angles = current_angles[:]
            
            # Random perturbation
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            new_positions[idx][0] += dx
            new_positions[idx][1] += dy
            
            # Sometimes perturb angle
            if random.random() < 0.3:
                new_angles[idx] += random.gauss(0, 0.2)
            
            # Sometimes perturb multiple points
            if random.random() < 0.2:
                idx2 = random.randint(0, N-1)
                if idx2 != idx:
                    dx2 = random.gauss(0, 0.2)
                    dy2 = random.gauss(0, 0.2)
                    new_positions[idx2][0] += dx2
                    new_positions[idx2][1] += dy2
            
            # Compute penalty
            penalty = soft_overlap_penalty(new_positions, new_angles, T)
            
            # Compute S
            pts = np.array(new_positions)
            S = float('inf')
            for beta in np.linspace(0, math.pi/6, 5):
                S_beta = compute_enclosing_hex(pts, beta)
                if S_beta < S:
                    S = S_beta
            
            total_cost = S + penalty * 10.0
            
            # Accept or reject
            if total_cost < best_S:
                best_positions = [p[:] for p in new_positions]
                best_angles = new_angles[:]
                best_S = total_cost
                current_positions = new_positions
                current_angles = new_angles
            else:
                # Metropolis criterion
                if random.random() < math.exp(-total_cost / (T + EPS)):
                    current_positions = new_positions
                    current_angles = new_angles
            
            T *= cooling
        
        return best_positions, best_angles, best_S
    
    # Multi-start simulated annealing
    best_global_S = float('inf')
    best_global_positions = None
    best_global_angles = None
    
    n_starts = 8
    for start in range(n_starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Different initial configurations for each start
        random.seed(start * 12345 + 678)
        
        if start == 0:
            # Hexagonal lattice
            lattice = [
                (0, 0),
                (2*APOTHEM, 0), (-2*APOTHEM, 0),
                (APOTHEM, 1.5), (-APOTHEM, 1.5),
                (APOTHEM, -1.5), (-APOTHEM, -1.5),
                (3*APOTHEM, 0), (-3*APOTHEM, 0),
                (0, 3), (0, -3)
            ]
        elif start == 1:
            # Square-ish arrangement
            lattice = [
                (0, 0),
                (2, 0), (-2, 0), (0, 2), (0, -2),
                (2, 2), (-2, 2), (2, -2), (-2, -2),
                (4, 0), (-4, 0)
            ]
        elif start == 2:
            # Ring arrangement
            lattice = []
            lattice.append((0, 0))
            for k in range(6):
                a = k * math.pi / 3
                lattice.append((2*APOTHEM*math.cos(a), 2*APOTHEM*math.sin(a)))
            for k in range(5):
                a = math.pi/6 + k * math.pi / 3
                lattice.append((3.5*math.cos(a), 3.5*math.sin(a)))
        else:
            # Random initial positions
            lattice = []
            for i in range(N):
                r = random.uniform(0, 3)
                a = random.uniform(0, 2*math.pi)
                lattice.append((r*math.cos(a), r*math.sin(a)))
        
        # Select 11 positions
        random.shuffle(lattice)
        positions = [list(p) for p in lattice[:N]]
        angles = [0.0] * N
        
        # Run simulated annealing
        positions, angles, S = simulated_annealing()
        
        if S < best_global_S:
            best_global_S = S
            best_global_positions = positions
            best_global_angles = angles
    
    # Refine the best solution
    if best_global_positions is None:
        best_global_positions = [[0.0, 0.0]] * N
        best_global_angles = [0.0] * N
        best_global_S = float('inf')
    
    # Final refinement with local search
    def local_refinement(positions, angles):
        best_S = float('inf')
        best_pos = [p[:] for p in positions]
        best_ang = angles[:]
        
        for iteration in range(100):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            improved = False
            for i in range(N):
                for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                        new_pos = [p[:] for p in best_pos]
                        new_pos[i][0] += dx
                        new_pos[i][1] += dy
                        
                        pts = np.array(new_pos)
                        S = float('inf')
                        for beta in np.linspace(0, math.pi/6, 7):
                            S_beta = compute_enclosing_hex(pts, beta)
                            if S_beta < S:
                                S = S_beta
                        
                        # Check for overlaps
                        if fast_overlap_check(new_pos, best_ang):
                            S += 100.0
                        
                        if S < best_S - 1e-6:
                            best_S = S
                            best_pos = new_pos
                            improved = True
            
            if not improved:
                break
        
        return best_pos, best_ang, best_S
    
    best_global_positions, best_global_angles, best_global_S = local_refinement(best_global_positions, best_global_angles)
    
    # Find the optimal beta (rotation of outer hexagon)
    best_beta = 0.0
    best_S_final = float('inf')
    pts = np.array(best_global_positions)
    
    for beta in np.linspace(0, math.pi/6, 101):
        S = compute_enclosing_hex(pts, beta)
        if S < best_S_final:
            best_S_final = S
            best_beta = beta
    
    # Compute the center of the enclosing hexagon
    from scipy.optimize import linprog
    normals = []
    for k in range(6):
        a = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(a), math.sin(a)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx = res.x[0]
            cy = res.x[1]
            D = res.x[2]
            S_final = (D + APOTHEM) / APOTHEM
        else:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            S_final = best_S_final
    except Exception:
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        S_final = best_S_final
    
    angle_deg = math.degrees(best_beta)
    
    # Scale positions slightly to ensure no overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for i, (x, y) in enumerate(best_global_positions):
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    # Recompute S with scaled positions
    pts_scaled = np.array(pts_scaled)
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for i, (x, y) in enumerate(pts_scaled):
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
