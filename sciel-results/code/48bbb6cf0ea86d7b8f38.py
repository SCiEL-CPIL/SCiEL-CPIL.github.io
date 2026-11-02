import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def min_enclosing_hexagon(pts, beta):
        """Find minimum enclosing regular hexagon of rotation beta around pts.
        Returns (D, cx, cy) where D is max projection."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Use LP to find center that minimizes max projection
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
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
            else:
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                D = 0
                for nx, ny in normals:
                    for px, py in pts:
                        proj = abs((px - cx) * nx + (py - cy) * ny)
                        D = max(D, proj)
        except Exception:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            D = 0
            for nx, ny in normals:
                for px, py in pts:
                    proj = abs((px - cx) * nx + (py - cy) * ny)
                    D = max(D, proj)
        
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def fitness(pts, beta):
        """Compute fitness = side length of enclosing hexagon."""
        S, _, _ = min_enclosing_hexagon(pts, beta)
        return S

    # Genetic algorithm
    # Chromosome: for each of 11 hexagons: (x, y, angle_deg)
    # We'll work in Cartesian coordinates with angles
    
    def encode(pts, angles_deg):
        """Encode positions and angles into a flat vector."""
        chrom = []
        for i in range(N):
            chrom.append(pts[i, 0])
            chrom.append(pts[i, 1])
            chrom.append(angles_deg[i])
        return np.array(chrom)

    def decode(chrom):
        """Decode flat vector back to pts and angles."""
        pts = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            pts[i, 0] = chrom[3*i]
            pts[i, 1] = chrom[3*i+1]
            angles[i] = chrom[3*i+2]
        return pts, angles

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
        ang_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = ang_rad + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def check_overlap(v1, v2, tol=1e-6):
        """Check if two hexagons overlap using SAT (Separating Axis Theorem)."""
        # Get all 12 vertices and edges
        edges = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1) % 6]
                edge = p2 - p1
                # Normal to edge
                n = np.array([-edge[1], edge[0]])
                if np.linalg.norm(n) > 1e-12:
                    n = n / np.linalg.norm(n)
                    edges.append(n)
        
        # Test each axis
        for n in edges:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Check for separation
            if max1 < min2 - tol or max2 < min1 - tol:
                return False, None
        
        # No separating axis found - they overlap
        # Find minimum translation vector
        # Project onto each axis and find the axis with minimum overlap
        min_overlap = float('inf')
        best_axis = None
        best_dir = 1.0
        
        for n in edges:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Overlap amount
            if max1 < min2 or max2 < min1:
                continue  # Already separated
            
            # Calculate overlap
            left = min(min1, min2)
            right = max(max1, max2)
            overlap = right - left
            
            # Determine direction: push apart
            if min1 < min2:
                # v1 is to the left, push v1 left (negative) and v2 right (positive)
                # But we want to separate them, so we move one in the direction of n
                if max1 <= max2:
                    # v1 is mostly to the left
                    overlap_dir = 1.0  # Move v2 in +n direction
                else:
                    overlap_dir = -1.0
            else:
                if max2 <= max1:
                    overlap_dir = -1.0
                else:
                    overlap_dir = 1.0
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = n
                best_dir = overlap_dir
        
        if best_axis is not None:
            # Return the translation needed for hexagon 2 (positive) or hexagon 1 (negative)
            # We'll return the vector to apply to hexagon 2
            return True, best_axis * best_dir * min_overlap
        else:
            return True, None

    def repair(pts, angles):
        """Repair overlapping hexagons by projecting onto Minkowski sum boundary."""
        # Simple iterative repair
        for iteration in range(20):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    v1 = hex_vertices(pts[i, 0], pts[i, 1], angles[i])
                    v2 = hex_vertices(pts[j, 0], pts[j, 1], angles[j])
                    
                    overlaps, tv = check_overlap(v1, v2)
                    if overlaps:
                        if tv is not None:
                            # Apply half the translation to each
                            pts[i] -= tv * 0.5
                            pts[j] += tv * 0.5
                            changed = True
            if not changed:
                break
        
        return pts, angles

    def evaluate(chrom):
        """Evaluate a chromosome: repair then compute fitness."""
        pts, angles = decode(chrom)
        pts, angles = repair(pts, angles)
        
        # Find best beta for this configuration
        best_S = float('inf')
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, _, _ = min_enclosing_hexagon(pts, beta)
            if S < best_S:
                best_S = S
        
        return best_S, pts, angles

    # Initial population from diverse axial configurations
    axial_configs = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (1, 3), (2, 3)],
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2),
         (0, 3), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (2, 0),
         (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (1, 3), (2, 3),
         (0, -1), (3, 1)],
    ]
    
    # Create initial population
    pop_size = 40
    population = []
    fitnesses = []
    
    for i in range(pop_size):
        if i < len(axial_configs):
            pts = axial_to_cartesian(axial_configs[i])
            angles = np.zeros(N)
            chrom = encode(pts, angles)
        else:
            # Random perturbation of a random config
            base_idx = np.random.randint(0, len(axial_configs))
            pts = axial_to_cartesian(axial_configs[base_idx])
            angles = np.zeros(N)
            # Add random noise
            noise_scale = 0.5
            pts += np.random.randn(N, 2) * noise_scale
            angles += np.random.randn(N) * 30  # degrees
            chrom = encode(pts, angles)
        
        population.append(chrom)
        S, pts, angles = evaluate(chrom)
        fitnesses.append(S)
    
    # Genetic algorithm loop
    best_S = min(fitnesses)
    best_chrom = population[np.argmin(fitnesses)]
    best_pts, best_angles = decode(best_chrom)
    best_pts, best_angles = repair(best_pts, best_angles)
    
    # Track best overall
    global_best_S = best_S
    global_best_pts = best_pts.copy()
    global_best_angles = best_angles.copy()
    
    gen = 0
    max_gens = 200
    
    while gen < max_gens:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        gen += 1
        new_population = []
        new_fitnesses = []
        
        # Elitism: keep top 10%
        elite_count = max(4, pop_size // 10)
        sorted_idx = np.argsort(fitnesses)
        for i in range(elite_count):
            idx = sorted_idx[i]
            new_population.append(population[idx].copy())
            new_fitnesses.append(fitnesses[idx])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Tournament selection
            def tournament():
                idxs = np.random.randint(0, pop_size, size=3)
                best_idx = idxs[np.argmin([fitnesses[i] for i in idxs])]
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover: uniform
            mask = np.random.randint(0, 2, size=len(parent1))
            child = np.where(mask == 1, parent1, parent2)
            
            # Mutation
            mutation_rate = 0.3
            mutation_scale = 1.0
            for i in range(len(child)):
                if np.random.random() < mutation_rate:
                    if i % 3 == 2:  # angle
                        child[i] += np.random.randn() * mutation_scale * 15
                    else:  # position
                        child[i] += np.random.randn() * mutation_scale * 0.5
            
            # Evaluate
            S, pts, angles = evaluate(child)
            new_population.append(child)
            new_fitnesses.append(S)
            
            if S < global_best_S:
                global_best_S = S
                global_best_pts = pts.copy()
                global_best_angles = angles.copy()
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Local search on best individual
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            best_idx = np.argmin(fitnesses)
            best_chrom_local = population[best_idx].copy()
            
            # Perturb and repair
            for attempt in range(20):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                perturbed = best_chrom_local.copy()
                # Random small perturbation
                n_perturb = np.random.randint(1, 4)
                for _ in range(n_perturb):
                    idx = np.random.randint(0, len(perturbed))
                    if idx % 3 == 2:
                        perturbed[idx] += np.random.randn() * 0.5
                    else:
                        perturbed[idx] += np.random.randn() * 0.2
                
                S, pts, angles = evaluate(perturbed)
                if S < fitnesses[best_idx]:
                    population[best_idx] = perturbed
                    fitnesses[best_idx] = S
                    best_chrom_local = perturbed
                    if S < global_best_S:
                        global_best_S = S
                        global_best_pts = pts.copy()
                        global_best_angles = angles.copy()
    
    # Final optimization: find best beta for global best
    best_beta = 0.0
    best_S_final = global_best_S
    for i in range(61):
        beta = i * (math.pi / 6) / 60
        S, center, D = min_enclosing_hexagon(global_best_pts, beta)
        if S < best_S_final:
            best_S_final = S
            best_beta = beta
    
    # Refine beta
    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda b: min_enclosing_hexagon(global_best_pts, b)[0], 
                             bounds=(0, math.pi / 6), method='bounded',
                             options={'maxiter': 500, 'xatol': 1e-14})
        if res.fun < best_S_final:
            best_S_final = res.fun
            best_beta = res.x
    except Exception:
        pass
    
    # Get center for final beta
    _, final_center, _ = min_enclosing_hexagon(global_best_pts, best_beta)
    
    # Apply tiny scale to ensure strict non-overlap
    SCALE = 1.0 + 1e-10
    cx, cy = final_center
    pts_scaled = []
    for x, y in global_best_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Compute final side length
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(pts_scaled[i, 0]), float(pts_scaled[i, 1]), float(global_best_angles[i])])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation
