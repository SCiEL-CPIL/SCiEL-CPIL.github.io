"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def hexagon_edge_normals(side, angle_rad):
        """Outward normals of a regular hexagon with given side and rotation."""
        edge_angles = angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([
            np.cos(edge_angles + np.pi/2),
            np.sin(edge_angles + np.pi/2)
        ])
        return normals

    def min_separation(cx1, cy1, a1, cx2, cy2, a2):
        """Compute minimum separation between two unit hexagons.
        Returns negative if overlapping."""
        verts1 = hexagon_vertices(cx1, cy1, 1.0, a1)
        verts2 = hexagon_vertices(cx2, cy2, 1.0, a2)
        normals1 = hexagon_edge_normals(1.0, a1)
        normals2 = hexagon_edge_normals(1.0, a2)
        center1 = np.array([cx1, cy1])
        center2 = np.array([cx2, cy2])
        min_gap = np.inf
        all_normals = np.vstack([normals1, normals2])
        for normal in all_normals:
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            d_center = np.dot(center1 - center2, normal)
            if d_center >= 0:
                gap = np.min(proj2) - np.max(proj1)
            else:
                gap = np.min(proj1) - np.max(proj2)
            if gap < min_gap:
                min_gap = gap
        return min_gap

    def compute_outer_side(vertices_list, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in vertices_list:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def objective_positions(positions, orientations, outer_angle=0.0):
        """Objective: minimize outer side length with separation penalty.
        positions: (12, 2) array of (x, y)
        orientations: (12,) array of rotation angles in radians
        """
        if time.time() > deadline:
            return 1e6
        
        inner_x = positions[:, 0]
        inner_y = positions[:, 1]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, orientations[i])
            inner_hexes.append(verts)
        
        # Separation penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = min_separation(
                    inner_x[i], inner_y[i], orientations[i],
                    inner_x[j], inner_y[j], orientations[j]
                )
                if sep < 0:
                    penalty += 100.0 * sep * sep
        
        side = compute_outer_side(inner_hexes, 0.0, 0.0, outer_angle)
        return side + penalty

    # Canonical orientations: multiples of 30 degrees
    canonical_orients = np.radians([0, 30, 60, 90, 120, 150])
    
    # Generate initial hexagonal lattice configurations with different orientation patterns
    np.random.seed(42)
    sqrt3 = np.sqrt(3)
    
    best_positions = None
    best_orientations = None
    best_score = np.inf
    
    # Try different orientation patterns
    orientation_patterns = [
        np.zeros(N),  # All aligned
        np.full(N, np.pi/6),  # All rotated 30°
        np.tile([0, np.pi/6], N//2),  # Alternating
        np.array([0]*4 + [np.pi/6]*4 + [0]*4),  # Blocks
    ]
    
    for orient_pattern in orientation_patterns:
        # Generate hexagonal lattice positions
        lattice_configs = []
        
        # 3x4 honeycomb
        for scale in [1.0, 0.98, 1.02]:
            hx, hy = [], []
            for r in range(3):
                for c in range(4):
                    x = c * sqrt3 * scale
                    y = r * 1.5 * scale
                    if r % 2 == 1:
                        x += sqrt3 * scale / 2
                    hx.append(x); hy.append(y)
            hx = np.array(hx); hy = np.array(hy)
            hx -= np.mean(hx); hy -= np.mean(hy)
            lattice_configs.append(np.column_stack([hx, hy]))
        
        # 4x3 honeycomb
        for scale in [1.0, 0.98]:
            hx, hy = [], []
            for r in range(4):
                for c in range(3):
                    x = c * sqrt3 * scale
                    y = r * 1.5 * scale
                    if r % 2 == 1:
                        x += sqrt3 * scale / 2
                    hx.append(x); hy.append(y)
            hx = np.array(hx); hy = np.array(hy)
            hx -= np.mean(hx); hy -= np.mean(hy)
            lattice_configs.append(np.column_stack([hx, hy]))
        
        # 2x6 arrangement
        for scale in [1.0, 0.98]:
            hx, hy = [], []
            for r in range(2):
                for c in range(6):
                    x = c * sqrt3 * scale
                    y = r * 1.5 * scale
                    if r % 2 == 1:
                        x += sqrt3 * scale / 2
                    hx.append(x); hy.append(y)
            hx = np.array(hx); hy = np.array(hy)
            hx -= np.mean(hx); hy -= np.mean(hy)
            lattice_configs.append(np.column_stack([hx, hy]))
        
        # Compact circular
        for radius in [2.2, 2.4, 2.6]:
            angles = 2 * np.pi * np.arange(N) / N
            hx = radius * np.cos(angles)
            hy = radius * np.sin(angles)
            lattice_configs.append(np.column_stack([hx, hy]))
        
        # Evaluate each lattice config
        for positions in lattice_configs:
            score = objective_positions(positions, orient_pattern)
            if score < best_score:
                best_score = score
                best_positions = positions.copy()
                best_orientations = orient_pattern.copy()
    
    # Phase 1: Local optimization of positions for best orientation pattern
    if time.time() < deadline - 40:
        flat_pos = best_positions.flatten()
        
        def flat_objective(flat_p):
            pos = flat_p.reshape(N, 2)
            return objective_positions(pos, best_orientations)
        
        bounds_pos = [(-4, 4)] * (2 * N)
        try:
            result = minimize(
                flat_objective, flat_pos, method='L-BFGS-B',
                bounds=bounds_pos,
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_positions = result.x.reshape(N, 2)
        except Exception:
            pass
        
        # Nelder-Mead refinement
        if time.time() < deadline - 20:
            try:
                result = minimize(
                    flat_objective, flat_pos, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_positions = result.x.reshape(N, 2)
            except Exception:
                pass
    
    # Phase 2: Try other orientation patterns with current positions
    if time.time() < deadline - 30:
        for orient_pattern in orientation_patterns:
            if np.array_equal(orient_pattern, best_orientations):
                continue
            score = objective_positions(best_positions, orient_pattern)
            if score < best_score:
                best_score = score
                best_orientations = orient_pattern.copy()
                
                # Optimize positions for this orientation
                flat_pos = best_positions.flatten()
                def flat_objective2(flat_p):
                    pos = flat_p.reshape(N, 2)
                    return objective_positions(pos, best_orientations)
                try:
                    result = minimize(
                        flat_objective2, flat_pos, method='L-BFGS-B',
                        bounds=bounds_pos,
                        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if result.fun < best_score:
                        best_score = result.fun
                        best_positions = result.x.reshape(N, 2)
                except Exception:
                    pass
    
    # Phase 3: Differential evolution on positions for best orientation
    if time.time() < deadline - 25:
        try:
            from scipy.optimize import differential_evolution
            
            def de_flat_objective(flat_p):
                if time.time() > deadline - 5:
                    return 1e6
                pos = flat_p.reshape(N, 2)
                return objective_positions(pos, best_orientations)
            
            bounds_de = [(-3, 3)] * (2 * N)
            result = differential_evolution(
                de_flat_objective, bounds_de,
                seed=42, maxiter=100, popsize=12,
                tol=1e-6, polish=True
            )
            if result.fun < best_score:
                best_score = result.fun
                best_positions = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Phase 4: Final refinement
    if time.time() < deadline - 10:
        flat_pos = best_positions.flatten()
        def final_objective(flat_p):
            pos = flat_p.reshape(N, 2)
            return objective_positions(pos, best_orientations)
        
        try:
            result = minimize(
                final_objective, flat_pos, method='L-BFGS-B',
                bounds=bounds_pos,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_positions = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # Phase 5: Multi-start local optimization
    if time.time() < deadline - 5:
        for trial in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = best_positions + np.random.normal(0, 0.02, (N, 2))
            flat_p = perturbed.flatten()
            def pert_objective(flat_p):
                pos = flat_p.reshape(N, 2)
                return objective_positions(pos, best_orientations)
            try:
                result = minimize(
                    pert_objective, flat_p, method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_positions = result.x.reshape(N, 2)
            except Exception:
                pass
    
    # Extract solution
    inner_x = best_positions[:, 0]
    inner_y = best_positions[:, 1]
    inner_a = best_orientations
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    
    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            sep = min_separation(
                inner_x[i], inner_y[i], inner_a[i],
                inner_x[j], inner_y[j], inner_a[j]
            )
            if sep < -1e-4:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: spread out hexagons
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    # Recompute exact side length
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    exact_side = compute_outer_side(inner_hexes, 0.0, 0.0, 0.0)
    
    return inner, (0.0, 0.0), exact_side, 0.0
# EVOLVE-BLOCK-END
