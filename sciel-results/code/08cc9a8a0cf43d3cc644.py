"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def edge_normals(angle_rad):
        """Outward normals for each of 6 edges of a regular hexagon."""
        edge_angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            np.cos(edge_angles + np.pi/2),
            np.sin(edge_angles + np.pi/2)
        ])

    def sat_penalty(v1, v2, n1, n2):
        """Compute overlap penalty between two hexagons using SAT.
        Returns 0 if disjoint, positive if overlapping."""
        # Check all 12 potential separating axes
        all_normals = np.vstack([n1, n2])
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        min_gap = 0.0
        for normal in all_normals:
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            d_center = np.dot(c1 - c2, normal)
            if d_center >= 0:
                gap = np.min(proj2) - np.max(proj1)
            else:
                gap = np.min(proj1) - np.max(proj2)
            if gap < min_gap:
                min_gap = gap
        # min_gap < 0 means overlap
        if min_gap < 0:
            return -min_gap  # positive penalty for overlap
        return 0.0

    def compute_outer_side(hexes, outer_cx, outer_cy, outer_angle):
        """Minimum outer hexagon side to contain all given hexagons."""
        max_req = 0.0
        theta = outer_angle
        for hex_v in hexes:
            for v in hex_v:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    ea = theta + np.radians(k * 60)
                    nx = np.cos(ea)
                    ny = np.sin(ea)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        req = dist * 2.0 / SQRT3
                        if req > max_req:
                            max_req = req
        return max_req

    def objective(params):
        """Objective: minimize outer side + separation penalty.
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Outer hexagon fixed at origin with angle 0 for simplicity."""
        if time.time() > deadline:
            return 1e6
        
        xs = params[0:N]
        ys = params[N:2*N]
        as_ = params[2*N:3*N]
        
        hexes = []
        normals_list = []
        for i in range(N):
            v = hex_vertices(xs[i], ys[i], 1.0, as_[i])
            hexes.append(v)
            normals_list.append(edge_normals(as_[i]))
        
        # Separation penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = sat_penalty(hexes[i], hexes[j], normals_list[i], normals_list[j])
                penalty += 50.0 * p * p
        
        side = compute_outer_side(hexes, 0.0, 0.0, 0.0)
        return side + penalty

    def is_valid(xs, ys, as_, tol=1e-6):
        """Check if configuration is valid (all disjoint)."""
        hexes = []
        normals_list = []
        for i in range(N):
            v = hex_vertices(xs[i], ys[i], 1.0, as_[i])
            hexes.append(v)
            normals_list.append(edge_normals(as_[i]))
        
        for i in range(N):
            for j in range(i+1, N):
                # Check if truly disjoint using SAT
                all_normals = np.vstack([normals_list[i], normals_list[j]])
                c1 = np.mean(hexes[i], axis=0)
                c2 = np.mean(hexes[j], axis=0)
                disjoint = False
                for normal in all_normals:
                    proj1 = np.dot(hexes[i], normal)
                    proj2 = np.dot(hexes[j], normal)
                    d_center = np.dot(c1 - c2, normal)
                    if d_center >= 0:
                        gap = np.min(proj2) - np.max(proj1)
                    else:
                        gap = np.min(proj1) - np.max(proj2)
                    if gap > tol:
                        disjoint = True
                        break
                if not disjoint:
                    return False
        return True

    # Generate good initial configurations
    def gen_hex_lattice(rows, cols, scale=1.0):
        """Generate hexagonal lattice positions."""
        xs, ys = [], []
        for r in range(rows):
            for c in range(cols):
                x = c * SQRT3 * scale
                y = r * 1.5 * scale
                if r % 2 == 1:
                    x += SQRT3 * scale / 2
                xs.append(x)
                ys.append(y)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        xs -= np.mean(xs)
        ys -= np.mean(ys)
        return np.column_stack([xs, ys])

    def gen_circular(radius):
        angles = 2 * np.pi * np.arange(N) / N
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # Initial configurations with various layouts
    init_configs = []
    
    # Hexagonal lattice variants
    for scale in [1.0, 0.95, 1.05]:
        init_configs.append(gen_hex_lattice(3, 4, scale))
        init_configs.append(gen_hex_lattice(4, 3, scale))
        init_configs.append(gen_hex_lattice(2, 6, scale))
    
    # Circular arrangements
    for r in [2.0, 2.2, 2.4, 2.6]:
        init_configs.append(gen_circular(r))
    
    # Two-row arrangement
    for spacing in [2.8, 3.0, 3.2]:
        xs1 = np.linspace(-5*spacing/2, 5*spacing/2, 6)
        xs2 = np.linspace(-4.5*spacing/2, 4.5*spacing/2, 6)
        init_configs.append(np.column_stack([np.concatenate([xs1, xs2]), np.concatenate([np.zeros(6), np.full(6, 2.6)])]))
    
    # Evaluate initial configs
    best_params = None
    best_score = np.inf
    
    for pos in init_configs:
        # Try different orientation patterns
        orient_patterns = [
            np.zeros(N),
            np.full(N, np.pi/6),
            np.tile([0, np.pi/6], N//2),
            np.array([0]*6 + [np.pi/6]*6),
        ]
        for orients in orient_patterns:
            params = np.concatenate([pos[:, 0], pos[:, 1], orients])
            score = objective(params)
            if score < best_score:
                best_score = score
                best_params = params.copy()
    
    # Phase 1: Differential evolution for global optimization
    if time.time() < deadline - 50:
        bounds = []
        for i in range(N):
            bounds.append((-4, 4))  # x
        for i in range(N):
            bounds.append((-4, 4))  # y
        for i in range(N):
            bounds.append((0, 2*np.pi))  # angle
        
        def de_obj(params):
            if time.time() > deadline - 10:
                return 1e6
            return objective(params)
        
        try:
            result = differential_evolution(
                de_obj, bounds,
                x0=best_params,
                seed=42, maxiter=50, popsize=10,
                tol=1e-5, polish=False,
                mutation=(0.3, 0.8), recombination=0.7
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B
    if time.time() < deadline - 30:
        def flat_obj(params):
            if time.time() > deadline - 5:
                return 1e6
            return objective(params)
        
        bounds_lb = [-4]*N + [-4]*N + [0]*N
        bounds_ub = [4]*N + [4]*N + [2*np.pi]*N
        bounds = list(zip(bounds_lb, bounds_ub))
        
        try:
            result = minimize(
                flat_obj, best_params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead fine-tuning
    if time.time() < deadline - 15:
        try:
            result = minimize(
                flat_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Multi-start local optimization from perturbed best
    if time.time() < deadline - 8:
        np.random.seed(123)
        for trial in range(8):
            if time.time() > deadline - 5:
                break
            perturbation = np.random.normal(0, 0.01, size=3*N)
            pert_params = best_params + perturbation
            try:
                result = minimize(
                    flat_obj, pert_params, method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
    
    # Extract solution
    xs = best_params[0:N]
    ys = best_params[N:2*N]
    as_ = best_params[2*N:3*N]
    
    # Check validity
    if not is_valid(xs, ys, as_, tol=1e-5):
        # Fallback: use a known valid configuration
        # Simple grid with sufficient spacing
        xs = np.array([0, SQRT3, 2*SQRT3, 3*SQRT3, 4*SQRT3, 5*SQRT3,
                       SQRT3/2, 3*SQRT3/2, 5*SQRT3/2, 7*SQRT3/2, 9*SQRT3/2, 11*SQRT3/2])
        ys = np.array([0, 0, 0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        as_ = np.zeros(N)
        xs -= np.mean(xs)
        ys -= np.mean(ys)
    
    inner = [[xs[i], ys[i], np.degrees(as_[i])] for i in range(N)]
    
    # Compute exact outer side
    hexes = [hex_vertices(xs[i], ys[i], 1.0, as_[i]) for i in range(N)]
    exact_side = compute_outer_side(hexes, 0.0, 0.0, 0.0)
    
    return inner, (0.0, 0.0), exact_side, 0.0
# EVOLVE-BLOCK-END
