"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import scipy.optimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint (circumradius of unit hexagon is 1)
        if dist_sq > 4.0 + 1e-9:
            return True
        
        # Precise SAT check
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Precompute edge normals for both hexagons
        normals = []
        for i in range(6):
            e1 = v1[i] - v1[(i+1) % 6]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-12:
                normals.append(n1 / norm1)
            
            e2 = v2[i] - v2[(i+1) % 6]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-12:
                normals.append(n2 / norm2)
        
        for normal in normals:
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) - 1e-10 or np.max(proj2) < np.min(proj1) - 1e-10:
                return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Compute minimum outer side length required to contain all inner hexagons."""
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon (pointing outward)
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / SQRT3

    def initial_guess():
        """Create a valid initial configuration using hexagonal packing."""
        params = np.zeros(3*N + 3)
        positions = []
        
        # Use proper hexagonal lattice positions
        # Axial coordinates (q, r) -> Cartesian: x = 3/2*q, y = sqrt(3)/2*(q + 2*r)
        # For unit hexagons, adjacent centers are distance 2 apart
        
        # Center
        positions.append((0.0, 0.0, 0.0))
        
        # Ring 1: 6 hexagons around center
        ring1_axial = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        for q, r in ring1_axial:
            x = 1.5 * q
            y = SQRT3/2 * (q + 2*r)
            positions.append((x, y, 0.0))
        
        # Ring 2: Pick 5 well-separated positions
        # Ring 2 axial coords include (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1), (1,0)
        # But (1,0) is already used in ring 1, so pick from the rest
        # Pick corners and some edge midpoints for good spacing
        ring2_candidates = [(2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1)]
        
        # Pick 5 that are well-separated
        # Use a simple heuristic: pick every other one
        selected = []
        for i in range(0, len(ring2_candidates), 2):
            if len(selected) >= 5:
                break
            selected.append(ring2_candidates[i])
        
        # If we didn't get 5, add more
        while len(selected) < 5:
            for q, r in ring2_candidates:
                if (q, r) not in selected:
                    selected.append((q, r))
                    if len(selected) >= 5:
                        break
        
        for q, r in selected:
            x = 1.5 * q
            y = SQRT3/2 * (q + 2*r)
            positions.append((x, y, 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        side = compute_min_side_fast(inner, center, angle)
        return side

    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = best_score
    
    # Simulated annealing with adaptive step size
    T = 1.5
    T_min = 0.0005
    cooling_rate = 0.9995
    base_step = 0.25
    max_steps = 50000
    
    step_count = 0
    no_improve_count = 0
    last_improve_step = 0
    
    while time.time() < deadline - 30.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Adaptive step size: larger when T is high, smaller when T is low
        # Also reduce step size if no improvement for a while
        if step_count - last_improve_step > 5000:
            base_step *= 0.9
        current_step = base_step * (0.3 + 0.7 * T / 1.5)
        
        # Perturb 1-3 random parameters
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, 3*N + 2)
            new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, T_min)):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
                no_improve_count = 0
                last_improve_step = step_count
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Polish with Nelder-Mead using smooth objective
    def smooth_penalty(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
        return penalty

    def polish_objective(params):
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        side = compute_min_side_fast(inner, center, angle)
        pen = smooth_penalty(params)
        return side + 50.0 * pen

    # Multiple polish attempts with different initial perturbations
    for attempt in range(3):
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Slightly perturb the best solution for diversity
            if attempt == 0:
                start_p = best_params.copy()
            else:
                start_p = best_params.copy()
                for idx in random.sample(range(3*N+3), 6):
                    start_p[idx] += random.gauss(0, 0.05)
            
            result = scipy.optimize.minimize(
                polish_objective, 
                start_p, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 10000}
            )
            
            p = result.x
            inner = [(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not hexagons_disjoint_fast(inner[i], inner[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                new_side = compute_min_side_fast(inner, (p[3*N], p[3*N+1]), p[3*N+2])
                if new_side < best_score:
                    best_score = new_side
                    best_params = p
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
