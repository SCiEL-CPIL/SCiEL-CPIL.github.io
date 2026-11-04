"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin before 150s kill
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon (centered at ocx, ocy, rot oangle) to contain all vertices."""
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                lx = v[0] - ocx
                ly = v[1] - ocy
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, oangle]
        Returns S + penalty.
        """
        n = N
        inner_params = params[:3 * n]
        container_params = params[3 * n:]
        ocx, ocy, oangle = container_params
        
        vertices_list = []
        for i in range(n):
            cx = inner_params[3 * i]
            cy = inner_params[3 * i + 1]
            angle = inner_params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        penalty += 100.0 * (2.0 - dist)**2
        
        return S + penalty

    def generate_initial_guess(method='lattice'):
        """Generate initial parameters for the optimizer."""
        params = np.zeros(3 * N + 3)
        
        if method == 'lattice':
            # Center
            params[0:3] = [0, 0, 0]
            
            # Ring 1 (6 hexagons)
            idx = 1
            for k in range(6):
                theta = k * np.pi / 3
                cx = np.sqrt(3) * np.cos(theta)
                cy = np.sqrt(3) * np.sin(theta)
                params[3 * idx] = cx
                params[3 * idx + 1] = cy
                params[3 * idx + 2] = 0
                idx += 1
                
            # Ring 2 (4 hexagons) - symmetric selection
            for k in range(4):
                theta = k * np.pi / 2
                cx = 2 * np.sqrt(3) * np.cos(theta)
                cy = 2 * sqrt(3) * np.sin(theta)
                params[3 * idx] = cx
                params[3 * idx + 1] = cy
                params[3 * idx + 2] = 0
                idx += 1
                
        elif method == 'circle':
            # Place hexagons in a circle
            params[0:3] = [0, 0, 0]
            idx = 1
            radius = 2.0
            for k in range(10):
                theta = k * 2 * np.pi / 10
                cx = radius * np.cos(theta)
                cy = radius * np.sin(theta)
                params[3 * idx] = cx
                params[3 * idx + 1] = cy
                params[3 * idx + 2] = 0
                idx += 1
                
        elif method == 'random':
            # Random initialization within a reasonable box
            for i in range(N):
                cx = np.random.uniform(-3, 3)
                cy = np.random.uniform(-3, 3)
                angle = np.random.uniform(0, 2 * np.pi)
                params[3 * i] = cx
                params[3 * i + 1] = cy
                params[3 * i + 2] = angle
                
        # Container initial guess
        params[3 * N] = 0.0
        params[3 * N + 1] = 0.0
        params[3 * N + 2] = 0.0
        
        return params

    # Try multiple optimization strategies
    best_params = None
    best_score = np.inf
    
    # Strategy 1: Nelder-Mead with lattice initialization
    params = generate_initial_guess('lattice')
    score = objective(params)
    if score < best_score:
        best_score = score
        best_params = params.copy()
        
    res = minimize(objective, params, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if time.time() - start_time > DEADLINE:
        # Return best found so far
        inner = []
        for i in range(N):
            cx = best_params[3 * i]
            cy = best_params[3 * i + 1]
            angle = best_params[3 * i + 2]
            inner.append([cx, cy, np.degrees(angle)])
        
        ocx = best_params[3 * N]
        ocy = best_params[3 * N + 1]
        oangle = best_params[3 * N + 2]
        S = compute_S_for_container([hex_vertices(inner[i][0], inner[i][1], np.radians(inner[i][2])) for i in range(N)], ocx, ocy, oangle)
        return inner, (ocx, ocy), S, np.degrees(oangle)
    
    if res.fun < best_score:
        best_score = res.fun
        best_params = res.x.copy()
    
    # Strategy 2: Powell with lattice initialization
    if time.time() - start_time < DEADLINE - 30:
        params = generate_initial_guess('lattice')
        res = minimize(objective, params, method='Powell', 
                      options={'maxiter': 3000, 'maxfev': 30000})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    
    # Strategy 3: Nelder-Mead with circle initialization
    if time.time() - start_time < DEADLINE - 30:
        params = generate_initial_guess('circle')
        res = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    
    # Strategy 4: Random restarts with Nelder-Mead
    if time.time() - start_time < DEADLINE - 20:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            params = generate_initial_guess('random')
            res = minimize(objective, params, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
    
    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        
    ocx = best_params[3 * N]
    ocy = best_params[3 * N + 1]
    oangle = best_params[3 * N + 2]
    S = compute_S_for_container([hex_vertices(inner[i][0], inner[i][1], np.radians(inner[i][2])) for i in range(N)], ocx, ocy, oangle)

    return inner, (ocx, ocy), S, np.degrees(oangle)
# EVOLVE-BLOCK-END
