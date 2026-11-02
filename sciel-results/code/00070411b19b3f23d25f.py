import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Precompute hexagon normals (for axis-aligned unit hexagon)
HEX_NORMALS = np.array([
    [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
    for k in range(6)
])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_container_normals(beta):
        """Get the 6 outward normals of the container hexagon."""
        angles = beta + math.pi/6 + np.arange(6) * math.pi/3
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def smooth_max(values, temperature=0.1):
        """Smooth approximation of max using log-sum-exp."""
        if len(values) == 0:
            return 0.0
        max_v = np.max(values)
        if max_v > 1000:
            return max_v
        shifted = (values - max_v) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_v / temperature
        return temperature * log_sum

    def objective(params):
        """
        params: [x1,y1,...,x11,y11, cx, cy, beta]
        Returns smooth objective: container_side_length + overlap_penalty
        """
        n_pts = N
        pts = params[:2*n_pts].reshape(n_pts, 2)
        cx = params[2*n_pts]
        cy = params[2*n_pts+1]
        beta = params[2*n_pts+2]
        
        # Normalize beta to [0, pi/6)
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        # Container normals
        c_normals = get_container_normals(beta)
        
        # Project points onto container normals
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ c_normals.T  # (n_pts, 6)
        
        # D = max over all projections and their negatives
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        D = smooth_max(all_projs, temperature=0.1)
        
        # Container side length
        S = (D + APOTHEM) / APOTHEM
        
        # Overlap penalty: for each pair, check if they overlap
        # Two unit hexagons overlap if their centers are closer than sqrt(3)
        # in the "hexagonal metric" - specifically, if there exists a normal
        # direction where |d . n| < sqrt(3)
        penalty = 0.0
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs_pair = np.abs(d @ HEX_NORMALS.T)
                min_proj = np.min(projs_pair)
                if min_proj < math.sqrt(3):
                    violation = math.sqrt(3) - min_proj
                    penalty += violation * 50.0
        
        return S + penalty

    def exact_side_length(pts, cx, cy, beta):
        """Calculate exact container side length."""
        c_normals = get_container_normals(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ c_normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        D = np.max(all_projs)
        return (D + APOTHEM) / APOTHEM

    def check_validity(pts, cx, cy, beta):
        """Check if configuration is valid (no overlaps)."""
        for i in range(N):
            for j in range(i+1, N):
                d = pts[i] - pts[j]
                projs_pair = np.abs(d @ HEX_NORMALS.T)
                min_proj = np.min(projs_pair)
                if min_proj < math.sqrt(3) - 1e-9:
                    return False
        return True

    # Generate diverse initial configurations
    def generate_initial_configurations():
        configs = []
        
        # Configuration 1: Hexagonal lattice (center + 6 neighbors + 4 more)
        axial1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                  (2,0), (-2,0), (0,2), (0,-2)]
        pts1 = np.array([[math.sqrt(3)*(q+r/2), 1.5*r] for q,r in axial1])
        configs.append(pts1.copy())
        
        # Configuration 2: 4x3 grid-like
        axial2 = [(q, r) for q in range(4) for r in range(3)]
        # Remove one to get 11
        axial2 = axial2[:11]
        pts2 = np.array([[math.sqrt(3)*(q+r/2), 1.5*r] for q,r in axial2])
        configs.append(pts2.copy())
        
        # Configuration 3: Compact cluster
        axial3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1),
                  (0,2), (1,2), (2,2), (1,3), (2,3)]
        pts3 = np.array([[math.sqrt(3)*(q+r/2), 1.5*r] for q,r in axial3])
        configs.append(pts3.copy())
        
        # Configuration 4: Elongated
        axial4 = [(q, 0) for q in range(6)] + [(q, 1) for q in range(5)]
        pts4 = np.array([[math.sqrt(3)*(q+r/2), 1.5*r] for q,r in axial4])
        configs.append(pts4.copy())
        
        # Configuration 5: Random perturbation of config 1
        pts5 = pts1 + np.random.normal(0, 0.5, size=(N, 2))
        configs.append(pts5)
        
        # Configuration 6: Another random
        pts6 = np.random.uniform(-3, 3, size=(N, 2))
        configs.append(pts6)
        
        # Configuration 7: Diamond shape
        axial7 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1),
                  (2,0), (-2,0), (1,-1), (-1,-1)]
        pts7 = np.array([[math.sqrt(3)*(q+r/2), 1.5*r] for q,r in axial7])
        configs.append(pts7.copy())
        
        # Configuration 8: 3-4-3-1 arrangement
        axial8 = [(0,0),(1,0),(2,0),
                  (0,1),(1,1),(2,1),(3,1),
                  (0,2),(1,2),(2,2),
                  (1,3)]
        pts8 = np.array([[math.sqrt(3)*(q+r/2), 1.5*r] for q,r in axial8])
        configs.append(pts8.copy())
        
        return configs

    best_S = float('inf')
    best_params = None

    initial_configs = generate_initial_configurations()
    
    # For each initial configuration, run optimization
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Center the initial configuration
        center = np.mean(init_pts, axis=0)
        init_pts_centered = init_pts - center
        
        # Initial parameters
        init_params = np.concatenate([
            init_pts_centered.ravel(),
            [0.0, 0.0, 0.0]  # cx, cy, beta
        ])
        
        # Try multiple optimization methods
        # Method 1: Nelder-Mead
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(objective, init_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    # Verify
                    n_pts = N
                    pts = res.x[:2*n_pts].reshape(n_pts, 2)
                    cx = res.x[2*n_pts]
                    cy = res.x[2*n_pts+1]
                    beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    if check_validity(pts, cx, cy, beta):
                        S_exact = exact_side_length(pts, cx, cy, beta)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = res.x.copy()
        except Exception:
            pass
        
        # Method 2: Powell
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(objective, init_params, method='Powell',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    n_pts = N
                    pts = res.x[:2*n_pts].reshape(n_pts, 2)
                    cx = res.x[2*n_pts]
                    cy = res.x[2*n_pts+1]
                    beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    if check_validity(pts, cx, cy, beta):
                        S_exact = exact_side_length(pts, cx, cy, beta)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = res.x.copy()
        except Exception:
            pass
        
        # Method 3: COBYLA
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(objective, init_params, method='COBYLA',
                              options={'maxiter': 5000, 'rhobeg': 0.1})
                if res.fun < best_S:
                    n_pts = N
                    pts = res.x[:2*n_pts].reshape(n_pts, 2)
                    cx = res.x[2*n_pts]
                    cy = res.x[2*n_pts+1]
                    beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    if check_validity(pts, cx, cy, beta):
                        S_exact = exact_side_length(pts, cx, cy, beta)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = res.x.copy()
        except Exception:
            pass
        
        # Method 4: Simulated annealing-like local search
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                current_params = init_params.copy()
                current_obj = objective(current_params)
                best_local_params = current_params.copy()
                best_local_obj = current_obj
                
                temperature = 0.5
                cooling = 0.999
                step_size = 0.3
                
                for iteration in range(20000):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    # Perturb
                    new_params = current_params + np.random.normal(0, step_size, size=current_params.shape)
                    new_obj = objective(new_params)
                    
                    # Acceptance criterion
                    if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / temperature):
                        current_params = new_params
                        current_obj = new_obj
                        
                        if current_obj < best_local_obj:
                            best_local_obj = current_obj
                            best_local_params = current_params.copy()
                    
                    temperature *= cooling
                    step_size *= 0.9995
                
                # Verify best local
                n_pts = N
                pts = best_local_params[:2*n_pts].reshape(n_pts, 2)
                cx = best_local_params[2*n_pts]
                cy = best_local_params[2*n_pts+1]
                beta = ((best_local_params[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                if check_validity(pts, cx, cy, beta):
                    S_exact = exact_side_length(pts, cx, cy, beta)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_params = best_local_params.copy()
        except Exception:
            pass

    # Final refinement with very tight tolerances
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                n_pts = N
                pts = res.x[:2*n_pts].reshape(n_pts, 2)
                cx = res.x[2*n_pts]
                cy = res.x[2*n_pts+1]
                beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                if check_validity(pts, cx, cy, beta):
                    S_exact = exact_side_length(pts, cx, cy, beta)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_params = res.x.copy()
        except Exception:
            pass

    # If no valid solution found, use fallback
    if best_params is None:
        # Use first initial configuration
        pts = initial_configs[0]
        center = np.mean(pts, axis=0)
        cx, cy = 0.0, 0.0
        beta = 0.0
        best_S = exact_side_length(pts, cx, cy, beta)
        best_params = np.concatenate([pts.ravel(), [cx, cy, beta]])

    # Extract final solution
    n_pts = N
    pts = best_params[:2*n_pts].reshape(n_pts, 2)
    cx = best_params[2*n_pts]
    cy = best_params[2*n_pts+1]
    beta = ((best_params[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
    
    # Calculate exact side length
    S_final = exact_side_length(pts, cx, cy, beta)
    
    # Build output
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(math.degrees(beta))])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation
