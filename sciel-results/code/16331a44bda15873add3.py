"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    N = 50
    
    # Ring structure: 1 center + rings of 6, 12, 12, 19
    ring_sizes = [1, 6, 12, 12, 19]
    n_rings = len(ring_sizes) - 1  # 4 rings
    
    def get_centers(params):
        """Generate centers from parameters: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]"""
        centers = [[0.0, 0.0]]
        for i in range(1, n_rings + 1):
            r = params[i-1]
            theta_offset = params[n_rings + i - 1]
            n = ring_sizes[i]
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + theta_offset
            for a in angles:
                centers.append([r * np.cos(a), r * np.sin(a)])
        return np.array(centers)
    
    def compute_radius(centers):
        """Compute the packing radius from centers."""
        if len(centers) != N:
            return -1.0
        dists = np.hypot(centers[:, 0], centers[:, 1])
        if np.any(dists > 1.0):
            return -1.0
        # Compute minimum pairwise distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < min_dist:
                    min_dist = d
        r_pair = min_dist / 2
        r_boundary = 1.0 - np.max(dists)
        return min(r_pair, r_boundary)
    
    def objective_exact(params):
        """Exact nonsmooth objective: negative packing radius."""
        centers = get_centers(params)
        r = compute_radius(centers)
        return -r
    
    def objective_smooth(params, beta=100.0):
        """Smooth approximation using soft-min of the two constraints."""
        centers = get_centers(params)
        dists = np.hypot(centers[:, 0], centers[:, 1])
        if np.any(dists > 1.0):
            return 1e6
        
        # Compute all pairwise distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < min_dist:
                    min_dist = d
        
        r_pair = min_dist / 2
        r_boundary = 1.0 - np.max(dists)
        
        # Soft-min: smooth approximation of min(r_pair, r_boundary)
        if r_pair < 0 or r_boundary < 0:
            return 1e6
        m = min(r_pair, r_boundary)
        if m < 1e-10:
            return -m
        soft_min_val = m - np.log(np.exp(beta*(r_pair - m)) + np.exp(beta*(r_boundary - m))) / beta
        return -soft_min_val
    
    # Better initial guess based on known packing geometry
    # For 50 circles with r≈0.1258, ring radii should be approximately:
    initial_params = np.array([0.252, 0.504, 0.756, 0.944, 
                               0.0, np.pi/6, 0.0, np.pi/19])
    
    best_result = initial_params.copy()
    best_score = objective_exact(best_result)
    
    # Phase 1: Multiple restarts with smooth objective
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > deadline - 30:
            break
            
        if trial == 0:
            params0 = initial_params.copy()
        elif trial == 1:
            params0 = initial_params + np.array([0.01, -0.01, 0.01, -0.01, 
                                                  np.pi/12, 0, np.pi/24, 0])
        elif trial == 2:
            params0 = initial_params + np.array([-0.01, 0.01, -0.01, 0.01, 
                                                  -np.pi/12, 0, -np.pi/24, 0])
        else:
            params0 = initial_params + np.random.randn(8) * 0.02
        
        try:
            # Use different beta values for diversity
            beta = 50.0 + trial * 10.0
            res = minimize(lambda p: objective_smooth(p, beta=beta), params0, 
                          method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
            
            # Evaluate with exact objective
            exact_score = objective_exact(res.x)
            if exact_score > best_score:
                best_score = exact_score
                best_result = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish with exact objective and tighter tolerances
    if time.time() < deadline - 15:
        try:
            res_polish = minimize(objective_exact, best_result, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15})
            if res_polish.fun < -best_score:
                best_score = -res_polish.fun
                best_result = res_polish.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional local refinement with smaller steps
    if time.time() < deadline - 5:
        try:
            for _ in range(5):
                res_refine = minimize(objective_exact, best_result, method='Nelder-Mead',
                                      options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16,
                                               'initial_simplex': None})
                if res_refine.fun < -best_score:
                    best_score = -res_refine.fun
                    best_result = res_refine.x.copy()
        except Exception:
            pass
    
    centers = get_centers(best_result)
    
    # Final validation
    dists = np.hypot(centers[:, 0], centers[:, 1])
    if np.any(dists > 1.0) or len(centers) != N:
        # Fallback to hexagonal packing
        spacing = 0.25
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return centers
# EVOLVE-BLOCK-END
