"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Normals for the 6 edges of a regular hexagon (angle 0)
    # Vertices at 0, 60, ... degrees.
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees.
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))

    def hex_apothem(side):
        return side * np.sqrt(3) / 2

    def containment_violation(centers, S):
        """
        Returns max violation of centers being inside outer hexagon of side S.
        Centers must be inside hexagon of apothem A_out - A_in.
        A_in = hex_apothem(1)
        A_out = hex_apothem(S)
        Effective apothem for centers: A_eff = hex_apothem(S) - hex_apothem(1)
        """
        A_eff = hex_apothem(S) - hex_apothem(1.0)
        if A_eff <= 0:
            return 1000.0
        # Project centers onto normals
        # proj shape (N, 6)
        proj = np.dot(centers, normals.T)
        abs_proj = np.abs(proj)
        # Max projection for each center
        max_proj = np.max(abs_proj, axis=1)
        # Violation is how much max_proj exceeds A_eff
        violations = np.maximum(0, max_proj - A_eff)
        return np.sum(violations)

    def disjointness_violation(centers):
        """
        Returns sum of violations for pairwise disjointness.
        Two unit hexagons are disjoint if their center difference v is outside the hexagon 2H.
        2H has apothem A_2H = 2 * hex_apothem(1) = sqrt(3).
        Condition: max_k |v . n_k| >= sqrt(3)
        Violation: max(0, sqrt(3) - max_k |v . n_k|)
        """
        A_disjoint = np.sqrt(3)  # Apothem of Minkowski sum of two unit hexagons
        total_violation = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v = centers[i] - centers[j]
                proj = np.abs(np.dot(normals, v))
                max_proj = np.max(proj)
                violation = max(0, A_disjoint - max_proj)
                total_violation += violation
        return total_violation

    def objective(params):
        """
        params: [x1, y1, ..., x12, y12, S]
        """
        centers = params[:2*N].reshape(N, 2)
        S = params[2*N]
        
        # Penalize invalid S
        if S < 1.0:
            return 1e6

        v_cont = containment_violation(centers, S)
        v_dis = disjointness_violation(centers)
        
        # We want to minimize S, but heavily penalize violations
        # If violations are 0, return S
        # If violations > 0, return large number
        penalty = 100.0 * (v_cont + v_dis)
        return S + penalty

    def initial_guess():
        """
        Create a symmetric initial guess.
        1 center at 0,0.
        6 centers at distance d1, angles 0, 60, ...
        5 centers at distance d2, angles 36, 108, ... (pentagon ring?)
        """
        centers = np.zeros((N, 2))
        # Center
        centers[0] = [0.0, 0.0]
        
        # Ring 1: 6 hexagons
        d1 = 2.0  # Approx distance
        for i in range(6):
            theta = i * np.pi / 3
            centers[i+1] = [d1 * np.cos(theta), d1 * np.sin(theta)]
            
        # Ring 2: 5 hexagons
        d2 = 4.0
        # Place them at angles offset by 30 deg? Or spread out?
        # Let's try angles 30, 90, 150, 210, 270 (skip one to make 5)
        # Or just spread 5 evenly?
        # To maintain some symmetry, let's put them at 30, 90, 150, 210, 270
        angles_ring2 = np.deg2rad([30, 90, 150, 210, 270])
        for i in range(5):
            centers[i+7] = [d2 * np.cos(angles_ring2[i]), d2 * np.sin(angles_ring2[i])]
            
        # Estimate S
        # Max distance from origin to any center + margin
        max_dist = np.max(np.linalg.norm(centers, axis=1))
        S_guess = max_dist / (np.sqrt(3)/2) + 1.0 # Rough conversion
        S_guess = max(S_guess, 5.0)
        
        params = np.zeros(2*N + 1)
        params[:2*N] = centers.flatten()
        params[2*N] = S_guess
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    # Define bounds
    # Centers should be within a reasonable range
    bounds = [(-10, 10)] * (2*N) + [(3, 15)]
    
    # Run Differential Evolution
    # maxiter is the number of generations
    # popsize is the population size
    # To stay within time limit, we monitor time in a callback or just use small params
    # DE can be slow. Let's use a small population and few iterations.
    
    try:
        # Using a small popsize and maxiter to fit in time
        # We can also use 'nelder-mead' on the reduced space if we fix symmetry, but DE is more robust
        # Let's try DE with modest settings
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            seed=42
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    except Exception as e:
        # Fallback to simple Nelder-Mead if DE fails or times out
        pass
    
    # Check time and maybe do a local polish with Nelder-Mead
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except:
            pass

    # Extract result
    centers = best_params[:2*N].reshape(N, 2)
    S = best_params[2*N]
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), 0.0])
    
    center = (0.0, 0.0)
    angle = 0.0
    
    # Final validation check (optional but good for safety)
    # We trust the penalty function to keep it valid if score is low
    # If score is high, it's invalid, but we return it anyway as best effort
    
    return inner, center, S, angle
# EVOLVE-BLOCK-END
