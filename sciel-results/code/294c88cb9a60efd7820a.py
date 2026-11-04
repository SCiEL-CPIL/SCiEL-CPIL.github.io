import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Normals for the hexagon (flat top)
    # Angles: 0, 60, 120, 180, 240, 300
    # Cos/Sin
    angles = np.arange(6) * np.pi / 3
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    normals = np.column_stack((cos_a, sin_a)) # 6x2
    
    # Apothem of unit hexagon
    apothem_unit = np.sqrt(3) / 2
    
    # Forbidden region apothem (Minkowski sum of two unit hexagons)
    # The width of unit hex is sqrt(3).
    # The forbidden region is a hexagon with apothem sqrt(3).
    # Wait, let's re-verify.
    # Projection of unit hex on normal is [-sqrt(3)/2, sqrt(3)/2].
    # Sum of projections is [-sqrt(3), sqrt(3)].
    # So the half-width is sqrt(3).
    # So the condition for NO overlap is:
    # Exists k such that | (pi - pj) dot n_k | > sqrt(3)
    # Overlap if for ALL k, | (pi - pj) dot n_k | <= sqrt(3)
    # Violation V_ij = max_k ( sqrt(3) - | (pi - pj) dot n_k | )
    # If V_ij > 0, they overlap.
    
    forbidden_apothem = np.sqrt(3)
    
    def calculate_violations(x):
        # x is flat array of length 2*N
        # First N-1? No, L is separate or included?
        # Let's include L in the optimization variable for simplicity, or fix L and check feasibility.
        # Better: Minimize L. 
        # But standard minimizers minimize a scalar.
        # We can minimize L + penalty.
        
        # In this function, x contains L at index 0, and 2*N coords after.
        L = x[0]
        coords = x[1:].reshape(N, 2)
        
        # Boundary Violation
        # Centers must be in Hexagon of side L-1.
        # Apothem of container for centers: (L-1) * sqrt(3) / 2
        if L <= 1.0:
            return 1e6, L # Invalid
        center_apothem = (L - 1.0) * apothem_unit
        
        # Project centers onto normals
        # coords @ normals.T -> N x 6
        projections = coords @ normals.T
        # Absolute projections
        abs_projections = np.abs(projections)
        # Violation is max(0, abs_proj - center_apothem)
        b_viol = np.maximum(0, abs_projections - center_apothem)
        b_penalty = np.sum(b_viol**2)
        
        # Overlap Violation
        o_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = coords[i] - coords[j]
                # Projection of difference onto normals
                d_projs = np.abs(d @ normals.T)
                # Violation for this pair
                v = np.max(forbidden_apothem - d_projs)
                if v > 0:
                    o_penalty += v**2
        
        total_penalty = b_penalty + o_penalty
        cost = L + 10.0 * total_penalty # Weight for penalty
        
        # Check time
        if time.time() > deadline:
            # Return current best? This function doesn't store best.
            # We handle time in the main loop.
            pass
            
        return cost, L
    
    # Initialization
    # Honeycomb pattern
    # Ring 0: 1 hex? No, 12 hexes.
    # Try 2 rings: 1 center + 6 ring 1 + 5 ring 2?
    # Or 0 center + 6 ring 1 + 6 ring 2?
    # Let's try a 3x4 grid mapped to hex coords?
    # Or just a circular cluster.
    
    # Initial L guess:
    # Area of 12 unit hexes = 12 * (3*sqrt(3)/2) = 18*sqrt(3) ~ 31.1
    # Area of outer hex side L = (3*sqrt(3)/2) * L^2
    # L^2 >= 12 -> L >= sqrt(12) = 3.46
    # Packing efficiency ~ 0.9 -> L ~ 3.6
    # Let's start with L=4.0
    
    L0 = 4.0
    
    # Generate initial positions
    # 6 hexes at radius r1, 6 hexes at radius r2
    # r1 = sqrt(3) ~ 1.732 (touching center if there was one)
    # Since no center, maybe two clusters?
    # Let's use a symmetric grid.
    # Hex coords (q, r)
    # q from -2 to 2?
    # Let's just place them in a circle.
    
    np.random.seed(42)
    
    # Strategy: Run multiple random restarts
    best_L = float('inf')
    best_x = None
    
    num_restarts = 50
    steps_per_restart = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            # Symmetric init
            coords = np.zeros((N, 2))
            # 6 at radius 1.5, angles 0..300
            for i in range(6):
                ang = i * np.pi / 3
                coords[i] = [1.5 * np.cos(ang), 1.5 * np.sin(ang)]
            # 6 at radius 3.0, angles 30..330
            for i in range(6):
                ang = (i + 0.5) * np.pi / 3
                coords[i+6] = [3.0 * np.cos(ang), 3.0 * np.sin(ang)]
            x0 = np.concatenate(([4.0], coords.flatten()))
        else:
            # Random init within a hexagon of side 4
            # Simple random points
            coords = np.random.uniform(-2, 2, (N, 2))
            x0 = np.concatenate(([4.0], coords.flatten()))
            
        # Optimize
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B is better for smooth-ish functions.
        # Our penalty is smooth (squared max).
        
        res = minimize(
            lambda x: calculate_violations(x)[0],
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res.success or res.fun < best_L:
            L_val = res.x[0]
            # Verify validity
            # Check if penalties are close to 0
            cost, L_final = calculate_violations(res.x)
            # If penalty is small, it's a good candidate
            # We want to minimize L, but only if valid.
            # The cost function includes penalty.
            # If L is small but penalty is high, cost is high.
            # We track best_L based on L_final IF penalty is negligible.
            
            # Let's check the raw penalty
            b_pen, o_pen = _get_penalties(res.x)
            if b_pen < 1e-4 and o_pen < 1e-4:
                if L_final < best_L:
                    best_L = L_final
                    best_x = res.x
    
    # Final Polish
    if best_x is not None:
        # Run a high-precision polish
        res_final = minimize(
            lambda x: calculate_violations(x)[0],
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res_final.fun < best_L: # This compares cost, not L. Need to be careful.
             # Actually, if best_L was valid, cost ~ L.
             # Let's just extract L from res_final
             L_val = res_final.x[0]
             b_pen, o_pen = _get_penalties(res_final.x)
             if b_pen < 1e-6 and o_pen < 1e-6 and L_val < best_L:
                 best_L = L_val
                 best_x = res_final.x
                 
    # Format output
    if best_x is None:
        # Fallback to a known valid configuration
        # 12 hexes in a 3x4 grid?
        # Just return the initial guess if optimization failed
        coords = np.zeros((N, 2))
        for i in range(N):
             coords[i] = [i * 1.8, 0]
        return [[float(coords[i][0]), float(coords[i][1]), 0.0] for i in range(N)], (0.0, 0.0), 10.0, 0.0

    coords = best_x[1:].reshape(N, 2)
    L_out = float(best_L)
    
    inner = []
    for i in range(N):
        inner.append([float(coords[i][0]), float(coords[i][1]), 0.0])
        
    return inner, (0.0, 0.0), L_out, 0.0

def _get_penalties(x):
    # Helper to get penalties without the L cost
    L = x[0]
    coords = x[1:].reshape(N, 2)
    angles = np.arange(6) * np.pi / 3
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    apothem_unit = np.sqrt(3) / 2
    forbidden_apothem = np.sqrt(3)
    
    if L <= 1.0:
        return 1e6, 1e6
        
    center_apothem = (L - 1.0) * apothem_unit
    projections = coords @ normals.T
    abs_projections = np.abs(projections)
    b_viol = np.maximum(0, abs_projections - center_apothem)
    b_penalty = np.sum(b_viol**2)
    
    o_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = coords[i] - coords[j]
            d_projs = np.abs(d @ normals.T)
            v = np.max(forbidden_apothem - d_projs)
            if v > 0:
                o_penalty += v**2
                
    return b_penalty, o_penalty
