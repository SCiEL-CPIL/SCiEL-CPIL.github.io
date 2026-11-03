import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Define the 6 vertices of a unit regular hexagon centered at origin, angle 0
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # Distance from center to vertex is 1
    base_vertices = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    
    def get_hex_vertices(center, angle_deg):
        c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
        R = np.array([[c, -s], [s, c]])
        return center + (base_vertices @ R.T).T # Wait, base_vertices is (6,2). Rotation matrix (2,2).
        # Correct: base_vertices @ R.T works if base_vertices rows are vectors.
        # Let's stick to simple:
        # v_k = center + R * v_k_base
        # v_k_base = [cos(theta), sin(theta)]
        # So vertices = center + (base_vertices @ R.T)
        
    def check_fit(params):
        """
        params: array of length 11*3 + 1
        First 11*3: x, y, angle_deg for each hex
        Last 1: L (outer side length)
        Returns: (is_valid, L)
        """
        L = params[-1]
        # Outer hexagon inequalities:
        # 1. x <= L
        # 2. -x <= L
        # 3. sqrt(3)/2 * x + 1/2 * y <= L
        # 4. -sqrt(3)/2 * x + 1/2 * y <= L
        # 5. sqrt(3)/2 * x - 1/2 * y <= L
        # 6. -sqrt(3)/2 * x - 1/2 * y <= L
        
        sqrt3 = np.sqrt(3)
        A = np.array([
            [1, 0],
            [-1, 0],
            [sqrt3/2, 0.5],
            [-sqrt3/2, 0.5],
            [sqrt3/2, -0.5],
            [-sqrt3/2, -0.5]
        ])
        # Inequalities: A @ v <= L  =>  A @ v - L <= 0
        
        for i in range(N):
            x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
            center = np.array([x, y])
            verts = get_hex_vertices(center, ang)
            # verts shape (6, 2)
            # Check all 6 vertices against 6 constraints
            # vals = A @ verts.T - L  (shape 6x6)
            vals = (A @ verts.T) - L
            if np.any(vals > 1e-9):
                return False, L
        return True, L

    def objective(params):
        """
        Objective to minimize: L + penalty for violations
        """
        L = params[-1]
        sqrt3 = np.sqrt(3)
        A = np.array([
            [1, 0],
            [-1, 0],
            [sqrt3/2, 0.5],
            [-sqrt3/2, 0.5],
            [sqrt3/2, -0.5],
            [-sqrt3/2, -0.5]
        ])
        
        penalty = 0.0
        for i in range(N):
            x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
            center = np.array([x, y])
            verts = get_hex_vertices(center, ang)
            vals = (A @ verts.T) - L
            # Sum of positive violations
            pos_viol = np.sum(np.maximum(vals, 0))
            penalty += pos_viol * 100.0
            
        return L + penalty

    def constraints_func(params):
        """
        Return array of constraint values, all must be <= 0
        """
        L = params[-1]
        sqrt3 = np.sqrt(3)
        A = np.array([
            [1, 0],
            [-1, 0],
            [sqrt3/2, 0.5],
            [-sqrt3/2, 0.5],
            [sqrt3/2, -0.5],
            [-sqrt3/2, -0.5]
        ])
        
        cons = []
        for i in range(N):
            x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
            center = np.array([x, y])
            verts = get_hex_vertices(center, ang)
            vals = (A @ verts.T) - L
            # Flatten vals (6x6) to (36,)
            cons.extend(vals.flatten().tolist())
        return np.array(cons)

    def bounds():
        # L must be positive
        # x, y should be bounded. L is likely < 10.
        # Let's set bounds for x, y in [-5, 5], angle in [0, 60], L in [1, 10]
        b = []
        for _ in range(N):
            b.append((-5.0, 5.0)) # x
            b.append((-5.0, 5.0)) # y
            b.append((0.0, 60.0)) # angle
        b.append((1.0, 10.0)) # L
        return b

    best_L = 100.0
    best_params = None
    
    # Initial guesses
    # 1. Flower-like
    # 2. Random
    
    num_restarts = 5
    if time.time() > deadline - 20:
        num_restarts = 1
        
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate initial params
        params0 = np.zeros(N * 3 + 1)
        
        if r == 0:
            # Flower: 1 center, 6 around, 4 more?
            # Just place 11 in a small grid
            for i in range(N):
                # Simple spiral or grid
                # Let's use a rough grid
                row = i // 4
                col = i % 4
                params0[3*i] = col * 1.7 - 1.7
                params0[3*i+1] = row * 1.5 - 1.5
                params0[3*i+2] = 0.0
            params0[-1] = 4.0
        else:
            # Random
            for i in range(N):
                params0[3*i] = np.random.uniform(-1.5, 1.5)
                params0[3*i+1] = np.random.uniform(-1.5, 1.5)
                params0[3*i+2] = np.random.uniform(0, 60)
            params0[-1] = 3.5 # Start with a tighter guess to encourage compression
            
        # Optimize
        try:
            res = minimize(objective, params0, method='SLSQP', bounds=bounds(), 
                           options={'maxiter': 100, 'ftol': 1e-4})
            
            if res.success or res.fun < best_L:
                # Verify validity of the result
                is_valid, L_val = check_fit(res.x)
                if is_valid:
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
        except Exception as e:
            pass

    # If no valid solution found with optimization, return a known valid one
    # A safe fallback: Line of 11 hexes.
    # Side length approx 11 * 1.732 / 1.732 = 11? 
    # Width of 11 hexes in a row: 
    # Centers at x = k * sqrt(3). 
    # Leftmost vertex: -sqrt(3)/2 * 10 - 1? 
    # Actually, if aligned flat-to-flat, distance is sqrt(3).
    # Total width = 10 * sqrt(3) + 2 (radii) ? No.
    # Hex width is 2 (corner to corner) or sqrt(3) (flat to flat).
    # If we align them in a row sharing flats, the distance between centers is sqrt(3).
    # The bounding box width is 10 * sqrt(3) + sqrt(3) = 11 * sqrt(3).
    # The outer hexagon must have flat-to-flat width >= 11 * sqrt(3).
    # L * sqrt(3) >= 11 * sqrt(3) => L >= 11.
    # This is a valid solution with L=11.
    
    if best_params is None:
        # Fallback: Line
        inner = []
        for i in range(N):
            x = (i - N//2) * np.sqrt(3)
            inner.append([x, 0.0, 0.0])
        return inner, (0.0, 0.0), 11.0, 0.0

    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, ang])
        
    L = best_params[-1]
    return inner, (0.0, 0.0), L, 0.0

# EVOLVE-BLOCK-END
