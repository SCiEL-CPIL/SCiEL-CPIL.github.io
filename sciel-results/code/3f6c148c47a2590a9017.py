import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: hexagonal grid arrangement
    # Flat-top hexagons, side 1
    # Center-to-center distances: horizontal = sqrt(3), vertical = 1.5
    
    sqrt3 = np.sqrt(3)
    
    # Place hexagons in a compact arrangement
    # Row 0: 3 hexagons at x = -sqrt3, 0, sqrt3
    # Row 1: 4 hexagons at x = -sqrt3/2, sqrt3/2, ... offset
    # Let me think of a better initial arrangement
    
    # For 11 hexagons in hexagonal grid:
    # Center (0,0)
    # 6 neighbors at distance sqrt3
    # 4 more from the second ring
    
    # Hexagonal grid coordinates to Cartesian (flat-top):
    # (q, r) -> (sqrt3*(q + r/2), 1.5*r)
    
    def hex_to_cart(q, r):
        return (sqrt3*(q + r/2), 1.5*r)
    
    # 11 hexagons: center + 6 in first ring + 4 in second ring
    # First ring: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Second ring: pick 4 from (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    
    positions_hex = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                      (2,0), (1,-2), (0,-2), (-1,-1)]
    
    inner = []
    for q, r in positions_hex:
        x, y = hex_to_cart(q, r)
        inner.append([x, y, 0.0])
    
    # Compute initial outer hexagon size
    # Try to find minimum S
    cx, cy = 0.0, 0.0
    
    def compute_S(inner_positions, cx, cy, S):
        """Check if all inner hexagons fit in outer hexagon of side S."""
        # For each inner hexagon, check all 6 vertices
        for xi, yi, theta in inner_positions:
            theta_rad = np.radians(theta)
            for k in range(6):
                angle = theta_rad + k * np.pi / 3
                vx = xi + np.cos(angle)
                vy = yi + np.sin(angle)
                dx = vx - cx
                dy = vy - cy
                # Check 6 constraints
                if dx > S or -dx > S:
                    return False
                if dy > S*sqrt3/2 or -dy > S*sqrt3/2:
                    return False
                if sqrt3*dx + dy > S*sqrt3:
                    return False
                if -sqrt3*dx - dy > S*sqrt3:
                    return False
        return True
    
    # Binary search for minimum S
    def min_S(inner_positions, cx, cy):
        lo, hi = 0.0, 100.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if compute_S(inner_positions, cx, cy, mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    S0 = min_S(inner, cx, cy)
    
    # Optimize positions and outer hexagon center
    def objective(params):
        if time.time() > deadline:
            return 1e6
        # params: x0,y0,theta0, x1,y1,theta1, ..., cx, cy, S
        n_params = 3*N + 3
        inner_pos = []
        for i in range(N):
            x, y, theta = params[3*i:3*i+3]
            inner_pos.append([x, y, theta])
        cx, cy, S = params[-3:]
        
        # Check constraints
        if S < 1.0:
            return 1e6
        
        for xi, yi, theta in inner_pos:
            theta_rad = np.radians(theta)
            for k in range(6):
                angle = theta_rad + k * np.pi / 3
                vx = xi + np.cos(angle)
                vy = yi + np.sin(angle)
                dx = vx - cx
                dy = vy - cy
                if dx > S or -dx > S:
                    return 1e6 + abs(dx - S)
                if dy > S*sqrt3/2 or -dy > S*sqrt3/2:
                    return 1e6 + abs(dy - S*sqrt3/2)
                if sqrt3*dx + dy > S*sqrt3:
                    return 1e6 + abs(sqrt3*dx + dy - S*sqrt3)
                if -sqrt3*dx - dy > S*sqrt3:
                    return 1e6 + abs(-sqrt3*dx - dy - S*sqrt3)
        
        return S
    
    # Initial params
    params0 = []
    for x, y, theta in inner:
        params0.extend([x, y, theta])
    params0.extend([cx, cy, S0])
    
    # Optimize
    best_S = S0
    best_params = params0[:]
    
    try:
        result = minimize(objective, params0, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'maxfun': 5000})
        if result.fun < best_S:
            best_S = result.fun
            best_params = result.x[:]
    except Exception:
        pass
    
    if time.time() > deadline:
        # Return best found
        inner_final = []
        for i in range(N):
            inner_final.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        return inner_final, (best_params[-3], best_params[-2]), best_S, 0.0
    
    # Try to improve with more iterations if time allows
    if time.time() < deadline - 10:
        try:
            result2 = minimize(objective, best_params, method='L-BFGS-B',
                              options={'maxiter': 2000, 'maxfun': 10000})
            if result2.fun < best_S:
                best_S = result2.fun
                best_params = result2.x[:]
        except Exception:
            pass
    
    inner_final = []
    for i in range(N):
        inner_final.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    return inner_final, (best_params[-3], best_params[-2]), best_S, 0.0
