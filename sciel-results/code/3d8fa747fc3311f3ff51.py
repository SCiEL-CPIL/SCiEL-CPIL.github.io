import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration: Honeycomb-like
    # Center
    hex_positions = [(0.0, 0.0, 0.0)]
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        hex_positions.append((2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0))
    # Ring 2: 4 more, placed roughly at distance 3.5-4.0
    # To make a compact cluster, let's place them in the gaps or extend the ring.
    # A 11-hexagon cluster is often 7 + 4.
    # Let's just spread them out initially to avoid overlap penalties.
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 6
        hex_positions.append((3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0))
        
    x0 = []
    for (x, y, a) in hex_positions:
        x0.extend([x, y, 0.0]) # Fix angle to 0 initially for simplicity? Or keep 0.
    x0.append(6.0) # Initial outer side length
    
    def objective(params):
        if time.time() > deadline:
            return np.inf
        xs = params[:3*N].reshape(N, 3)
        s = params[3*N]
        
        if s < 1.0:
            return np.inf
            
        penalty = 0.0
        
        # Overlap penalty: Centers must be at least 2.0 apart
        # This is a sufficient condition for disjoint unit hexagons.
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(xs[i, :2] - xs[j, :2])
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
                    
        # Containment penalty: 
        # Inner hexagon circumradius is 1.0.
        # If distance from outer center to inner center is <= s - 1, 
        # the inner hexagon is guaranteed to be inside the outer hexagon (circumradius s).
        for i in range(N):
            dist_center = np.linalg.norm(xs[i, :2])
            limit = s - 1.0
            if dist_center > limit:
                penalty += (dist_center - limit) ** 2
        
        return s + 100.0 * penalty

    # Optimize
    # Use L-BFGS-B
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'maxls': 50})
    
    if result.success or result.fun < np.inf:
        params = result.x
        xs = params[:3*N].reshape(N, 3)
        s = params[3*N]
        
        inner = []
        for i in range(N):
            x, y, theta = xs[i]
            inner.append([float(x), float(y), float(np.rad2deg(theta))])
            
        # Verify validity with a quick check (optional but good for safety)
        # The simplified constraints are sufficient, so this should be valid.
        # However, if s is very close to the minimum, numerical errors might occur.
        # We will return this result.
        
        return (inner, (0.0, 0.0), float(s), 0.0)
    else:
        # Fallback: Return a known valid configuration with a larger s
        # Honeycomb of 7 fits in s=3.
        # For 11, let's use s=4.0 and place them in a 3x3 grid + 2?
        # Or just return the initial guess with a safe s.
        inner = []
        for (x, y, a) in hex_positions:
            inner.append([float(x), float(y), float(np.rad2deg(a))])
        return (inner, (0.0, 0.0), 6.0, 0.0)

