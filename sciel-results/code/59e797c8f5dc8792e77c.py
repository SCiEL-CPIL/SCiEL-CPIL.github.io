"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import random
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
TIME_LIMIT = 115  # seconds, safety margin

def _bary_to_cart(u, v):
    """Convert barycentric coords (u, v) with w=1-u-v to Cartesian.
    Vertices: (0,0), (1,0), (0.5, S3H)
    x = u*1 + v*0.5
    y = u*0 + v*S3H
    """
    x = u + 0.5 * v
    y = S3H * v
    return x, y

def _min_area_raw(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a

def _objective_bary(x):
    """
    Objective function for Differential Evolution.
    x: flat array of 2*N values. Pairs (u, v) are barycentric coords.
    Returns: -min_area + penalty.
    """
    u = x[0::2]
    v = x[1::2]
    
    # Check validity: u>=0, v>=0, u+v<=1
    # Penalty for invalid points
    invalid_u = np.maximum(0, -u)
    invalid_v = np.maximum(0, -v)
    invalid_w = np.maximum(0, u + v - 1.0)
    penalty = 100.0 * (np.sum(invalid_u) + np.sum(invalid_v) + np.sum(invalid_w))
    
    if penalty > 0:
        # Return a large value if significantly outside to guide DE back
        # But we still need to compute area for gradient-like info? No, DE is derivative free.
        # Just return large penalty.
        # However, if points are far outside, area calculation might be unstable or large.
        # We can just return penalty.
        return penalty
    
    # Convert to Cartesian
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i, 0] = u[i] + 0.5 * v[i]
        pts[i, 1] = S3H * v[i]
        
    min_a = _min_area_raw(pts)
    return -min_a + penalty

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Bounds for u and v: [0, 1]
    # 2*N variables
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    best_x = None
    best_val = float('inf')
    
    # Try Differential Evolution
    # popsize=15, maxiter=100 should be fast enough.
    # We run it once. If it times out, we return best found so far?
    # DE in scipy doesn't have a callback to stop early easily without custom wrapper.
    # But 22 variables, popsize 15, maxiter 100 -> 1500 evaluations.
    # Each eval is ~165 cross products. Very fast.
    
    try:
        res = differential_evolution(
            _objective_bary, 
            bounds, 
            seed=42, 
            popsize=20, 
            maxiter=200, 
            tol=1e-8, 
            polish=True, 
            init='sobol'
        )
        
        if res.x is not None:
            # Check time
            if time.time() < start_time + TIME_LIMIT:
                # Convert result to points
                u = res.x[0::2]
                v = res.x[1::2]
                # Clamp to valid range just in case
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                # Ensure u+v <= 1
                sum_uv = u + v
                mask = sum_uv > 1.0
                if np.any(mask):
                    # Scale down
                    u[mask] /= sum_uv[mask]
                    v[mask] /= sum_uv[mask]
                
                pts = np.zeros((N, 2))
                for i in range(N):
                    pts[i, 0] = u[i] + 0.5 * v[i]
                    pts[i, 1] = S3H * v[i]
                
                # Verify
                min_a = _min_area_raw(pts)
                normalized = min_a / (S3H / 2)
                
                return pts
                
    except Exception as e:
        pass
        
    # Fallback: Random + Local Polish (original logic, simplified)
    if time.time() > start_time + TIME_LIMIT - 10:
        # Not enough time for polish, return random
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    break
            pts[i, 0] = u + 0.5 * v
            pts[i, 1] = S3H * v
        return pts

    # Try a few local polishes from random starts if DE didn't work or we want to be safe
    best_score = 0.0
    best_points = None
    
    for s in range(10):
        if time.time() > start_time + TIME_LIMIT - 5:
            break
            
        # Random init in barycentric
        u = np.random.random(N)
        v = np.random.random(N)
        # Normalize to be inside
        sum_uv = u + v
        mask = sum_uv > 1.0
        if np.any(mask):
            u[mask] /= sum_uv[mask]
            v[mask] /= sum_uv[mask]
            
        x0 = np.zeros(2*N)
        x0[0::2] = u
        x0[1::2] = v
        
        # Local polish using Nelder-Mead on barycentric coords
        # We need a cartesian objective for NM? No, we can use barycentric objective.
        # But NM needs a smooth-ish landscape. _objective_bary is non-smooth.
        # Let's stick to DE as primary. If we are here, DE failed or timed out?
        # Actually, the try block should have returned.
        # If we are here, it means DE threw an exception or we skipped it.
        
        # Simple random fallback
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                uu, vv = rng.random(), rng.random()
                if uu + vv < 1.0:
                    break
            pts[i, 0] = uu + 0.5 * vv
            pts[i, 1] = S3H * vv
            
        min_a = _min_area_raw(pts)
        normalized = min_a / (S3H / 2)
        
        if normalized > best_score:
            best_score = normalized
            best_points = pts.copy()
            
    if best_points is not None:
        return best_points
        
    # Absolute fallback
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i, 0] = 0.5
        pts[i, 1] = 0.1
    return pts

# EVOLVE-BLOCK-END
