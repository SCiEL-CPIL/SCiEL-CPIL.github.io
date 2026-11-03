import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

# Triangle vertices for unit area equilateral triangle
# Area = 1 => (sqrt(3)/4) * s^2 = 1 => s = sqrt(4/sqrt(3)) = 2 * 3^(-1/4)
S = 2.0 * (3.0 ** -0.25)
A = np.array([0.0, 0.0])
B = np.array([S, 0.0])
C = np.array([S / 2.0, S * np.sqrt(3) / 2.0])

# Precompute indices for all unique triplets
indices = list(itertools.combinations(range(N), 3))
n_triplets = len(indices)

def bary_to_cart(u_v):
    """
    Convert flattened array of 2N barycentric coords (u, v) for N points 
    to Cartesian coordinates.
    w = 1 - u - v
    P = u*A + v*B + w*C
    """
    u = u_v[0::2]
    v = u_v[1::2]
    w = 1.0 - u - v
    
    # P = u*A + v*B + w*C
    # A is (0,0), so term is 0
    # B is (S, 0)
    # C is (S/2, S*sqrt(3)/2)
    x = v * S + w * (S / 2.0)
    y = w * (S * np.sqrt(3) / 2.0)
    return np.column_stack((x, y))

def compute_min_area_soft(pts_cart, beta):
    """
    Compute the soft-min of the triangle areas.
    Returns the negative soft-min to be minimized.
    """
    n = len(pts_cart)
    areas = np.zeros(n_triplets)
    
    # Vectorized calculation of areas
    # Area = 0.5 * | (P_j - P_i) x (P_k - P_i) |
    # Using vectorization for speed
    for idx, (i, j, k) in enumerate(indices):
        pi = pts_cart[i]
        pj = pts_cart[j]
        pk = pts_cart[k]
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        areas[idx] = 0.5 * abs(cross)
        
    # Soft min: -1/beta * ln(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # log(sum(exp(-beta * a))) = -beta * max(a) + log(sum(exp(-beta * (a - max(a)))))
    # Actually, we want min, so we look at large areas? No, we want to maximize the min.
    # Let's maximize f(x) = min(a_i).
    # Minimize -f(x).
    # Soft approximation of min: -1/beta * ln(sum(exp(-beta * a_i))) is an approximation of min(a_i) for large beta?
    # Let's check: if a_1 << a_2, exp(-beta*a_1) >> exp(-beta*a_2). Sum ~ exp(-beta*a_1).
    # ln(sum) ~ -beta*a_1.
    # -1/beta * ln(sum) ~ a_1. Yes, it approximates the minimum.
    
    # So we want to MAXIMIZE this value.
    # We will MINIMIZE the negative of this value.
    
    max_a = np.max(areas)
    # Shift to prevent underflow in exp
    shifted = -beta * (areas - max_a)
    # Note: areas are positive. 
    # We want min(areas).
    # Let's use the standard log-sum-exp for stability.
    # We want to compute M = -1/beta * ln( sum( exp(-beta * a_k) ) )
    # Let a_max = max(a_k).
    # sum(exp(-beta * a_k)) = exp(-beta * a_max) * sum(exp(-beta * (a_k - a_max)))
    # ln(sum) = -beta * a_max + ln( sum(exp(-beta * (a_k - a_max))) )
    # M = a_max - 1/beta * ln( sum(exp(-beta * (a_k - a_max))) )
    
    exp_terms = np.exp(shifted) # exp(-beta * (a_k - a_max))
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        return max_a # Fallback
    log_sum = np.log(sum_exp)
    soft_min_val = max_a - (1.0 / beta) * log_sum
    
    return -soft_min_val

def objective(u_v, beta):
    pts = bary_to_cart(u_v)
    return compute_min_area_soft(pts, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_score = -np.inf
    best_u_v = None
    
    # Multi-start optimization
    # We will try a few different initializations and beta values
    
    # Initialization strategies:
    # 1. Grid-like distribution
    # 2. Random
    
    # Generate a base grid of points in barycentric coords
    # 11 points: 3 corners, 3 midpoints, 2 internal? Or 1 center, 10 around?
    # Let's just use random restarts with good bounds.
    
    num_restarts = 20
    
    rng = np.random.RandomState(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial barycentric coords
        # u, v >= 0, u+v <= 1
        # Random sampling in triangle
        u = rng.uniform(0, 1, N)
        v = rng.uniform(0, 1, N)
        # Reject if u+v > 1? 
        # Better: sample u, v in [0,1], if u+v>1, reflect or resample.
        # Or sample r1, r2 in [0,1], u = sqrt(r1), v = (1-sqrt(r1))*r2 ... standard triangle sampling
        # Let's use simple rejection for simplicity in init, or just clip.
        # Actually, let's use a structured init for first few restarts.
        
        if restart == 0:
            # Structured init: 3 corners, 3 edge midpoints, 2 internal, 1 center?
            # Corners: (1,0), (0,1), (0,0) in (u,v) where w=1-u-v
            # Let's just use a random init for all to keep code simple, 
            # but Nelder-Mead is local. 
            # A better init: 11 points in a regular hexagonal lattice pattern clipped to triangle.
            pass
            
        # Simple random init with some spread
        u_init = rng.uniform(0.1, 0.9, N)
        v_init = rng.uniform(0.1, 0.9, N)
        # Ensure u+v < 1
        mask = u_init + v_init > 0.8
        if np.any(mask):
            # Scale down
            scale = 0.8 / (u_init[mask] + v_init[mask])
            u_init[mask] *= scale
            v_init[mask] *= scale
            
        x0 = np.concatenate([u_init, v_init])
        
        # Try with a moderate beta
        beta = 100.0
        
        res = minimize(
            lambda x: objective(x, beta),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            # Evaluate true min area for this candidate
            pts = bary_to_cart(res.x)
            # Calculate true min area
            min_a = np.inf
            for i, j, k in indices:
                pi = pts[i]
                pj = pts[j]
                pk = pts[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
            
            if min_a > best_score:
                best_score = min_a
                best_u_v = res.x.copy()
                
    # If no valid result found (shouldn't happen), return random
    if best_u_v is None:
         u = np.full(N, 1.0/N)
         v = np.full(N, 1.0/N)
         best_u_v = np.concatenate([u, v])
         
    final_pts = bary_to_cart(best_u_v)
    return final_pts

