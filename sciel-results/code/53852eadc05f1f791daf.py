import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50
DEADLINE = 110.0 # Safety margin

def soft_min(vals, beta):
    # vals: array of radii
    # return soft min
    m = np.min(vals)
    # shift to avoid underflow/overflow
    exp_vals = np.exp(-beta * (vals - m))
    s = np.sum(exp_vals)
    return m - np.log(s) / beta

def score_and_grad(flat_c, beta):
    # flat_c: 2N array
    c = flat_c.reshape(N, 2)
    
    # Compute distances
    # Pairwise
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(c[i] - c[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Wall distances
    walls = np.zeros(N)
    for i in range(N):
        walls[i] = np.min([0.5 - c[i,0], c[i,0] + 0.5, 0.5 - c[i,1], c[i,1] + 0.5])
        
    # Combine all radii
    # We need the gradient w.r.t. flat_c
    # r_k are the radii
    # We compute weights w_k = exp(-beta * r_k)
    
    # To be efficient, we can compute the sum of exp directly in the gradient
    # But let's stick to the formula:
    # grad = sum(w_k * grad_r_k) / sum(w_k)
    
    # List of radii and their gradients
    # This is tricky to vectorize perfectly without storing all pairs.
    # N=50, pairs=1225. Walls=50. Total 1275 radii.
    
    radii = []
    grads = [] # list of 2N vectors
    
    for i in range(N):
        for j in range(i+1, N):
            d = dists[i, j]
            if d < 1e-10: d = 1e-10
            r = d / 2.0
            radii.append(r)
            # grad w.r.t c_i and c_j
            g_i = (c[i] - c[j]) / (2 * d)
            g_j = (c[j] - c[i]) / (2 * d)
            g = np.zeros(2*N)
            g[2*i:2*i+2] = g_i
            g[2*j:2*j+2] = g_j
            grads.append(g)
            
    for i in range(N):
        w = walls[i]
        radii.append(w)
        g = np.zeros(2*N)
        # Determine which wall is min
        # x: 0.5 - x, -0.5 - x? No, distance to -0.5 is x - (-0.5) = x+0.5
        # dist to 0.5 is 0.5 - x
        # dist to -0.5 is x + 0.5
        # dist to 0.5 (y) is 0.5 - y
        # dist to -0.5 (y) is y + 0.5
        dx_pos = 0.5 - c[i, 0]
        dx_neg = c[i, 0] + 0.5
        dy_pos = 0.5 - c[i, 1]
        dy_neg = c[i, 1] + 0.5
        
        if w == dx_pos: g[2*i] = -1
        elif w == dx_neg: g[2*i] = 1
        elif w == dy_pos: g[2*i+1] = -1
        elif w == dy_neg: g[2*i+1] = 1
        
        grads.append(g)
        
    radii = np.array(radii)
    grads = np.array(grads) # Shape (M, 2N)
    
    # Weights
    # To avoid overflow in exp, shift by min
    m = np.min(radii)
    weights = np.exp(-beta * (radii - m))
    sum_w = np.sum(weights)
    
    # Soft min value
    val = m - np.log(sum_w) / beta
    
    # Gradient
    # grad = (weights @ grads) / sum_w
    # Note: weights is shape (M,), grads is (M, 2N).
    # Result is (2N,)
    grad = (weights @ grads) / sum_w
    
    return -val, -grad # Minimize negative soft min

def true_score(c):
    # c: (N, 2)
    r_min = 0.5
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(c[i] - c[j]) / 2
            if d < r_min: r_min = d
    for i in range(N):
        w = np.min([0.5 - c[i,0], c[i,0] + 0.5, 0.5 - c[i,1], c[i,1] + 0.5])
        if w < r_min: r_min = w
    return r_min

def construct():
    start_time = time.time()
    best_c = None
    best_score = 0.0
    
    # Initializations
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # 2. Hexagonal
    # ...
    
    # List of starting points
    starts = [grid]
    
    # Add random starts
    for _ in range(10):
        starts.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # Add perturbed grid
    for _ in range(10):
        starts.append(grid + np.random.uniform(-0.05, 0.05, (N, 2)))
    
    beta = 100.0
    
    for c0 in starts:
        if time.time() - start_time > DEADLINE:
            break
            
        flat_c0 = c0.flatten()
        
        # L-BFGS-B
        try:
            res = minimize(
                score_and_grad, 
                flat_c0, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (2*N),
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            c_opt = res.x.reshape(N, 2)
            # Ensure bounds are respected strictly (L-BFGS-B should do it, but clip just in case)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            
            s = true_score(c_opt)
            
            if s > best_score:
                best_score = s
                best_c = c_opt
                
        except Exception:
            continue
            
    if best_c is None:
        return grid
        
    return best_c

construct()
