import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2
DEADLINE = 110.0 # Safety margin for 120s limit

def get_pairs(n):
    return list(itertools.combinations(range(n), 2))

PAIRS = get_pairs(N)
NUM_PAIRS = len(PAIRS)

def compute_distances(P):
    # P is (N, D)
    # Returns array of squared distances for all pairs
    # Shape: (NUM_PAIRS,)
    dists = np.zeros(NUM_PAIRS)
    for k, (i, j) in enumerate(PAIRS):
        diff = P[i] - P[j]
        dists[k] = np.dot(diff, diff)
    return dists

def soft_max_min(dists, beta):
    # dists: (NUM_PAIRS,)
    # Returns M, m, S_max, S_min, weights_w, weights_v
    # To avoid overflow, shift by max/min
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    # Soft Max
    # M = (1/beta) * log(sum(exp(beta * (d - max_d)))) + max_d
    shifted_max = beta * (dists - max_d)
    exp_max = np.exp(shifted_max)
    S_max = np.sum(exp_max)
    M = max_d + (1.0 / beta) * np.log(S_max)
    w = exp_max / S_max # Normalized weights for gradient
    
    # Soft Min
    # m = -(1/beta) * log(sum(exp(-beta * (d - min_d)))) + min_d
    shifted_min = -beta * (dists - min_d)
    exp_min = np.exp(shifted_min)
    S_min = np.sum(exp_min)
    m = min_d - (1.0 / beta) * np.log(S_min)
    v = exp_min / S_min # Normalized weights for gradient
    
    return M, m, w, v

def objective_and_grad(P_flat, beta):
    P = P_flat.reshape(N, D)
    dists = compute_distances(P)
    
    # Handle case where min dist is 0 (points coincide)
    if np.min(dists) < 1e-12:
        # Add penalty for coincidence
        penalty = 1e6 * np.sum(dists[dists < 1e-12])
        return 1e6 + penalty, np.zeros_like(P_flat)
        
    M, m, w, v = soft_max_min(dists, beta)
    
    if m < 1e-12:
        return 1e6, np.zeros_like(P_flat)

    f = M / m
    
    # Gradient calculation
    grad_P = np.zeros_like(P)
    
    # Precompute contributions for each pair
    # grad_M_i = sum_{k involving i} w_k * d(d_k^2)/d(p_i)
    # grad_m_i = sum_{k involving i} v_k * d(d_k^2)/d(p_i)
    
    for k, (i, j) in enumerate(PAIRS):
        diff = P[i] - P[j]
        # d(d_k^2)/d(p_i) = 2 * diff
        # d(d_k^2)/d(p_j) = -2 * diff
        
        grad_M_i = w[k] * 2.0 * diff
        grad_M_j = -w[k] * 2.0 * diff
        
        grad_m_i = v[k] * 2.0 * diff
        grad_m_j = -v[k] * 2.0 * diff
        
        grad_P[i] += grad_M_i
        grad_P[j] += grad_M_j
        
        # Store for second term? No, we need sum of grad_m too.
        # Let's accumulate grad_M and grad_m separately first.
        # This loop is O(N^2) which is fine.
        
    # Wait, the above loop accumulates grad_M into grad_P directly? No, I mixed them.
    # Let's redo gradient accumulation properly.
    
    grad_M = np.zeros_like(P)
    grad_m = np.zeros_like(P)
    
    for k, (i, j) in enumerate(PAIRS):
        diff = P[i] - P[j]
        d_d2_di = 2.0 * diff
        d_d2_dj = -2.0 * diff
        
        grad_M[i] += w[k] * d_d2_di
        grad_M[j] += w[k] * d_d2_dj
        
        grad_m[i] += v[k] * d_d2_di
        grad_m[j] += v[k] * d_d2_dj
        
    # f = M / m
    # df = (m * grad_M - M * grad_m) / m^2
    grad_f = (m * grad_M - M * grad_m) / (m * m)
    
    return f, grad_f.flatten()

def hexagonal_init():
    # Generate hexagonal lattice points
    points = []
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    # Center
    points -= np.mean(points, axis=0)
    # Sort by distance to origin
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)
    # Take 16 closest
    P = points[indices[:N]]
    # Scale to reasonable size
    P *= 1.0
    return P

def grid_init():
    side = int(np.ceil(N ** 0.5))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    P = np.array(grid, dtype=float)
    P -= np.mean(P, axis=0)
    return P

def random_init():
    return np.random.randn(N, D) * 1.0

def construct():
    start_time = time.time()
    best_P = None
    best_score = float('inf')
    
    # Initializations
    inits = [
        hexagonal_init(),
        grid_init(),
        random_init(),
        random_init(),
        random_init()
    ]
    
    beta = 20.0
    
    for P_init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        P_flat = P_init.flatten()
        
        try:
            res = minimize(
                objective_and_grad, 
                P_flat, 
                args=(beta,), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            P_opt = res.x.reshape(N, D)
            
            # Calculate true score
            dists = compute_distances(P_opt)
            d_min = np.sqrt(np.min(dists))
            d_max = np.sqrt(np.max(dists))
            
            if d_min < 1e-9:
                score = 1e6
            else:
                score = (d_max / d_min) ** 2
            
            if score < best_score:
                best_score = score
                best_P = P_opt.copy()
                
        except Exception as e:
            pass
            
    if best_P is None:
        return hexagonal_init()
        
    return best_P

if __name__ == "__main__":
    P = construct()
    dists = compute_distances(P)
    d_min = np.sqrt(np.min(dists))
    d_max = np.sqrt(np.max(dists))
    print(f"Score: {(d_max / d_min)**2}")
