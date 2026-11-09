import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Use epigraph formulation to handle the max in M smoothly.
    2. Multi-start SLSQP with diverse starting points.
    3. Final high-precision polish pass on the best solution.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin
    
    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def shannon(p_dist):
        p_nonzero = p_dist[p_dist > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    # Epigraph formulation: minimize -H_d + t subject to t >= H_m0, t >= H_m1, t >= H_m2
    # Variables: [p (100), t (1)] = 101 variables
    def objective_epigraph(x):
        p_flat = x[:100]
        t = x[100]
        
        p_flat_clipped = np.clip(p_flat, 0, None)
        p_sum = np.sum(p_flat_clipped)
        if p_sum < 1e-9:
            return 1e9
        p_norm = p_flat_clipped / p_sum
        
        H_d, M = compute_entropies_from_p(p_norm)
        
        # Penalty if t < M
        penalty = max(0, M - t) * 1000
        
        return -H_d + t + penalty

    def constraint_sum(x):
        return np.sum(x[:100]) - 1.0

    # Generate Starting Points
    starts = []
    
    # 1. Uniform
    p_uniform = np.ones(100) / 100.0
    H_d_u, M_u = compute_entropies_from_p(p_uniform)
    starts.append(np.concatenate([p_uniform, [M_u]]))
    
    # 2. "Concentrated X" solutions - uniform on each row
    for x0 in range(10):
        p_flat = np.zeros(100)
        p_flat[x0 * 10 : (x0+1) * 10] = 1.0 / 10.0
        H_d_c, M_c = compute_entropies_from_p(p_flat)
        starts.append(np.concatenate([p_flat, [M_c]]))
        
    # 3. "Concentrated Y" solutions - uniform on each column
    for y0 in range(10):
        p_flat = np.zeros(100)
        for i in range(10):
            p_flat[i * 10 + y0] = 1.0 / 10.0
        H_d_c, M_c = compute_entropies_from_p(p_flat)
        starts.append(np.concatenate([p_flat, [M_c]]))
        
    # 4. Random Dirichlet with varied alpha
    rng = np.random.default_rng(42)
    for _ in range(30):
        alpha = rng.uniform(0.1, 3.0, 100)
        p_flat = rng.dirichlet(alpha)
        H_d_r, M_r = compute_entropies_from_p(p_flat)
        starts.append(np.concatenate([p_flat, [M_r]]))
        
    # 5. "Diagonal" patterns
    p_diag = np.zeros(100)
    for i in range(10):
        p_diag[i*10+i] = 1.0/10.0
    H_d_d, M_d = compute_entropies_from_p(p_diag)
    starts.append(np.concatenate([p_diag, [M_d]]))
    
    # 6. "Anti-diagonal" patterns
    p_adiag = np.zeros(100)
    for i in range(10):
        p_adiag[i*10+(9-i)] = 1.0/10.0
    H_d_a, M_a = compute_entropies_from_p(p_adiag)
    starts.append(np.concatenate([p_adiag, [M_a]]))

    # 7. Sparse patterns - concentrate on few cells
    for trial in range(15):
        k = rng.integers(3, 8)  # 3 to 7 cells
        idx = rng.choice(100, size=k, replace=False)
        p_flat = np.zeros(100)
        weights = rng.dirichlet(np.ones(k) * 2.0)
        p_flat[idx] = weights
        H_d_s, M_s = compute_entropies_from_p(p_flat)
        starts.append(np.concatenate([p_flat, [M_s]]))

    best_score = -1e9
    p_best = None
    
    # Optimize from each start
    for i, start_x in enumerate(starts):
        if time.time() > deadline - 30:
            break
            
        bounds = [(0.0, 1.0) for _ in range(100)] + [(0.0, 10.0)]
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        
        try:
            res = minimize(
                objective_epigraph, 
                start_x, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            p_res = res.x[:100]
            p_res = np.clip(p_res, 0, None)
            p_sum = np.sum(p_res)
            if p_sum > 1e-9:
                p_res = p_res / p_sum
            
            score = get_score(p_res)
            
            if score > best_score:
                best_score = score
                p_best = p_res
                
        except Exception as e:
            pass

    # Final high-precision polish on the best solution
    if p_best is not None and time.time() < deadline - 10:
        H_d_b, M_b = compute_entropies_from_p(p_best)
        x_init = np.concatenate([p_best, [M_b]])
        
        bounds = [(0.0, 1.0) for _ in range(100)] + [(0.0, 10.0)]
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        
        try:
            res = minimize(
                objective_epigraph, 
                x_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            
            p_res = res.x[:100]
            p_res = np.clip(p_res, 0, None)
            p_sum = np.sum(p_res)
            if p_sum > 1e-9:
                p_res = p_res / p_sum
            
            score = get_score(p_res)
            
            if score > best_score:
                best_score = score
                p_best = p_res
                
        except Exception as e:
            pass

    # Final validation and normalization
    if p_best is None:
        P = np.ones((n, n)) / (n * n)
    else:
        P = p_best.reshape((n, n))
        P = np.clip(P, 0.0, None)
        s = np.sum(P)
        if s > 1e-9:
            P = P / s
        else:
            P = np.ones((n, n)) / (n * n)
        
    return P
