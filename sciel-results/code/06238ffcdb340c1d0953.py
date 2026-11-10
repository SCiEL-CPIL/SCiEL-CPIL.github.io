import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes D/M where D = H(x-y), M = max(H(x), H(x+y), H(x+2y)).
    Uses smooth max surrogate, log-softmax parameterization, and multi-method optimization.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # 40s safety margin from the 630s kill

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]

    # Flatten for binning
    d_val = (xs - ys).flatten()       # x - y
    m0_val = xs.flatten()              # x
    m1_val = (xs + ys).flatten()       # x + y
    m2_val = (xs + 2 * ys).flatten()   # x + 2y

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

    def log_softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    def objective_smooth(z, temperature=0.01):
        """Objective using smooth max surrogate and log-softmax parameterization"""
        p_flat = log_softmax(z)
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        
        # Smooth max via log-sum-exp for the three entropies
        H_vals = np.array([H_m0, H_m1, H_m2])
        M_smooth = temperature * np.log(np.sum(np.exp(H_vals / temperature)))
        
        if M_smooth < 1e-9:
            return 1e9
        return -H_d / M_smooth

    # --- Generate Initializers ---
    rng = np.random.default_rng(12345)
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(100) / 100.0)
    
    # 2. Structured starts
    # Diagonals (x-y = const)
    for offset in range(-9, 10):
        p = np.zeros(100)
        count = 0
        for y in range(10):
            for x in range(10):
                if x - y == offset:
                    p[y*10+x] = 1.0
                    count += 1
        if count > 0:
            p /= count
            starts.append(p)
            
    # Anti-diagonals (x+y = const)
    for offset in range(0, 20):
        p = np.zeros(100)
        count = 0
        for y in range(10):
            for x in range(10):
                if x + y == offset:
                    p[y*10+x] = 1.0
                    count += 1
        if count > 0:
            p /= count
            starts.append(p)
            
    # Lines x = const
    for x0 in range(10):
        p = np.zeros(100)
        p[x0*10:(x0+1)*10] = 1.0/10.0
        starts.append(p)
        
    # Lines y = const
    for y0 in range(10):
        p = np.zeros(100)
        for x in range(10):
            p[y0*10+x] = 1.0/10.0
        starts.append(p)
        
    # 3. Random Dirichlet starts
    for _ in range(30):
        alpha = rng.uniform(0.1, 2.0, 100)
        p = rng.dirichlet(alpha)
        starts.append(p)
        
    # 4. Gaussian blobs
    for _ in range(30):
        cy, cx = rng.uniform(1, 8, 2)
        sigma = rng.uniform(0.5, 2.0)
        yy, xx = np.mgrid[0:10, 0:10]
        z = -((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2)
        p = np.exp(z).flatten()
        p /= np.sum(p)
        starts.append(p)

    # 5. Sparse random starts (many zeros)
    for _ in range(30):
        k = rng.integers(5, 30)
        idx = rng.choice(100, size=k, replace=False)
        p = np.zeros(100)
        weights = rng.dirichlet(np.ones(k))
        p[idx] = weights
        starts.append(p)

    # --- Phase 1: Quick Evaluation and Selection ---
    best_score = -1e9
    p_best = starts[0].copy()
    
    scores = []
    for p in starts:
        s = get_score(p)
        scores.append(s)
        if s > best_score:
            best_score = s
            p_best = p.copy()
            
    # Select top 10 for further optimization
    top_indices = np.argsort(scores)[-10:]
    top_starts = [starts[i].copy() for i in top_indices]
    top_starts.append(p_best.copy())

    # --- Phase 2: Nelder-Mead on z-space (log-softmax) ---
    # Using smooth objective for better gradient info approximation
    
    def objective_z(z):
        return objective_smooth(z, temperature=0.05)

    best_z = np.log(p_best + 1e-15)
    
    for p_init in top_starts:
        if time.time() > deadline - 60.0:
            break
        z_init = np.log(p_init + 1e-15)
        try:
            res_nm = minimize(
                objective_z,
                z_init,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'fatol': 1e-10, 'xatol': 1e-12}
            )
            p_nm = log_softmax(res_nm.x)
            score_nm = get_score(p_nm)
            if score_nm > best_score:
                best_score = score_nm
                p_best = p_nm
                best_z = res_nm.x.copy()
        except Exception:
            pass

    # --- Phase 3: Simulated Annealing for local refinement ---
    # Refine the best solution found so far
    
    def simulated_annealing(p_init, max_iters=3000, T_start=0.5, T_end=0.001):
        if time.time() > deadline - 30.0:
            return p_init, get_score(p_init)
            
        p = p_init.copy()
        current_score = get_score(p)
        best_p_local = p.copy()
        best_score_local = current_score
        
        for i in range(max_iters):
            if time.time() > deadline - 30.0:
                break
                
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Propose move: move mass from one cell to another
            i_from = rng.integers(0, 100)
            i_to = rng.integers(0, 100)
            
            if i_from == i_to:
                continue
                
            # Amount to move: small fraction
            delta = 0.005 * rng.uniform(0.1, 1.0)
            
            if p[i_from] < delta:
                continue
                
            p_new = p.copy()
            p_new[i_from] -= delta
            p_new[i_to] += delta
            
            score_new = get_score(p_new)
            
            # Acceptance criterion
            delta_score = score_new - current_score
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
                p = p_new
                current_score = score_new
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_p_local = p.copy()
                    
        return best_p_local, best_score_local

    p_sa, score_sa = simulated_annealing(p_best)
    if score_sa > best_score:
        best_score = score_sa
        p_best = p_sa

    # --- Phase 4: SLSQP Final Polish on smooth objective ---
    # Use smooth max surrogate for SLSQP
    
    def objective_slsqp(z):
        return objective_smooth(z, temperature=0.01)

    bounds_z = [(-20, 20) for _ in range(100)]
    
    if time.time() < deadline - 20.0:
        try:
            res_slsqp = minimize(
                objective_slsqp,
                best_z,
                method='SLSQP',
                bounds=bounds_z,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            p_final = log_softmax(res_slsqp.x)
            score_final = get_score(p_final)
            if score_final > best_score:
                best_score = score_final
                p_best = p_final
        except Exception:
            pass

    # --- Phase 5: Final tight polish with even smaller temperature ---
    if time.time() < deadline - 10.0:
        def objective_tight(z):
            return objective_smooth(z, temperature=0.001)
        
        try:
            res_tight = minimize(
                objective_tight,
                best_z,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-14}
            )
            p_tight = log_softmax(res_tight.x)
            score_tight = get_score(p_tight)
            if score_tight > best_score:
                best_score = score_tight
                p_best = p_tight
        except Exception:
            pass

    # --- Final Validation ---
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)

    return P
