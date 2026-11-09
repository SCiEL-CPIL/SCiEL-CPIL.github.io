# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Directly optimize the 100 log-probabilities (unconstrained real numbers) 
       using L-BFGS-B with multiple random restarts.
    3. Two-phase approach: coarse search then fine polishing.
    4. Final coordinate-wise polish for extra precision.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin (630s limit, 40s buffer)
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
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
    
    def _shannon(p_dist):
        """Shannon entropy in bits, ignoring zero probabilities."""
        p_nonzero = p_dist[p_dist > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def compute_all_entropies(p_flat):
        """
        p_flat: shape (100,) normalized probabilities
        Returns: (H_d, H_m0, H_m1, H_m2)
        """
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        H_d = _shannon(p_d)
        H_m0 = _shannon(p_m0)
        H_m1 = _shannon(p_m1)
        H_m2 = _shannon(p_m2)
        return H_d, H_m0, H_m1, H_m2

    def soft_max_ents(H_m0, H_m1, H_m2, temperature=0.1):
        """
        Compute soft max of three entropies using log-sum-exp for smoothness.
        """
        H = np.array([H_m0, H_m1, H_m2])
        H_max = np.max(H)
        if H_max < 1e-15:
            return 0.0
        scaled = (H - H_max) / temperature
        log_sum_exp = H_max / temperature + np.log(np.sum(np.exp(scaled)))
        return temperature * log_sum_exp

    def objective(log_p_params, use_soft_max=True, temperature=0.1):
        """
        log_p_params: shape (100,) log-probabilities (unnormalized)
        Returns: negative ratio -D/M (to be minimized)
        """
        if time.time() > deadline:
            return 1e9
            
        log_p_flat = log_p_params - np.max(log_p_params)
        p_flat = np.exp(log_p_flat)
        sum_p = np.sum(p_flat)
        
        if sum_p < 1e-15:
            return 1e9
            
        p_flat /= sum_p
        
        H_d, H_m0, H_m1, H_m2 = compute_all_entropies(p_flat)
        
        if use_soft_max:
            M_soft = soft_max_ents(H_m0, H_m1, H_m2, temperature)
            if M_soft < 1e-9:
                return 1e9
            return -H_d / M_soft
        else:
            M = max(H_m0, H_m1, H_m2)
            if M < 1e-9:
                return 1e9
            return -H_d / M

    def true_score(params):
        """Compute the true (non-soft) score for a parameter vector."""
        return -objective(params, use_soft_max=False)

    # Initial best: Uniform distribution
    init_params = np.zeros(100)
    best_score = true_score(init_params)
    best_params = init_params.copy()
    
    rng = np.random.default_rng(42)
    
    # Additional structured initializations based on problem geometry
    def gen_diagonal_spread():
        """Spread mass evenly across diagonals x-y, but cluster within each diagonal."""
        start = np.zeros(100)
        for d in range(-9, 10):
            # Find all (x,y) with x-y=d
            cells = [(x, y) for x in range(10) for y in range(10) if x - y == d]
            if cells:
                # Pick one cell per diagonal with some randomness
                idx = rng.integers(0, len(cells))
                x, y = cells[idx]
                start[y*10+x] += 2.0
                # Add small weight to neighbors on same diagonal
                for x2, y2 in cells:
                    if x2 != x:
                        start[y2*10+x2] += 0.3
        return start
    
    def gen_x_concentrated():
        """Concentrate in few x-values to minimize H(x)."""
        start = np.zeros(100)
        # Pick 2-3 x-values
        num_x = rng.integers(2, 4)
        chosen_x = rng.choice(10, size=num_x, replace=False)
        for x in chosen_x:
            for y in range(10):
                start[y*10+x] += 1.5 + rng.uniform(0, 0.5)
        return start
    
    def gen_xy_concentrated():
        """Concentrate in few x+y values to minimize H(x+y)."""
        start = np.zeros(100)
        num_s = rng.integers(2, 5)
        chosen_s = rng.choice(19, size=num_s, replace=False)
        for s in chosen_s:
            cells = [(x, y) for x in range(10) for y in range(10) if x + y == s]
            if cells:
                # Spread within this sum
                weights = rng.uniform(0.5, 2.0, len(cells))
                for (x, y), w in zip(cells, weights):
                    start[y*10+x] += w
        return start
    
    def gen_x2y_concentrated():
        """Concentrate in few x+2y values to minimize H(x+2y)."""
        start = np.zeros(100)
        num_t = rng.integers(2, 6)
        chosen_t = rng.choice(28, size=num_t, replace=False)
        for t in chosen_t:
            cells = [(x, y) for x in range(10) for y in range(10) if x + 2*y == t]
            if cells:
                weights = rng.uniform(0.5, 2.0, len(cells))
                for (x, y), w in zip(cells, weights):
                    start[y*10+x] += w
        return start
    
    def gen_hybrid_diagonal():
        """Hybrid: diagonal spread with x-clustering."""
        start = np.zeros(100)
        # For each diagonal, cluster around a preferred x
        preferred_x = {}
        for d in range(-9, 10):
            cells = [(x, y) for x in range(10) for y in range(10) if x - y == d]
            if cells:
                # Cluster around median x of this diagonal
                xs_on_diag = [x for x, y in cells]
                preferred_x[d] = int(np.median(xs_on_diag))
        
        for x in range(10):
            for y in range(10):
                d = x - y
                pref = preferred_x.get(d, 5)
                dist = abs(x - pref)
                start[y*10+x] = 2.0 * np.exp(-dist * 0.5) + rng.normal(0, 0.1)
        return start
    
    def run_optimization(start_params, maxiter=2000, temperature=0.1):
        try:
            res = minimize(
                lambda x: objective(x, use_soft_max=True, temperature=temperature), 
                start_params, 
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res
        except Exception:
            return None

    # Phase 1: Coarse search with diverse initializations
    def gen_gaussian(scale=1.0):
        return rng.normal(0, scale, 100)
    
    def gen_corner_biased():
        start = rng.normal(0, 0.5, 100)
        start[9] += 3.0   # (y=0, x=9)
        start[90] += 3.0  # (y=9, x=0)
        return start
    
    def gen_diagonal_biased():
        start = rng.normal(0, 0.5, 100)
        start[0] += 2.0   # (y=0, x=0)
        start[99] += 2.0  # (y=9, x=9)
        return start
    
    def gen_anti_diagonal():
        start = rng.normal(0, 0.3, 100)
        for y in range(10):
            for x in range(10):
                diff = abs(x - y)
                if diff >= 7:
                    start[y*10+x] += 1.5
        return start
    
    def gen_rank1_perturb():
        a = rng.normal(0, 1.0, 10)
        b = rng.normal(0, 1.0, 10)
        start = np.zeros(100)
        for y in range(10):
            for x in range(10):
                start[y*10+x] = a[y] + b[x]
        return start + rng.normal(0, 0.1, 100)
    
    def gen_concentrated_center():
        start = rng.normal(0, 0.3, 100)
        for y in range(3, 7):
            for x in range(3, 7):
                start[y*10+x] += 2.0
        return start
    
    def gen_edge_biased():
        start = rng.normal(0, 0.3, 100)
        # Bias toward edges of the grid
        for y in range(10):
            for x in range(10):
                edge_dist = min(x, y, 9-x, 9-y)
                if edge_dist <= 1:
                    start[y*10+x] += 1.5
        return start
    
    def gen_sparse_random():
        start = np.zeros(100)
        # Pick 15-25 random cells and give them high weight
        num_active = rng.integers(15, 25)
        active = rng.choice(100, size=num_active, replace=False)
        start[active] = rng.uniform(1.0, 3.0, num_active)
        start += rng.normal(0, 0.1, 100)
        return start
    
    def gen_bimodal():
        start = rng.normal(0, 0.3, 100)
        # Two clusters
        cx1, cy1 = rng.integers(0, 5), rng.integers(0, 5)
        cx2, cy2 = rng.integers(5, 10), rng.integers(5, 10)
        for y in range(10):
            for x in range(10):
                d1 = (x - cx1)**2 + (y - cy1)**2
                d2 = (x - cx2)**2 + (y - cy2)**2
                if d1 < 6 or d2 < 6:
                    start[y*10+x] += 2.0
        return start
    
    generators = [
        gen_gaussian, gen_corner_biased, gen_diagonal_biased,
        gen_anti_diagonal, gen_rank1_perturb, gen_concentrated_center,
        gen_edge_biased, gen_sparse_random, gen_bimodal,
        gen_diagonal_spread, gen_x_concentrated, gen_xy_concentrated,
        gen_x2y_concentrated, gen_hybrid_diagonal
    ]
    
    temperatures_coarse = [0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.35, 0.45, 0.25, 0.15, 0.4, 0.55, 0.3, 0.2]
    
    # Adaptive number of restarts based on available time
    time_per_restart = 3.0  # Estimated seconds per L-BFGS-B run
    remaining_time = deadline - time.time()
    num_coarse = min(80, max(20, int(remaining_time / time_per_restart)))
    
    for i in range(num_coarse):
        if time.time() > deadline - 60:
            break
            
        gen = generators[i % len(generators)]
        temp = temperatures_coarse[i % len(temperatures_coarse)]
        
        start = gen()
        res = run_optimization(start, maxiter=1500, temperature=temp)
        if res is not None:
            score = true_score(res.x)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        
        # Periodically check if we should stop early if time is running out
        if i % 10 == 0 and time.time() > deadline - 90:
            break

    # Phase 2: Fine polishing from best solution with adaptive temperature schedule
    if time.time() < deadline - 30:
        # Multi-stage annealing: start with higher temp for exploration, decrease for exploitation
        annealing_schedule = [
            (0.15, 5000),  # (temperature, maxiter)
            (0.08, 8000),
            (0.04, 10000),
            (0.02, 15000),
            (0.01, 20000),
        ]
        
        for temp, maxiter in annealing_schedule:
            if time.time() > deadline - 15:
                break
            res = run_optimization(best_params, maxiter=maxiter, temperature=temp)
            if res is not None:
                score = true_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            
            # Try multiple perturbations at each temperature
            num_perturb = 5 if temp > 0.05 else 8
            for _ in range(num_perturb):
                if time.time() > deadline - 15:
                    break
                # Scale perturbation with temperature
                pert_scale = 0.05 + temp * 0.5
                perturbed = best_params + rng.normal(0, pert_scale, 100)
                res2 = run_optimization(perturbed, maxiter=min(maxiter, 5000), temperature=temp)
                if res2 is not None:
                    score2 = true_score(res2.x)
                    if score2 > best_score:
                        best_score = score2
                        best_params = res2.x.copy()

    # Phase 3: High-precision polish using BFGS with very tight tolerances
    if time.time() < deadline - 10:
        # First, try BFGS with tight tolerance
        try:
            res_bfgs = minimize(
                lambda x: objective(x, use_soft_max=False), 
                best_params, 
                method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-10, 'maxfun': 100000}
            )
            if res_bfgs is not None and res_bfgs.success:
                score_bfgs = true_score(res_bfgs.x)
                if score_bfgs > best_score:
                    best_score = score_bfgs
                    best_params = res_bfgs.x.copy()
        except Exception:
            pass
        
        # Coordinate-wise polish with finer steps
        if time.time() < deadline - 5:
            for sweep in range(8):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(100):
                    if time.time() > deadline - 5:
                        break
                    best_step = 0.0
                    best_trial_score = best_score
                    # Finer step sizes for high-precision polish
                    for step in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                        trial = best_params.copy()
                        trial[i] += step
                        score = true_score(trial)
                        if score > best_trial_score:
                            best_trial_score = score
                            best_step = step
                    if best_step != 0.0:
                        best_params[i] += best_step
                        best_score = best_trial_score
                        improved = True
                if not improved:
                    break

    # Convert best parameters back to P
    log_p_flat = best_params - np.max(best_params)
    p_flat = np.exp(log_p_flat)
    p_flat /= np.sum(p_flat)
    
    P = p_flat.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
