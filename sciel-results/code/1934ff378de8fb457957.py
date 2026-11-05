import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=np.float64)
    
    # Fast evaluation setup (moderate resolution for search)
    N_fast = 2048
    thetas_fast = 2 * np.pi * np.arange(N_fast) / N_fast
    z_fast = np.exp(1j * thetas_fast)
    
    powers_fast = np.ones((24, N_fast), dtype=complex)
    for k in range(1, 24):
        powers_fast[k] = powers_fast[k-1] * z_fast
    
    # Precompute delta vectors for incremental updates
    # Flipping coeff i changes P by -2*c_i*z^i
    delta_vecs = np.array([-2.0 * powers_fast[k] for k in range(24)])
    
    # High-precision setup for final evaluation
    N_high = 8192
    thetas_high = 2 * np.pi * np.arange(N_high) / N_high
    z_high = np.exp(1j * thetas_high)
    powers_high = np.ones((24, N_high), dtype=complex)
    for k in range(1, 24):
        powers_high[k] = powers_high[k-1] * z_high
    
    def compute_P(coeffs):
        return np.dot(coeffs, powers_fast)
    
    def fast_eval_from_P(P_vals):
        return np.max(np.abs(P_vals))
    
    def eval_at_theta(coeffs, theta):
        z_val = np.exp(1j * theta)
        val = 0.0 + 0.0j
        zk = 1.0 + 0.0j
        for k in range(24):
            val += coeffs[k] * zk
            zk *= z_val
        return abs(val)
        
    def golden_section_refine(coeffs, theta_lo, theta_hi, tol=1e-12, max_iter=80):
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a = theta_lo
        b = theta_hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = eval_at_theta(coeffs, c)
        fd = eval_at_theta(coeffs, d)
        
        for _ in range(max_iter):
            if b - a < tol:
                break
            if fc < fd:
                a = c
                c = d
                d = a + gr * (b - a)
                fc = fd
                fd = eval_at_theta(coeffs, d)
            else:
                b = d
                d = c
                c = b - gr * (b - a)
                fd = fc
                fc = eval_at_theta(coeffs, c)
        
        theta_mid = (a + b) / 2.0
        return eval_at_theta(coeffs, theta_mid), theta_mid
        
    def high_precision_sup_norm(coeffs):
        P_vals = np.dot(coeffs, powers_high)
        abs_vals = np.abs(P_vals)
        top_indices = np.argsort(abs_vals)[-20:]
        peak_width = 2 * np.pi / N_high
        best_refined = 0.0
        for idx in top_indices:
            theta_peak = thetas_high[idx % N_high]
            lo = theta_peak - 2 * peak_width
            hi = theta_peak + 2 * peak_width
            refined_val, _ = golden_section_refine(coeffs, lo, hi)
            if refined_val > best_refined:
                best_refined = refined_val
        return best_refined
    
    best_coeffs = best_coeffs.copy()
    P_vals = compute_P(best_coeffs)
    best_score = fast_eval_from_P(P_vals)
    
    rng = np.random.default_rng(42)
    
    def local_search(coeffs):
        coeffs = coeffs.copy()
        P_vals = compute_P(coeffs)
        current_score = fast_eval_from_P(P_vals)
        
        # 1-bit flips with incremental P update
        improved = True
        while improved:
            improved = False
            for i in range(24):
                # Delta when flipping coeff i: -2*c_i*z^i
                delta = coeffs[i] * delta_vecs[i]
                P_new = P_vals + delta
                new_score = fast_eval_from_P(P_new)
                if new_score < current_score:
                    coeffs[i] *= -1
                    P_vals = P_new
                    current_score = new_score
                    improved = True
                
                if time.time() > deadline:
                    return coeffs, current_score
        
        # 2-bit flips with incremental P update
        improved = True
        while improved:
            improved = False
            for i in range(23):
                for j in range(i + 1, 24):
                    delta = coeffs[i] * delta_vecs[i] + coeffs[j] * delta_vecs[j]
                    P_new = P_vals + delta
                    new_score = fast_eval_from_P(P_new)
                    if new_score < current_score:
                        coeffs[i] *= -1
                        coeffs[j] *= -1
                        P_vals = P_new
                        current_score = new_score
                        improved = True
                    
                    if time.time() > deadline:
                        return coeffs, current_score
        
        return coeffs, current_score

    # Run local search on the initial candidate
    best_coeffs, best_score = local_search(best_coeffs)
    
    # Try deterministic seeds
    seeds = [np.array(p[:24], dtype=np.float64)]
    for shift in range(1, 12):
        seeds.append(np.roll(p[:24], shift).astype(np.float64))
    seeds.append(-np.array(p[:24], dtype=np.float64))
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        candidate, score = local_search(seed)
        if score < best_score:
            best_score = score
            best_coeffs = candidate
    
    # Random restarts
    while time.time() < deadline - 30:
        candidate = rng.choice([-1, 1], size=24).astype(np.float64)
        candidate, score = local_search(candidate)
        if score < best_score:
            best_score = score
            best_coeffs = candidate
    
    return [int(c) for c in best_coeffs]
