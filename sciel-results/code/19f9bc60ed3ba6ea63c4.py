"""Combined program: structured initializations + multi-method refinement with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return l2_sq / (l1 * hh_max)
    
    def neg_ratio_log(params):
        """Negative ratio in log-space for minimization."""
        h = np.exp(params)
        h = h / np.sum(h)
        return -compute_ratio(h)
    
    def neg_ratio_direct(h):
        """Negative ratio for direct optimization on simplex (using softmax trick)."""
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-300:
            return 0.0
        h = h / s
        return -compute_ratio(h)
    
    # Strategy: diverse initializations + multiple optimization methods
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Structured initializations - more diverse
    candidates = []
    
    # Single spike
    h1 = np.zeros(N)
    h1[N // 2] = 1.0
    candidates.append(h1)
    
    # Two equal spikes at various separations
    for sep in [1, 2, 3, 5, 7, 10, 12, 15, 20, 25]:
        h2 = np.zeros(N)
        pos1 = N // 2 - sep // 2
        pos2 = N // 2 + sep // 2
        if 0 <= pos1 < N and 0 <= pos2 < N:
            h2[pos1] = 0.5
            h2[pos2] = 0.5
            candidates.append(h2.copy())
    
    # Three equal spikes
    for sep in [5, 8, 10, 12, 15, 18]:
        h3 = np.zeros(N)
        c = N // 2
        p1, p2, p3 = c - sep, c, c + sep
        if all(0 <= p < N for p in [p1, p2, p3]):
            h3[p1] = 1.0 / 3
            h3[p2] = 1.0 / 3
            h3[p3] = 1.0 / 3
            candidates.append(h3.copy())
    
    # Many equal spikes
    for k in [4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        h4 = np.zeros(N)
        positions = np.linspace(0, N - 1, k).astype(int)
        # Ensure unique positions
        unique_pos = list(set(positions))
        if len(unique_pos) >= 2:
            h4[unique_pos] = 1.0 / len(unique_pos)
            candidates.append(h4.copy())
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Two adjacent spikes
    h6 = np.zeros(N)
    h6[N // 2 - 1] = 0.5
    h6[N // 2] = 0.5
    candidates.append(h6)
    
    # Plateaus of various widths
    for width in [4, 6, 8, 10, 12, 15, 20, 25, 30, 40]:
        h7 = np.zeros(N)
        c = N // 2
        start_idx = max(0, c - width // 2)
        end_idx = min(N, start_idx + width)
        if end_idx > start_idx:
            h7[start_idx:end_idx] = 1.0 / (end_idx - start_idx)
            candidates.append(h7.copy())
    
    # Random sparse with various sparsities
    rng = np.random.default_rng(42)
    for trial in range(20):
        k = rng.integers(1, 20)
        h_rand = np.zeros(N)
        idx = rng.choice(N, size=k, replace=False)
        weights = rng.dirichlet(np.ones(k))
        h_rand[idx] = weights
        candidates.append(h_rand)
    
    # Gaussian-like with various widths
    x = np.arange(N)
    for sigma in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        h9 = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        h9 /= np.sum(h9)
        candidates.append(h9)
    
    # Bimodal Gaussian
    for sigma in [2, 3, 4, 5, 6, 8]:
        for center_sep in [8, 10, 12, 15, 20, 25]:
            h10 = 0.5 * np.exp(-0.5 * ((x - (N/4)) / sigma)**2) + 0.5 * np.exp(-0.5 * ((x - (3*N/4)) / sigma)**2)
            h10 /= np.sum(h10)
            candidates.append(h10.copy())
    
    # Cosine bumps
    for width in [8, 10, 12, 15, 20, 25, 30]:
        h11 = np.zeros(N)
        c = N // 2
        mask = np.abs(x - c) < width
        if np.any(mask):
            h11[mask] = 0.5 * (1 + np.cos(np.pi * (x[mask] - c) / width))
            if np.sum(h11) > 0:
                h11 /= np.sum(h11)
                candidates.append(h11.copy())
    
    # Triangular bumps
    for width in [5, 8, 10, 12, 15, 20, 25]:
        h12 = np.zeros(N)
        center = N // 2
        for i in range(N):
            d = abs(i - center)
            if d < width:
                h12[i] = 1.0 - d / width
        if np.sum(h12) > 0:
            h12 /= np.sum(h12)
            candidates.append(h12.copy())
    
    # Alternating patterns
    for period in [2, 3, 4, 5]:
        h14 = np.zeros(N)
        h14[::period] = 1.0 / (N // period + 1)
        candidates.append(h14.copy())
    
    # Edge-concentrated
    for frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
        h15 = np.zeros(N)
        w = int(N * frac)
        if w > 0:
            h15[:w] = 0.5 / w
            h15[-w:] = 0.5 / w
            candidates.append(h15.copy())
    
    # Concentrated in center with decay
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        h16 = np.zeros(N)
        c = N / 2
        for i in range(N):
            d = abs(i - c)
            h16[i] = np.exp(-alpha * d / N)
        h16 /= np.sum(h16)
        candidates.append(h16.copy())
    
    # Evaluate all candidates and keep best
    for h_cand in candidates:
        val = compute_ratio(h_cand)
        if val > best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Sort candidates by ratio
    scored = [(compute_ratio(h), h) for h in candidates]
    scored.sort(key=lambda x: -x[0])
    
    # Refine top candidates with multiple methods
    n_refine = min(15, len(scored))
    for i in range(n_refine):
        if elapsed() > DEADLINE - 15:
            break
        
        h_init = scored[i][1]
        
        # Method 1: L-BFGS-B in log-space
        if elapsed() < DEADLINE - 10:
            try:
                h_pos = h_init + 1e-10
                x0 = np.log(h_pos)
                res = minimize(neg_ratio_log, x0, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 1500})
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                val = compute_ratio(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Method 2: Coordinate ascent (cyclic)
        if elapsed() < DEADLINE - 5:
            try:
                h_cur = h_init.copy()
                h_cur = h_cur / np.sum(h_cur)
                improved = True
                max_coords = 50
                coord_iter = 0
                while improved and coord_iter < max_coords and elapsed() < DEADLINE - 5:
                    improved = False
                    coord_iter += 1
                    for j in range(N):
                        old_val = h_cur[j]
                        # Try different values for h_cur[j] while keeping others fixed
                        # Optimal h_j given others: we want to maximize ratio
                        # This is a 1D optimization
                        h_test = h_cur.copy()
                        # Try a range of values
                        for mult in [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
                            h_test[j] = old_val * mult
                            h_test = np.maximum(h_test, 0)
                            s = np.sum(h_test)
                            if s < 1e-300:
                                continue
                            h_test = h_test / s
                            val_test = compute_ratio(h_test)
                            if val_test > best_val:
                                best_val = val_test
                                best_h = h_test.copy()
                        # Set to best found
                        h_cur = best_h if best_h is not h_cur else h_cur
                        if np.any(h_cur != h_init):
                            improved = True
                        h_cur = h_cur / np.sum(h_cur)
            except Exception:
                pass
    
    # Final polish from the best solution found
    if elapsed() < DEADLINE - 3:
        try:
            h_pos = best_h + 1e-10
            x0 = np.log(h_pos)
            res = minimize(neg_ratio_log, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 3000})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Additional refinement: try perturbing the best solution
    if elapsed() < DEADLINE - 2:
        try:
            rng2 = np.random.default_rng(123)
            for _ in range(20):
                h_pert = best_h.copy()
                # Random perturbation
                mask = rng2.random(N) > 0.7
                h_pert[mask] *= rng2.uniform(0.5, 2.0, size=np.sum(mask))
                h_pert = np.maximum(h_pert, 0)
                h_pert = h_pert / np.sum(h_pert)
                val = compute_ratio(h_pert)
                if val > best_val:
                    best_val = val
                    best_h = h_pert.copy()
                    # Quick L-BFGS-B refinement
                    if elapsed() < DEADLINE - 2:
                        try:
                            h_pos = best_h + 1e-10
                            x0 = np.log(h_pos)
                            res = minimize(neg_ratio_log, x0, method='L-BFGS-B',
                                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 1000})
                            h_opt = np.exp(res.x)
                            h_opt = h_opt / np.sum(h_opt)
                            val_opt = compute_ratio(h_opt)
                            if val_opt > best_val:
                                best_val = val_opt
                                best_h = h_opt.copy()
                        except Exception:
                            pass
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
