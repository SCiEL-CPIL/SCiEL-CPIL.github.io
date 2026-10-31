"""Combined program: structured initializations + L-BFGS-B refinement with deadline guard."""
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
    
    def neg_ratio(params):
        """Negative ratio in log-space for minimization. Ensures nonnegativity via exp."""
        h = np.exp(params)
        # Scale doesn't matter for the ratio (homogeneous degree 0), but keep numerically stable
        h = h / np.sum(h)
        return -compute_ratio(h)
    
    # Strategy: try several structured initializations, then refine each with L-BFGS-B
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Structured initializations
    candidates = []
    
    # 1. Single spike
    h1 = np.zeros(N)
    h1[N // 2] = 1.0
    candidates.append(h1)
    
    # 2. Two equal spikes
    h2 = np.zeros(N)
    h2[N // 4] = 0.5
    h2[3 * N // 4] = 0.5
    candidates.append(h2)
    
    # 3. Three equal spikes
    h3 = np.zeros(N)
    h3[N // 6] = 1.0 / 3
    h3[N // 2] = 1.0 / 3
    h3[5 * N // 6] = 1.0 / 3
    candidates.append(h3)
    
    # 4. Many equal spikes (10)
    h4 = np.zeros(N)
    positions = np.linspace(0, N - 1, 10).astype(int)
    h4[positions] = 1.0 / len(positions)
    candidates.append(h4)
    
    # 5. Uniform
    candidates.append(np.ones(N) / N)
    
    # 6. Two adjacent spikes
    h6 = np.zeros(N)
    h6[N // 2 - 1] = 0.5
    h6[N // 2] = 0.5
    candidates.append(h6)
    
    # 7. Wide plateau
    h7 = np.zeros(N)
    h7[N//4 : 3*N//4] = 1.0 / (N//2)
    candidates.append(h7)
    
    # 8. Random sparse
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_rand = np.zeros(N)
        k = rng.integers(1, 15)
        idx = rng.choice(N, size=k, replace=False)
        h_rand[idx] = 1.0 / k
        candidates.append(h_rand)
    
    # 9. Gaussian-like
    x = np.arange(N)
    h9 = np.exp(-0.5 * ((x - N/2) / (N/4))**2)
    h9 /= np.sum(h9)
    candidates.append(h9)
    
    # 10. Bimodal Gaussian
    h10 = 0.5 * np.exp(-0.5 * ((x - N/4) / (N/8))**2) + 0.5 * np.exp(-0.5 * ((x - 3*N/4) / (N/8))**2)
    h10 /= np.sum(h10)
    candidates.append(h10)
    
    # 11. Cosine bump (smooth, compact support-like)
    h11 = np.zeros(N)
    mask = np.abs(x - N/2) < N/3
    h11[mask] = 0.5 * (1 + np.cos(np.pi * (x[mask] - N/2) / (N/3)))
    if np.sum(h11) > 0:
        h11 /= np.sum(h11)
    candidates.append(h11)
    
    # 12. Triangular bump
    h12 = np.zeros(N)
    center = N // 2
    width = N // 3
    for i in range(N):
        d = abs(i - center)
        if d < width:
            h12[i] = 1.0 - d / width
    if np.sum(h12) > 0:
        h12 /= np.sum(h12)
    candidates.append(h12)
    
    # 13. Sparse random with larger weights
    for _ in range(5):
        h13 = np.zeros(N)
        k = rng.integers(1, 8)
        idx = rng.choice(N, size=k, replace=False)
        weights = rng.dirichlet(np.ones(k))
        h13[idx] = weights
        candidates.append(h13)
    
    # 14. Alternating sparse (every other bin)
    h14 = np.zeros(N)
    h14[::2] = 1.0 / (N // 2)
    candidates.append(h14)
    
    # 15. Edge-concentrated
    h15 = np.zeros(N)
    h15[:N//4] = 0.5 / (N//4)
    h15[3*N//4:] = 0.5 / (N//4)
    candidates.append(h15)
    
    # Evaluate all candidates and keep best
    for h_cand in candidates:
        val = compute_ratio(h_cand)
        if val > best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Refine best candidates with L-BFGS-B
    # Sort candidates by ratio and refine top few
    scored = [(compute_ratio(h), h) for h in candidates]
    scored.sort(key=lambda x: -x[0])
    
    n_refine = min(12, len(scored))
    for i in range(n_refine):
        if elapsed() > DEADLINE - 10:
            break
        
        h_init = scored[i][1]
        # Work in log-space for nonnegativity
        h_pos = h_init + 1e-10
        x0 = np.log(h_pos)
        
        try:
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': 2000})
            if elapsed() > DEADLINE - 5:
                break
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish from the best solution found
    if elapsed() < DEADLINE - 3:
        try:
            h_pos = best_h + 1e-10
            x0 = np.log(h_pos)
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxfun': 5000})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
