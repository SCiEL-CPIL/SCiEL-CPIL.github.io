"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

N = 50

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * sup|autoconv(h)|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution via FFT for speed
    # h is defined on [-1/4, 1/4] with N points
    # autoconv = h * h (convolution)
    # We want sup |autoconv|
    # Using FFT:
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    autoconv = np.fft.irfft(H * H, n=n_fft)[:N]  # circular conv, but we need linear
    # Actually, let's just do linear convolution properly
    autoconv = np.convolve(h, h, mode='full')
    sup_auto = np.max(np.abs(autoconv))
    if sup_auto == 0:
        return 0.0
    return l2_sq / (l1 * sup_auto)

def _score_logsumexp(h, beta=100.0):
    """Smooth surrogate using log-sum-exp for the max in denominator."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    autoconv = np.convolve(h, h, mode='full')
    # Use log-sum-exp to approximate max
    max_ac = np.max(autoconv)
    if max_ac <= 0:
        return 0.0
    # log-sum-exp trick
    shifted = autoconv - max_ac
    log_sum = max_ac + np.log(np.sum(np.exp(shifted)))
    # This approximates the max, but we want the actual max for the hard problem
    # For the smooth surrogate, we use this
    return l2_sq / (l1 * log_sum)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Strategy: 
    # 1. Start with a structured initial guess (e.g., triangular or sparse)
    # 2. Use coordinate descent / local optimization to refine
    # 3. Try multiple restarts with different initializations
    
    def time_remaining():
        return deadline - time.time()
    
    # Helper: project onto non-negative orthant
    def project_nonneg(x):
        return np.maximum(x, 0.0)
    
    # Coordinate descent step for a single index
    def coord_descent_step(h, i, lr=0.1):
        """Update h[i] to improve the score."""
        old_val = h[i]
        # Try a few step sizes
        best_new_val = old_val
        best_new_score = _score(h)
        
        for trial in [0.0, old_val * 0.5, old_val * 0.75, old_val * 1.25, old_val * 1.5, 0.1, 0.5, 1.0, 2.0]:
            if trial < 0:
                trial = 0.0
            h[i] = trial
            s = _score(h)
            if s > best_new_score:
                best_new_score = s
                best_new_val = trial
        h[i] = best_new_val
        return best_new_score
    
    # Full coordinate descent pass
    def full_coord_descent(h, max_passes=10):
        h = h.copy()
        for p in range(max_passes):
            if time_remaining() < 5.0:
                break
            improved = False
            for i in range(N):
                old_score = _score(h)
                new_score = coord_descent_step(h, i)
                if new_score > old_score + 1e-10:
                    improved = True
            if not improved:
                break
        return h
    
    # Try different initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Triangular (peaked in middle)
    tri = np.zeros(N)
    center = N // 2
    for i in range(N):
        tri[i] = 1.0 - abs(i - center) / center
    tri = np.maximum(tri, 0.0)
    initializations.append(tri)
    
    # 3. Symmetric ramp
    ramp = np.zeros(N)
    for i in range(N):
        ramp[i] = (i + 1) / N
    initializations.append(ramp)
    
    # 4. Sparse: only a few non-zero entries
    sparse1 = np.zeros(N)
    sparse1[0] = 1.0
    sparse1[N//2] = 1.0
    initializations.append(sparse1)
    
    # 5. Near-Sidon set: select indices with minimal pairwise sum collisions
    def build_sidon_like(n_points, K):
        """Select K indices from n_points with minimal pairwise sum overlaps."""
        # Greedy: pick indices one by one, minimizing collision count
        selected = []
        sum_counts = np.zeros(2 * n_points - 1, dtype=int)
        
        for _ in range(K):
            best_idx = -1
            best_collisions = float('inf')
            for i in range(n_points):
                if i in selected:
                    continue
                # Count how many existing pairs would create sum collisions
                collisions = 0
                for j in selected:
                    s = i + j
                    collisions += sum_counts[s]
                if collisions < best_collisions:
                    best_collisions = collisions
                    best_idx = i
            if best_idx >= 0:
                selected.append(best_idx)
                for j in selected:
                    if j != best_idx:
                        s = best_idx + j
                        sum_counts[s] += 1
        
        h = np.zeros(n_points)
        for idx in selected:
            h[idx] = 1.0
        return h
    
    # Try K = 5, 10, 15, 20
    for K in [5, 10, 15, 20]:
        if time_remaining() < 5.0:
            break
        initializations.append(build_sidon_like(N, K))
    
    # 6. Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-2 * x**2)
    initializations.append(gauss)
    
    # Optimize each initialization
    for init in initializations:
        if time_remaining() < 10.0:
            break
        
        h = init.copy()
        # Normalize to have reasonable scale
        if np.sum(h) > 0:
            h = h / np.sum(h) * N  # keep similar total mass
        
        # Coordinate descent
        h_opt = full_coord_descent(h, max_passes=5)
        s = _score(h_opt)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Final polish: use scipy minimize with a smooth surrogate
    if time_remaining() > 20.0:
        # Use the best_h as starting point
        def neg_score(x):
            x = project_nonneg(x)
            return -_score(x)
        
        # Try Nelder-Mead for local optimization
        try:
            result = minimize(neg_score, best_h, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                     'adaptive': True})
            if result.success or -result.fun > best_score:
                h_nm = project_nonneg(result.x)
                s_nm = _score(h_nm)
                if s_nm > best_score:
                    best_score = s_nm
                    best_h = h_nm.copy()
        except:
            pass
    
    # Ensure non-negativity and return as list
    best_h = project_nonneg(best_h)
    return list(best_h)
# EVOLVE-BLOCK-END
