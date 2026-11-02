"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_score(heights):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k] for k in range
    # Using FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:N]
    # The autoconvolution at lag 0 is sum h[i]^2, but we need the full circular/linear conv
    # Actually for linear autoconvolution:
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _negative_log_ratio(params):
    """Objective: maximize log(ratio) via minimizing -log(ratio).
    params are log-heights to ensure positivity."""
    h = np.exp(params)
    # Normalize to avoid scale issues (ratio is scale-invariant)
    h = h / np.sum(h)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    ratio = l2_sq / (l1 * sup_norm)
    if ratio <= 0:
        return 0.0
    return -np.log(ratio)

def construct():
    """Construct 50 nonnegative heights to maximize the ratio.
    
    Strategy: Start from a Sidon-like set of indices with optimized amplitudes.
    Use gradient ascent on log-ratio to break symmetry and suppress off-diagonal peaks.
    """
    # Start with a Sidon set: indices where pairwise sums are distinct
    # A simple Sidon set in [0, 49]: use quadratic residues or a known construction
    # For N=50, use indices: i^2 mod p style, or simply a well-known Sidon set
    # Let's use a greedy Sidon set construction
    sidon = []
    used_sums = set()
    for i in range(N):
        valid = True
        for j in sidon:
            s = i + j
            if s in used_sums:
                valid = False
                break
        if valid:
            # Check sums with itself too (for i==i case, but we handle pairs)
            # Actually for Sidon set, we need all sums a+b (a<=b) to be distinct
            new_sums = set()
            for a in sidon + [i]:
                for b in sidon + [i]:
                    if a <= b:
                        new_sums.add(a + b)
            # Check against existing
            conflict = False
            for s in new_sums:
                if s in used_sums and s not in [2*x for x in sidon] and s not in [2*i]:
                    # Need more careful check
                    pass
            # Simpler: just check if any sum duplicates
            all_sums_new = set()
            for a in range(len(sidon) + 1):
                for b in range(a, len(sidon) + 1):
                    idx_a = sidon[a] if a < len(sidon) else i
                    idx_b = sidon[b] if b < len(sidon) else i
                    all_sums_new.add(idx_a + idx_b)
            if all_sums_new & used_sums:
                valid = False
            if valid:
                sidon.append(i)
                used_sums |= all_sums_new
    
    # If Sidon set is too small, use a different approach
    # For N=50, a Sidon set of size ~7 is possible. Let's use a known one or construct differently.
    # Actually, let's just start with a random sparse initialization and optimize amplitudes
    # on a fixed support, then optimize the support too.
    
    # Phase 1: Optimize amplitudes on a fixed sparse support
    # Use a pseudo-random sparse support with ~10-15 nonzero entries
    np.random.seed(42)
    
    best_ratio = 0.0
    best_h = None
    
    for trial in range(5):
        # Random sparse support
        k = 8 + trial * 2  # 8, 10, 12, 14, 16 nonzero entries
        if k > N:
            k = N
        support = np.random.choice(N, size=k, replace=False)
        
        # Initialize amplitudes: start with equal amplitudes on support
        h0 = np.zeros(N)
        h0[support] = 1.0
        
        # Optimize using log-amplitude parametrization
        # Only optimize the amplitudes on the support
        log_amp0 = np.zeros(k)
        
        def obj_log_amp(x):
            h = np.zeros(N)
            h[support] = np.exp(x)
            h = h / np.sum(h)  # normalize
            l2_sq = np.sum(h * h)
            l1 = np.sum(h)
            if l1 < 1e-15:
                return 0.0
            conv_linear = np.correlate(h, h, mode='full')
            sup_norm = np.max(conv_linear)
            if sup_norm < 1e-15:
                return 0.0
            ratio = l2_sq / (l1 * sup_norm)
            if ratio <= 0:
                return 0.0
            return -np.log(ratio)
        
        try:
            res = minimize(obj_log_amp, log_amp0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            h_opt = np.zeros(N)
            h_opt[support] = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            ratio = _compute_score(h_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Refine by trying to add/remove support points
    if best_h is not None:
        # Try to optimize the full vector with L1 regularization to maintain sparsity
        # Use a few more iterations with different initializations
        for trial in range(3):
            np.random.seed(100 + trial)
            # Perturb the best solution
            h_pert = best_h.copy()
            noise = np.random.randn(N) * 0.1
            h_pert = h_pert + noise
            h_pert = np.maximum(h_pert, 0)
            if np.sum(h_pert) < 1e-15:
                continue
            h_pert = h_pert / np.sum(h_pert)
            
            # Optimize the nonzero entries
            mask = h_pert > 1e-6
            if np.sum(mask) < 3:
                continue
            idx = np.where(mask)[0]
            amp0 = np.log(h_pert[idx] + 1e-12)
            
            def obj_full(x):
                h = np.zeros(N)
                h[idx] = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h * h)
                l1 = np.sum(h)
                if l1 < 1e-15:
                    return 0.0
                conv_linear = np.correlate(h, h, mode='full')
                sup_norm = np.max(conv_linear)
                if sup_norm < 1e-15:
                    return 0.0
                ratio = l2_sq / (l1 * sup_norm)
                if ratio <= 0:
                    return 0.0
                return -np.log(ratio)
            
            try:
                res = minimize(obj_full, amp0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                h_new = np.zeros(N)
                h_new[idx] = np.exp(res.x)
                h_new = h_new / np.sum(h_new)
                ratio = _compute_score(h_new)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_new.copy()
            except Exception:
                pass
    
    # Phase 3: Final refinement with all 50 entries allowed
    if best_h is not None:
        log_h0 = np.log(best_h + 1e-12)
        
        def obj_all(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h * h)
            l1 = np.sum(h)
            if l1 < 1e-15:
                return 0.0
            conv_linear = np.correlate(h, h, mode='full')
            sup_norm = np.max(conv_linear)
            if sup_norm < 1e-15:
                return 0.0
            ratio = l2_sq / (l1 * sup_norm)
            if ratio <= 0:
                return 0.0
            return -np.log(ratio)
        
        try:
            res = minimize(obj_all, log_h0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            h_final = np.exp(res.x)
            h_final = h_final / np.sum(h_final)
            ratio = _compute_score(h_final)
            if ratio > best_ratio:
                best_h = h_final.copy()
                best_ratio = ratio
        except Exception:
            pass
    
    # Ensure nonnegative and return
    if best_h is None:
        return [1.0] * N
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
