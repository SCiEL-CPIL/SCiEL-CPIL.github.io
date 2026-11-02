import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h.
    
    The step function f is defined as:
    f(x) = h[i] for x in [i*DELTA, (i+1)*DELTA), i = 0, ..., N-1
    f(x) = 0 otherwise
    
    The autoconvolution is (f*f)(t) = integral f(s) f(t-s) ds.
    This is a piecewise linear function with breakpoints at multiples of DELTA.
    The maximum occurs at one of these breakpoints (or we can check all).
    
    We compute (f*f)(k*DELTA) for k = 0, ..., 2*(N-1).
    """
    n = len(h)
    # Number of intervals where the convolution is nonzero: from 0 to 2*(N-1)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        # (f*f)(k*DELTA) = DELTA * sum_{i,j} h[i]*h[j] * overlap_length(i, j, k)
        # where overlap_length is the length of intersection of [i, i+1] and [k-j, k-j+1]
        # (in units of DELTA)
        # i ranges over [0, n-1], j ranges over [0, n-1]
        # For given i, j must satisfy k-j in [i-1, i+1], so j in [k-i-1, k-i+1]
        
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            # overlap of [i, i+1] and [k-j, k-j+1]
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Strategy based on Sidon set insight:
    # A Sidon set (B_2 sequence) has the property that all pairwise sums i+j 
    # (with i <= j) are distinct. For a height vector supported on a Sidon set,
    # the autoconvolution peaks at shifts corresponding to i+j are all at distinct 
    # locations, and the diagonal peak (shift 0, from i=j terms) dominates.
    # 
    # Actually, let me reconsider. The autoconvolution of a step function supported
    # on indices S has peaks at shifts k where k = i + j for i, j in S (in the 
    # discrete sense). For a Sidon set, all such sums i+j (with i <= j) are unique,
    # meaning the autoconvolution has narrow peaks.
    #
    # The key insight from the cell findings: using a Sidon set structure can achieve
    # a ratio close to 1.15, much higher than the uniform case (~0.82).
    #
    # Let's construct a Sidon-like pattern and then optimize.
    
    np.random.seed(42)
    
    # Generate a Sidon set in [0, N-1]
    # A simple greedy approach:
    def generate_sidon_set(n, max_size=None):
        if max_size is None:
            max_size = n
        sidon = [0]
        used_sums = {0}  # sums i+j where i <= j
        for candidate in range(1, n):
            # Check if adding candidate maintains Sidon property
            valid = True
            for i in sidon:
                s = i + candidate
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Check sums among existing elements with candidate
                for i in sidon:
                    if i + candidate in used_sums:
                        valid = False
                        break
                if valid:
                    sidon.append(candidate)
                    used_sums.add(candidate + candidate)
                    for i in sidon[:-1]:
                        used_sums.add(i + candidate)
            if len(sidon) >= max_size:
                break
        return sidon
    
    # For N=50, we want a Sidon set of reasonable size
    # Maximum Sidon set in [0, 49] has size around 10-15
    sidon_set = generate_sidon_set(N, max_size=12)
    
    initializations = []
    
    # 1. Sidon set with unit heights
    h_sidon = np.zeros(N)
    for idx in sidon_set:
        h_sidon[idx] = 1.0
    initializations.append(h_sidon)
    
    # 2. Sidon set with optimized heights
    initializations.append(h_sidon.copy())
    
    # 3. Sparse random patterns (inspired by Sidon structure)
    for trial in range(20):
        h_sparse = np.zeros(N)
        num_ones = np.random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sparse[indices] = np.random.uniform(0.5, 1.5, num_ones)
        initializations.append(h_sparse)
    
    # 4. Uniform
    initializations.append(np.ones(N))
    
    # 5. Triangular
    t = np.linspace(-1, 1, N)
    initializations.append(np.maximum(1 - t**2, 0.01))
    
    # 6. Inverse triangular  
    initializations.append(np.maximum(t**2 + 0.1, 0.01))
    
    # 7. Alternating sparse
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    # 8. Every 3
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    # 9. Gaussian
    initializations.append(np.exp(-t**2/0.3))
    
    # 10. Bimodal
    initializations.append(np.maximum(1.0 + 2.0*np.exp(-((t-0.6)**2)/0.1) + 2.0*np.exp(-((t+0.6)**2)/0.1), 0.05))
    
    # 11. More Sidon-like constructions with different sizes
    for size in [8, 10, 14]:
        s = generate_sidon_set(N, max_size=size)
        h_s = np.zeros(N)
        for idx in s:
            h_s[idx] = 1.0
        initializations.append(h_s)
    
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        def neg_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -_ratio(x / s)
        
        try:
            result = minimize(neg_ratio, h0.copy(), method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if time.time() - start_time > DEADLINE - 5:
                break
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            continue
    
    # If we still have time, do local refinement around the best solution
    if time.time() - start_time < DEADLINE - 10:
        try:
            def neg_ratio2(x):
                x = np.maximum(x, 0)
                s = np.sum(x)
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            
            result = minimize(neg_ratio2, best_h.copy(), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                sc = _ratio(h_candidate)
                if sc > best_score:
                    best_score = sc
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h.tolist()
