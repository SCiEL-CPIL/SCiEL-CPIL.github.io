"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    C_k = sum_i h_i * (1 - h_{i+k}) for shift k (positive and negative).
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1, index N-1 is shift 0
    # We want all shifts except 0
    c_neg = c[:N-1]      # negative shifts
    c_pos = c[N:]        # positive shifts (excluding shift 0 at index N-1)
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return float(np.max(overlaps))

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a uniform shift for exact sum.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h_test = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_test)
        if s < target:
            lo = mid
        else:
            hi = mid
        if abs(s - target) < 1e-14:
            break
    
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_lse(x, beta, lam_sum):
    """
    Log-sum-exp smoothed objective for the max overlap.
    x = h (length N). Returns LSE of overlaps + sum penalty.
    """
    h = np.clip(x, 0.0, 1.0)
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        lse_val = 0.0
    else:
        max_ov = np.max(overlaps)
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50.0 / beta, 0.0)
        lse_val = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return lse_val + sum_penalty

def _objective_grad(x, beta, lam_sum):
    """
    Objective and analytical gradient using log-sum-exp.
    """
    h = np.clip(x, 0.0, 1.0)
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        lse_val = 0.0
        grad_ov = np.zeros(N)
    else:
        max_ov = np.max(overlaps)
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50.0 / beta, 0.0)
        exps = np.exp(beta * shifted)
        sum_exp = np.sum(exps)
        weights = exps / sum_exp  # softmax weights
        lse_val = max_ov + np.log(sum_exp) / beta
        
        # Gradient of LSE w.r.t. h
        # overlap_k = sum_i h_i * (1 - h_{i+k})
        # d(overlap_k)/d(h_j) = (1 - h_{j+k}) - h_{j-k}  (for appropriate shifts)
        grad_ov = np.zeros(N)
        
        # For each shift k, the contribution to gradient
        # We need to be careful with the linear (non-circular) correlation
        # overlap for shift k>0: sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
        # d/dh_j = sum over k where j is involved
        
        # Let's compute numerically-ish but efficiently
        g = 1.0 - h
        # For positive shifts k=1..N-1:
        # C_k = sum_{i=0}^{N-1-k} h_i * g_{i+k}
        # dC_k/dh_j = g_j if j <= N-1-k (as first factor) 
        #            + h_{j-k} if j >= k (as second factor, since g_{i+k}=g_j means i=j-k)
        # Actually: C_k = sum_i h_i * g_{i+k}
        # dC_k/dh_j: j appears as h_j when i=j (if j+k < N), contributes g_{j+k}
        #            j appears in g_{i+k}=g_j when i+k=j, i=j-k (if j-k >= 0), contributes h_{j-k} * (-1)
        # So dC_k/dh_j = g_{j+k} (if j+k<N) - h_{j-k} (if j-k>=0)
        
        for k in range(1, N):
            w = weights[k-1]  # weight for positive shift k
            # dC_k/dh_j for j=0..N-1
            # g_{j+k} valid when j+k < N, i.e., j < N-k
            # h_{j-k} valid when j-k >= 0, i.e., j >= k
            grad_pos = np.zeros(N)
            # First term: g_{j+k} for j=0..N-1-k
            if N - k > 0:
                grad_pos[:N-k] += g[k:]
            # Second term: -h_{j-k} for j=k..N-1
            if k < N:
                grad_pos[k:] -= h[:N-k]
            grad_ov += w * grad_pos
        
        # For negative shifts k=-1..-(N-1), i.e., shift -k where k=1..N-1:
        # C_{-k} = sum_i h_i * g_{i-k} (valid when i-k >= 0, i.e., i >= k)
        # = sum_{i=k}^{N-1} h_i * g_{i-k}
        # dC_{-k}/dh_j: 
        #   j appears as h_j when j >= k, contributes g_{j-k}
        #   j appears in g_{i-k}=g_j when i-k=j, i=j+k (if j+k < N), contributes h_{j+k}*(-1)
        # So dC_{-k}/dh_j = g_{j-k} (if j>=k) - h_{j+k} (if j+k<N)
        
        for k in range(1, N):
            w = weights[N-1+k-1]  # weight for negative shift -k, index in overlaps
            # overlaps array: [c_neg (N-1 elements for shifts -1..-(N-1)), c_pos (N-1 elements for shifts 1..N-1)]
            # Wait, let me recheck the indexing
            # c_neg = c[:N-1] corresponds to shifts -(N-1), ..., -1
            # c_pos = c[N:] corresponds to shifts 1, ..., N-1
            # So overlaps[0] = shift -(N-1), overlaps[N-2] = shift -1
            # overlaps[N-1] = shift 1, overlaps[2N-3] = shift N-1
            pass
        
        # Let me redo this more carefully
        # overlaps = concat(c_neg, c_pos)
        # c_neg has N-1 elements: c[0]..c[N-2], corresponding to shifts -(N-1)..-1
        # c_pos has N-1 elements: c[N]..c[2N-2], corresponding to shifts 1..(N-1)
        
        # So for index idx in overlaps:
        # if idx < N-1: shift = -(N-1-idx) = idx-(N-1)
        # if idx >= N-1: shift = idx-(N-1)
        
        grad_ov = np.zeros(N)
        for idx in range(len(overlaps)):
            w = weights[idx]
            if idx < N - 1:
                # negative shift: k = idx - (N-1), so shift = k (negative)
                # Let k = N-1-idx (positive magnitude), shift = -k
                k = N - 1 - idx
                # C_{-k} = sum_{i=k}^{N-1} h_i * g_{i-k}
                # dC_{-k}/dh_j = g_{j-k} (if j>=k) - h_{j+k} (if j+k<N)
                grad_neg = np.zeros(N)
                if N - k > 0:
                    grad_neg[k:] += g[:N-k]
                if k < N:
                    grad_neg[:N-k] -= h[k:]
                grad_ov += w * grad_neg
            else:
                # positive shift: k = idx - (N-1)
                k = idx - (N - 1)
                # C_k = sum_{i=0}^{N-1-k} h_i * g_{i+k}
                # dC_k/dh_j = g_{j+k} (if j+k<N) - h_{j-k} (if j-k>=0)
                grad_pos = np.zeros(N)
                if N - k > 0:
                    grad_pos[:N-k] += g[k:]
                if k < N:
                    grad_pos[k:] -= h[:N-k]
                grad_ov += w * grad_pos
    
    # Sum constraint gradient
    grad_sum = 2.0 * lam_sum * (np.sum(h) - TARGET_SUM) * np.ones(N)
    
    # Clip gradient to respect bounds (projected gradient)
    grad = grad_ov + grad_sum
    # Zero out gradient where h is at bound
    grad[(h <= 0.0) & (grad < 0)] = 0.0
    grad[(h >= 1.0) & (grad > 0)] = 0.0
    
    return lse_val + sum_penalty, grad

def _generate_difference_set_init():
    """
    Generate initial h from a cyclic difference set or similar structure.
    For N=100, we look for a (v,k,lambda) design.
    A (100, 50, lambda) difference set would have lambda = k(k-1)/(v-1) = 50*49/99 ≈ 24.95
    This isn't an integer, so we use the nearest structure.
    
    Alternative: use a known good pattern.
    """
    # Start with a pattern that has good autocorrelation properties
    # Use a chirp-like or Gold code inspired pattern
    
    t = np.arange(N)
    
    # Pattern 1: Based on quadratic residues mod p (adapted for composite N)
    # For prime p, the set of quadratic residues has good correlation properties
    # For N=100, use a combination
    
    # Pattern: alternate blocks with varying density
    h = np.zeros(N)
    # Use a pattern inspired by m-sequences or Gold codes
    # Simple approach: use a pseudo-random but structured pattern
    
    # Try: h_i = 1 if (i^2 mod 97) < 48 else 0 (using prime 97 close to 100)
    p = 97
    residues = set()
    for x in range(p):
        residues.add((x*x) % p)
    
    h = np.zeros(N)
    for i in range(N):
        if i < p:
            h[i] = 1.0 if i in residues else 0.0
        else:
            h[i] = 0.5  # fill remaining
    
    h = _project_to_feasible(h)
    return h

def _generate_gold_code_init():
    """Generate initial h inspired by Gold codes."""
    # Gold codes have good autocorrelation properties
    # For N=100, we'll create a pattern with good spectral properties
    
    # Use a combination of two m-sequence-like patterns
    t = np.arange(N)
    
    # Pattern based on sine with specific frequency
    h1 = (np.sin(2 * np.pi * 7 * t / N) > 0).astype(float)
    h2 = (np.sin(2 * np.pi * 13 * t / N) > 0).astype(float)
    
    # XOR-like combination
    h = ((h1 + h2) > 1).astype(float)
    
    h = _project_to_feasible(h)
    return h

def _generate_random_binary_init(rng):
    """Generate a random binary-like initialization."""
    h = rng.integers(0, 2, N).astype(float)
    # Adjust to get exactly N/2 ones
    ones = np.sum(h == 1)
    if ones > N/2:
        excess = int(ones - N/2)
        one_indices = np.where(h == 1)[0]
        to_flip = rng.choice(one_indices, excess, replace=False)
        h[to_flip] = 0.0
    elif ones < N/2:
        deficit = int(N/2 - ones)
        zero_indices = np.where(h == 0)[0]
        to_flip = rng.choice(zero_indices, deficit, replace=False)
        h[to_flip] = 1.0
    return h

def _coordinate_descent_polish(h, max_iter=500, tol=1e-10):
    """
    Coordinate descent local search to polish the solution.
    For each coordinate, try to adjust it to reduce max overlap.
    """
    h = h.copy()
    best_score = _max_overlap(h)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            base = h[i]
            # Try several candidate values
            candidates = [0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
            # Add values near current
            candidates.append(base + 0.01)
            candidates.append(base - 0.01)
            candidates.append(base + 0.05)
            candidates.append(base - 0.05)
            
            best_local = base
            best_local_score = best_score
            
            for new_val in candidates:
                if 0.0 <= new_val <= 1.0:
                    h[i] = new_val
                    h = _project_to_feasible(h)
                    score = _max_overlap(h)
                    if score < best_local_score - tol:
                        best_local_score = score
                        best_local = new_val
            
            h[i] = best_local
            if best_local_score < best_score - tol:
                best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return h, best_score

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses log-sum-exp smoothed objective with multiple initializations,
    followed by coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(123)
    
    best_h = None
    best_score = float('inf')
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Difference set inspired
    h_ds = _generate_difference_set_init()
    inits.append(h_ds)
    
    # 2. Gold code inspired
    h_gc = _generate_gold_code_init()
    inits.append(h_gc)
    
    # 3. Random binary
    for seed in [1, 7, 42, 123]:
        rng_local = np.random.default_rng(seed)
        h_rb = _generate_random_binary_init(rng_local)
        inits.append(h_rb)
    
    # 4. Sinusoidal patterns
    t = np.arange(N)
    for freq in [3, 5, 7, 11]:
        h_sin = (0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N)).astype(float)
        h_sin = _project_to_feasible(h_sin)
        inits.append(h_sin)
    
    # 5. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.8
    h_alt[1::2] = 0.2
    h_alt = _project_to_feasible(h_alt)
    inits.append(h_alt)
    
    # Evaluate all initializations
    for h_init in inits:
        h_init = _project_to_feasible(h_init)
        score = _max_overlap(h_init)
        if score < best_score:
            best_score = score
            best_h = h_init.copy()
    
    # Phase 1: Smooth optimization with increasing beta
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
        
        h_current = best_h.copy()
        
        def obj_and_grad(x):
            return _objective_grad(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_and_grad, h_current, method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Phase 2: Higher beta for sharper optimization
    if time.time() < deadline - 20.0:
        for beta in [500, 1000]:
            if time.time() > deadline - 10.0:
                break
            
            h_current = best_h.copy()
            
            def obj_and_grad_hi(x):
                return _objective_grad(x, beta, 500.0)
            
            try:
                result = minimize(
                    obj_and_grad_hi, h_current, method='L-BFGS-B',
                    jac=True,
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 5.0:
        h_cd, score_cd = _coordinate_descent_polish(best_h, max_iter=300, tol=1e-10)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd
    
    # Final: ensure validity
    best_h = _project_to_feasible(best_h)
    
    # Exact sum correction
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        # Find elements not at bounds
        mask = (best_h > 1e-10) & (best_h < 1.0 - 1e-10)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
        else:
            # All at bounds, adjust one element
            idx = np.argmin(np.abs(best_h - 0.5))
            best_h[idx] += diff
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h

# EVOLVE-BLOCK-END
