# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _union_length_sq(p, N, n_c=200):
    """
    Smooth surrogate for ∫₀¹ L(c)² dc where L(c) = union length of N intervals
    [p[i] + c*i/N, p[i] + c*i/N + 1/N].
    
    Uses log-sum-exp trick: L(c) = sum_i w_i(c) where w_i(c) is the length of
    the part of interval i not covered by earlier intervals. We approximate with
    a soft version: for each c, L(c) ≈ sum of exp(log_w) / (1 + sum of exp(log_w))
    where log_w_i = log(1/N) + log(1 - overlap_fraction).
    
    Actually, a simpler approach: L(c) = max_i (right_i) - min_i (left_i) is wrong
    for union. Let me use a different smooth approximation.
    
    For N=8 intervals of width w=1/8 at height c:
    left_i = p[i] + c*i/N
    right_i = p[i] + c*i/N + w
    
    The union length is the total measure of ∪[left_i, right_i].
    
    A smooth approximation: for a grid of c values, compute the exact union length
    (which is piecewise linear in c), then integrate. But for differentiability,
    use a soft-min/soft-max formulation.
    
    Alternative: parameterize by sorting. For fixed c, sort intervals by left endpoint.
    The union length = sum over merged intervals. This is non-smooth due to sorting.
    
    Better approach for N=8: use the fact that union length = sum_i (right_i - left_i) 
    - sum of overlaps. For N=8, we can enumerate all 2^8 subsets... no, that's too slow.
    
    Let me use a different strategy: since N=8 is small, compute the exact union length
    at each c by sorting and merging, and use a numerical gradient-free optimizer
    (Nelder-Mead) on the objective f(p) = ∫₀¹ L(c)² dc computed via quadrature.
    """
    w = 1.0 / N
    # Sample c values
    c_vals = np.linspace(0, 1, n_c)
    
    total = 0.0
    for c in c_vals:
        # Compute left and right endpoints
        lefts = p + c * np.arange(N) / N
        rights = lefts + w
        
        # Sort by left endpoint
        order = np.argsort(lefts)
        s_lefts = lefts[order]
        s_rights = rights[order]
        
        # Merge intervals to compute union length
        union_len = 0.0
        cur_left = s_lefts[0]
        cur_right = s_rights[0]
        for k in range(1, N):
            if s_lefts[k] <= cur_right:
                cur_right = max(cur_right, s_rights[k])
            else:
                union_len += cur_right - cur_left
                cur_left = s_lefts[k]
                cur_right = s_rights[k]
        union_len += cur_right - cur_left
        
        total += union_len ** 2
    
    # Trapezoidal integration
    return total / (n_c - 1)


def _optimize_1d(N, max_time=580.0, initial_seed=42):
    """
    Optimize 1D base positions p[0..N-1] to minimize ∫₀¹ L(c)² dc.
    Uses Nelder-Mead with multiple restarts.
    """
    w = 1.0 / N
    best_p = None
    best_val = np.inf
    start_time = time.time()
    
    # Initial guess: Keich-like or uniform
    # Try several initializations
    rng = np.random.default_rng(initial_seed)
    
    # Candidate initializations
    inits = []
    
    # 1. Uniform spacing
    inits.append(np.linspace(0, 1 - w, N))
    
    # 2. Clumped at origin
    inits.append(np.zeros(N))
    
    # 3. Random
    for _ in range(5):
        inits.append(rng.uniform(-0.5, 1.5, N))
    
    # 4. Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    inits.append(keich.copy())
    inits.append(keich.copy() + 0.1)
    
    for init in inits:
        if time.time() - start_time > max_time - 30:
            break
        
        try:
            res = minimize(
                _union_length_sq,
                init,
                args=(N, 150),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            val = _union_length_sq(res.x, N, 150)
            if val < best_val:
                best_val = val
                best_p = res.x.copy()
        except Exception:
            continue
    
    if best_p is None:
        best_p = np.zeros(N)
    
    return best_p


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Exploits the factorization of the union volume:
    V = ∫₀¹ Lₓ(c) · Lᵧ(c) dc
    
    where Lₓ(c) is the 1D union length of x-intervals and Lᵧ(c) for y-intervals.
    Since the x and y problems are independent, we optimize each separately.
    """
    N = 8
    start_time = time.time()
    time_budget = 580.0  # Leave margin before 600s limit
    
    # Optimize x-positions
    x_pos = _optimize_1d(N, max_time=time_budget * 0.45, initial_seed=42)
    
    # Optimize y-positions  
    y_pos = _optimize_1d(N, max_time=time_budget * 0.45, initial_seed=123)
    
    # Build 2D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_pos[i_flat]
    base_y = y_pos[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
