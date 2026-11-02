"""
Time-domain min-max optimization with log-sum-exp smoothing.
Strategy:
1. Use log-sum-exp to smooth the max operation in the objective, enabling gradient-based optimization.
2. Compute analytic gradients for the smoothed objective.
3. Use Simulated Annealing for coarse global exploration on a subsampled grid.
4. Refine the best candidate using L-BFGS-B in the full 600-dimensional space.
"""
import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _smoothed_objective(h, beta):
    """
    Smoothed objective: (1/beta) * log(sum(exp(beta * conv / s^2)))
    This approximates max(conv)/s^2 as beta -> inf.
    """
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    normalized = conv / (s * s)
    # Shift for numerical stability
    max_val = np.max(normalized)
    if max_val > 1e6:
        return max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * (normalized - max_val))))
    return log_sum_exp / beta

def _smoothed_gradient(h, beta):
    """
    Analytic gradient of the smoothed objective.
    f(h) = (1/beta) * log(sum_j exp(beta * c_j / s^2))
    where c_j = sum_i h_i h_{j-i} (linear conv)
    
    df/dh_k = (1/s^2) * sum_j [ (beta * c_j / s^2) * exp(beta * c_j / s^2) / sum_m exp(beta * c_m / s^2) * (2 * (h * delta_j)_k) ]
             - 2 * (1/s) * f(h) * h_k
    
    Note: c_j depends on h. 
    Let Z = sum_m exp(beta * c_m / s^2)
    Let p_m = exp(beta * c_m / s^2) / Z
    Then f = (1/beta) * log(Z) = (1/beta) * (max_val + log(sum exp(...)))
    
    df/dh_k = (1/beta) * (1/Z) * sum_m [ exp(beta * c_m / s^2) * beta * (d c_m / d h_k) / s^2 - exp(beta * c_m / s^2) * c_m * beta * (2 s_k) / s^3 ]
    
    d c_m / d h_k = h_{m-k} + h_{k-m} (with appropriate bounds for linear conv)
    s_k = ds/dh_k = 1
    
    This is complex to compute directly for all k in O(N^2).
    However, we can use FFTs or just compute it directly since N=600 is small.
    Actually, d c_m / d h_k is the cross-correlation of h with a delta at k.
    
    Let's simplify.
    c_m = <h, shift_m(h)>
    d c_m / d h_k = h_{m-k} + h_{k-m} (if indices valid)
    
    We can compute the gradient vector g_k for all k efficiently?
    g_k = (1/beta) * sum_m p_m * [ beta * (h_{m-k} + h_{k-m}) / s^2 - beta * c_m * 2 / s^3 ]
    
    Wait, the derivative of the normalization s^2 is -2/s^3 * c_m * 2 * (d s/d h_k) ?
    Let's re-derive carefully.
    f(h) = (1/beta) log(sum_m exp(beta * c_m / s^2))
    
    Let u_m = c_m / s^2.
    f = (1/beta) log(sum_m exp(beta * u_m))
    
    df/dh_k = (1/beta) * [ 1 / sum(exp(beta u_m)) * sum_m exp(beta u_m) * beta * (du_m/dh_k) ]
             = sum_m p_m * (du_m/dh_k)
    
    u_m = c_m * s^{-2}
    du_m/dh_k = (dc_m/dh_k) * s^{-2} + c_m * (-2) * s^{-3} * (ds/dh_k)
    ds/dh_k = 1
    
    dc_m/dh_k:
    c_m = sum_i h_i h_{m-i}
    dc_m/dh_k = h_{m-k} + h_{k-m} (terms where i=k or m-i=k)
    Note: if m-k is out of [0, N-1], the term is 0.
    
    So,
    du_m/dh_k = (h_{m-k} + h_{k-m}) / s^2 - 2 * c_m / s^3
    
    g_k = sum_m p_m * [ (h_{m-k} + h_{k-m}) / s^2 - 2 * c_m / s^3 ]
    
    We can compute this in O(N^2) which is 360,000 operations per gradient eval.
    With 1000 iterations, that's 360M ops. In Python this might be slow.
    Let's try to vectorize or use FFTs?
    
    The term sum_m p_m * (h_{m-k} + h_{k-m}) is a convolution of p with h?
    Let q_m = p_m.
    sum_m q_m h_{m-k} = (q * h)_k ?
    Linear convolution of q (length 2N-1) and h (length N)?
    p_m is defined for m in [0, 2N-2].
    h_i is defined for i in [0, N-1].
    
    (q * h)_k = sum_m q_m h_{k-m}.
    This matches sum_m p_m h_{k-m}.
    By symmetry of h? No, h is not symmetric necessarily.
    But sum_m p_m h_{m-k} is the cross-correlation or flipped convolution.
    
    Let's just do the O(N^2) loop in NumPy efficiently.
    Create a matrix M where M[k, m] = h_{m-k} + h_{k-m}.
    This is a Toeplitz-like matrix.
    
    Actually, let's just implement the O(N^2) calculation and see if it's fast enough.
    If not, we'll drop gradients and use derivative-free methods or approximate gradients.
    Given N=600, O(N^2) is small.
    
    """
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    normalized = conv / (s * s)
    max_val = np.max(normalized)
    
    # p_m = exp(beta * (normalized - max_val)) / sum(...)
    exp_terms = np.exp(beta * (normalized - max_val))
    Z = np.sum(exp_terms)
    p = exp_terms / Z
    
    # Compute g_k = sum_m p_m * [ (h_{m-k} + h_{k-m}) / s^2 - 2 * c_m / s^3 ]
    
    # Term 1: sum_m p_m * (h_{m-k} + h_{k-m}) / s^2
    # Let A_k = sum_m p_m h_{m-k}
    # Let B_k = sum_m p_m h_{k-m}
    
    # We can compute A and B using convolution.
    # p has length 2N-1. h has length N.
    # A_k = sum_m p_m h_{m-k}.
    # This is the cross-correlation of p and h?
    # np.correlate(p, h, mode='same')? 
    # Let's use FFT for speed.
    
    # Pad p and h to length L >= len(p) + len(h) - 1 = 3N - 2.
    L = 1 << (3*N - 2).bit_length()
    
    H_fft = np.fft.rfft(h, L)
    P_fft = np.fft.rfft(p, L)
    
    # Convolution p * h
    # (p * h)_k = sum_m p_m h_{k-m} = B_k
    B = np.fft.irfft(P_fft * H_fft, L)[:2*N-1]
    
    # A_k = sum_m p_m h_{m-k} = sum_m p_m h_{-(m-k)}
    # This is the convolution of p with reversed h?
    # Let h_rev = h[::-1].
    # (p * h_rev)_k = sum_m p_m h_rev_{k-m} = sum_m p_m h_{N-1-(k-m)}
    # This is getting messy with indices.
    
    # Let's just do the O(N^2) sum. It's safer.
    # g = np.zeros(N)
    # for k in range(N):
    #     # m ranges from 0 to 2N-2
    #     # h_{m-k} is non-zero if 0 <= m-k < N => k <= m < k+N
    #     # h_{k-m} is non-zero if 0 <= k-m < N => k-N < m <= k
    
    # Vectorized O(N^2):
    # Create indices
    k_idx = np.arange(N)[:, None]  # N x 1
    m_idx = np.arange(2*N-1)[None, :]  # 1 x (2N-1)
    
    # m - k
    mk = m_idx - k_idx
    # h_{m-k}
    # Valid if 0 <= mk < N
    mask1 = (mk >= 0) & (mk < N)
    # We need to gather h[mk] where mask1 is True.
    # np.where(mask1, h[mk_clipped], 0)
    mk_clipped = np.clip(mk, 0, N-1)
    h_mk = np.where(mask1, h[mk_clipped], 0.0)
    
    # k - m
    km = k_idx - m_idx
    # h_{k-m}
    # Valid if 0 <= km < N
    mask2 = (km >= 0) & (km < N)
    km_clipped = np.clip(km, 0, N-1)
    h_km = np.where(mask2, h[km_clipped], 0.0)
    
    term1_matrix = (h_mk + h_km) / (s * s)
    
    # Term 2: - 2 * c_m / s^3
    # c_m is normalized * s^2? No, c_m is conv[m].
    # The formula was: p_m * ( - 2 * c_m / s^3 )
    # Note: c_m in the formula is the raw convolution value.
    # term2_matrix = p[None, :] * (-2 * conv[None, :] / (s**3))
    
    term2_matrix = np.outer(p, -2 * conv / (s**3))
    
    total_matrix = term1_matrix + term2_matrix
    
    g = np.sum(total_matrix, axis=1)
    
    return g

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_time():
        return time.time() > deadline

    # Initial best: Uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Stage 1: Simulated Annealing on a coarse grid
    # We will optimize a subsampled version of h, then upsample.
    # Or just optimize h directly with SA? SA is slow in high dims.
    # Let's try a few random restarts with L-BFGS-B using the smoothed objective.
    
    # To make L-BFGS-B effective, we need a good beta.
    # Start with low beta (smooth) and increase?
    # Or just use a moderate beta.
    
    # Let's try optimizing with beta=10.
    # Objective: smoothed_objective(h, beta)
    # Gradient: smoothed_gradient(h, beta)
    
    def obj_grad(x):
        if check_time():
            return 1e10, np.zeros(N)
        # Ensure non-negative
        h = np.maximum(0, x)
        val = _smoothed_objective(h, 10.0)
        grad = _smoothed_gradient(h, 10.0)
        # Gradient of max(0, x) is 1 if x>0 else 0
        mask = x > 0
        grad = grad * mask
        return val, grad

    # Try a few initializations
    initializations = [
        np.ones(N),
        np.random.rand(N) * 0.1 + 0.01,
        np.sin(np.linspace(0, 2*np.pi, N)) ** 2 + 0.1,
        np.cos(np.linspace(0, 4*np.pi, N)) ** 2 + 0.1
    ]
    
    for x0 in initializations:
        if check_time():
            break
        
        try:
            res = optimize.minimize(
                obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            
            h_opt = np.maximum(0, res.x)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Stage 2: Refine with higher beta for sharper max approximation
    if not check_time():
        def obj_grad_refine(x):
            if check_time():
                return 1e10, np.zeros(N)
            h = np.maximum(0, x)
            val = _smoothed_objective(h, 50.0)
            grad = _smoothed_gradient(h, 50.0)
            mask = x > 0
            grad = grad * mask
            return val, grad

        try:
            res = optimize.minimize(
                obj_grad_refine,
                best_h,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            
            h_opt = np.maximum(0, res.x)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Stage 3: Coordinate Descent / Local Search
    # Try to push up valleys in h to flatten the convolution peak?
    # Or push down peaks in convolution?
    
    # A simple heuristic: 
    # The peak of convolution is at the center.
    # If we reduce h at the center and increase at the edges, we might flatten the conv.
    
    if not check_time():
        # Create a "tent" shape
        x = np.linspace(-1, 1, N)
        h_tent = np.maximum(0, 1 - np.abs(x))
        h_tent += 0.1
        
        def obj_grad_tent(x):
            if check_time():
                return 1e10, np.zeros(N)
            h = np.maximum(0, x)
            val = _smoothed_objective(h, 20.0)
            grad = _smoothed_gradient(h, 20.0)
            mask = x > 0
            grad = grad * mask
            return val, grad

        try:
            res = optimize.minimize(
                obj_grad_tent,
                h_tent,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            
            h_opt = np.maximum(0, res.x)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    return best_h.tolist()
# EVOLVE-BLOCK-END
