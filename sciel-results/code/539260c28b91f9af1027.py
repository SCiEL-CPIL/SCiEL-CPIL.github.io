"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _autoconv_full(h):
    """Compute full autoconvolution using FFT for efficiency."""
    size = 1
    while size < 2 * N - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * H, size)
    return ac[:2 * N - 1]

def _score_sequence(s):
    """Compute the true score for a sequence: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = _autoconv_full(s)
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _soft_max_ac_with_grad(s, temperature=1.0):
    """
    Differentiable surrogate for max |autoconvolution| with gradient.
    Returns (value, gradient w.r.t. s).
    Uses log-sum-exp of (ac/temperature)^2 to approximate max ac^2.
    """
    ac = _autoconv_full(s)
    ac_sq = ac ** 2
    # log-sum-exp for numerical stability
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0, np.zeros_like(s)
    lse = max_val + np.log(np.sum(np.exp(ac_sq - max_val)))
    value = np.sqrt(lse)
    
    # Gradient: d/ds [sqrt(lse)] = (1/(2*sqrt(lse))) * d(lse)/ds
    # lse = max_val + log(sum(exp(ac_sq - max_val)))
    # d(lse)/ds = (1/sum(exp)) * sum(exp * 2*ac * dac/ds)
    # Note: ac[k] = sum_i s[i]*s[k-i], so dac/ds[j] = sum_k s[k-j] (for linear conv)
    # For full autoconv, dac[k]/ds[j] = s[k-j] if 0<=k-j<N else 0
    
    exp_terms = np.exp(ac_sq - max_val)
    weights = exp_terms / np.sum(exp_terms)
    
    # d(lse)/ds = 2 * sum_k weights[k] * ac[k] * dac[k]/ds
    # dac[k]/ds[j] = s[k-j] for valid k-j
    grad_lse = np.zeros(N)
    for k in range(len(ac)):
        w = weights[k]
        if w < 1e-10:
            continue
        # ac[k] = sum_{i} s[i] * s[k-i] where 0<=k-i<N
        # dac[k]/ds[j] = s[k-j] if 0<=k-j<N else 0
        # So d(lse)/ds[j] += 2 * weights[k] * ac[k] * s[k-j]
        # This is equivalent to: grad_lse = 2 * sum_k weights[k]*ac[k] * (s convolved with reversed s at appropriate shifts)
        # More efficiently: note that sum_k weights[k]*ac[k]*s[k-j] = (weights*ac) convolved with s at index N-1-j
        # Let's compute: g[k] = 2*weights[k]*ac[k], then grad_lse[j] = sum_k g[k]*s[k-j]
        # This is cross-correlation of g with s
        pass
    
    # Better approach: use FFT to compute gradient
    # d(lse)/ds = 2 * sum_k weights[k] * ac[k] * (s * s)[k]_j where (s*s)[k]_j = s[k-j]
    # Let g = 2 * weights * ac (length 2N-1)
    # Then grad_lse[j] = sum_k g[k] * s[k-j] for 0<=k-j<N
    # = sum_k g[k] * s[k-j]
    # This is the cross-correlation of g and s
    
    g = 2.0 * weights * ac
    # Compute cross-correlation: grad_lse[j] = sum_{k} g[k] * s[k-j]
    # Using FFT: this is like convolving g with reversed s
    size_fft = 1
    while size_fft < 2 * N:
        size_fft *= 2
    G = np.fft.rfft(g, size_fft)
    S = np.fft.rfft(s, size_fft)
    # Cross-correlation: ifft(fft(g) * conj(fft(s)))
    corr = np.fft.irfft(G * np.conj(S), size_fft)
    # corr[m] = sum_k g[k] * s[k-m+N] ... let me think more carefully
    # Actually, let's just do it directly for clarity since N=400
    # grad_lse[j] = sum_{k=0}^{2N-2} g[k] * s[k-j] where 0 <= k-j < N
    for j in range(N):
        k_min = j
        k_max = min(2 * N - 1, j + N - 1)
        if k_min <= k_max:
            idxs = np.arange(k_min, k_max + 1)
            s_idx = idxs - j
            grad_lse[j] = np.sum(g[idxs] * s[s_idx])
    
    grad_value = grad_lse / (2.0 * value) if value > 1e-10 else np.zeros(N)
    return value, grad_value

def _soft_objective_with_grad(x, temperature=1.0):
    """
    Soft objective with gradient.
    x is a vector of continuous values in [-1, 1] (via sigmoid transform).
    We minimize max|autoconv| / (sum s)^2.
    """
    s = np.tanh(x)  # map to [-1, 1]
    ds_dx = 1.0 - s ** 2  # derivative of tanh
    
    # Avoid division by zero
    sum_s = np.sum(s)
    sq_int = sum_s ** 2
    if sq_int < 1e-10:
        return 1e10, np.zeros_like(x)
    
    soft_max_ac, grad_ac = _soft_max_ac_with_grad(s, temperature)
    
    # f = soft_max_ac / sq_int
    # df/ds = (grad_ac * sq_int - soft_max_ac * 2 * sum_s) / sq_int^2
    #        = grad_ac / sum_s - 2 * soft_max_ac * s / sq_int
    
    if abs(sum_s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    grad_s = grad_ac / sum_s - 2.0 * soft_max_ac * s / sq_int
    
    # Chain rule: df/dx = df/ds * ds/dx
    grad_x = grad_s * ds_dx
    
    value = soft_max_ac / sq_int
    return value, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 40s safety margin
    
    best_score = np.inf
    best_s = None
    
    # Initial random sign patterns
    np.random.seed(42)
    
    # Start with a few random initializations
    for init_iter in range(20):
        if time.time() > deadline:
            break
        
        # Random initial x (before tanh)
        if init_iter % 3 == 0:
            x0 = np.random.randn(N) * 0.1
        elif init_iter % 3 == 1:
            x0 = np.random.randn(N) * 0.5
        else:
            # Start near a random sign pattern
            signs = np.random.choice([-1.0, 1.0], size=N)
            x0 = np.arctanh(np.clip(signs * 0.9, -0.99, 0.99))
        
        # Use L-BFGS-B on the soft objective with gradient
        try:
            result = minimize(
                _soft_objective_with_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                args=(1.0,),
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            
            # Extract the solution
            s_continuous = np.tanh(result.x)
            
            # Binary refinement: threshold to signs
            s_binary = np.sign(s_continuous)
            s_binary[s_binary == 0] = 1.0
            
            # Evaluate the true objective on the binary solution
            score_bin = _score_sequence(s_binary)
            
            if score_bin < best_score:
                best_score = score_bin
                best_s = s_binary.copy()
                
        except Exception:
            continue
        
        # Also try the continuous solution directly (in case it's better)
        s_cont = np.tanh(result.x)
        score_cont = _score_sequence(s_cont)
        if score_cont < best_score:
            best_score = score_cont
            best_s = s_cont.copy()
    
    # Local search: sign flips on the best solution found
    if best_s is not None:
        current_s = best_s.copy()
        current_score = best_score
        
        # Determine if we are working with binary or continuous values
        is_binary = np.allclose(current_s, np.sign(current_s))
        
        # First pass: greedy local search with multiple flips per iteration
        for flip_iter in range(800):
            if time.time() > deadline:
                break
            
            # Try flipping a random index (or adjusting value if continuous)
            idx = np.random.randint(N)
            old_val = current_s[idx]
            
            if is_binary:
                current_s[idx] = -old_val
            else:
                # Small random perturbation for continuous case
                current_s[idx] = old_val + np.random.randn() * 0.1
                current_s[idx] = np.clip(current_s[idx], -1, 1)
            
            # Compute new score
            new_score = _score_sequence(current_s)
            
            if new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_s = current_s.copy()
            else:
                # Revert
                current_s[idx] = old_val
        
        # Simulated annealing-like exploration
        for sa_iter in range(300):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(N)
            old_val = current_s[idx]
            
            if is_binary:
                current_s[idx] = -old_val
            else:
                current_s[idx] = old_val + np.random.randn() * 0.2
                current_s[idx] = np.clip(current_s[idx], -1, 1)
            
            new_score = _score_sequence(current_s)
            
            delta = new_score - current_score
            # Accept with probability exp(-delta / T) where T decreases
            T = max(0.01, 0.1 * (1 - sa_iter / 300.0))
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_s = current_s.copy()
            else:
                current_s[idx] = old_val
        
        # Final polish: try small continuous adjustments around the best solution
        if not is_binary:
            for polish_iter in range(100):
                if time.time() > deadline:
                    break
                idx = np.random.randint(N)
                old_val = current_s[idx]
                current_s[idx] = old_val + np.random.randn() * 0.05
                current_s[idx] = np.clip(current_s[idx], -1, 1)
                new_score = _score_sequence(current_s)
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    best_s = current_s.copy()
                else:
                    current_s[idx] = old_val
    
    # If we didn't find anything, return all ones
    if best_s is None:
        return np.ones(N)
    
    # Return as a sequence (list or array)
    return best_s
# EVOLVE-BLOCK-END
