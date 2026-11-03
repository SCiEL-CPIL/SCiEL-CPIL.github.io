import numpy as np
from scipy.optimize import minimize
import time

def compute_score(f):
    """
    Compute the score for the given step function f (array of nonnegative values).
    The score is the normalized maximum of the circular autoconvolution.
    score = max_k c_k / mean_k c_k = n * max(c_k) / (sum f)^2
    """
    n = len(f)
    if n == 0 or np.sum(f) == 0:
        return 0.0
    # Circular autoconvolution via FFT
    F = np.fft.fft(f)
    P = np.abs(F) ** 2
    c = np.fft.ifft(P).real
    # Numerical cleanup
    c = np.clip(c, 0, None)
    max_c = np.max(c)
    mean_c = np.mean(c)  # equals (sum f)^2 / n
    if mean_c == 0:
        return 0.0
    return n * max_c / (np.sum(f) ** 2)

def fft_autoconv(f):
    """Compute circular autoconvolution via FFT."""
    n = len(f)
    F = np.fft.fft(f)
    P = np.abs(F) ** 2
    c = np.fft.ifft(P).real
    return np.clip(c, 0, None)

def project_simplex(v, s=1.0):
    """
    Project vector v onto the simplex {x >= 0, sum(x) = s}.
    """
    n = len(v)
    if n == 0:
        return v
    v = np.asarray(v, dtype=float)
    # Scale to sum 1 first, then scale back
    v_pos = np.maximum(v, 0)
    s_v = v_pos.sum()
    if s_v == 0:
        return np.full(n, s / n)
    v_pos = v_pos / s_v  # now sums to 1
    # Project onto simplex sum=1
    u = np.sort(v_pos)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v_pos - theta, 0)
    # Scale to sum s
    w_sum = w.sum()
    if w_sum > 0:
        w = w * (s / w_sum)
    else:
        w = np.full(n, s / n)
    return w

def spectral_coordinate_descent(f0, max_passes=100, tol=1e-12, deadline=None):
    """
    Spectral coordinate descent: update one bin at a time.
    For each bin j, we compute the new autoconvolution if we change f[j].
    We use the fact that changing f[j] by delta changes c[k] by 2*delta*f[k-j] + delta^2 (for circular).
    Actually, c = ifft(|fft(f)|^2). If we change f[j] to f[j]+delta, the new FFT is F + delta*exp(-2pi i j k/n).
    This is complex. Instead, let's just recompute FFT for each candidate update.
    To speed up, we can precompute the current autoconvolution and use the linearity of the change.
    
    Actually, a simpler approach: for each bin j, try a few candidate values (e.g., scale by factors),
    recompute the score, and pick the best. This is coordinate descent with line search.
    """
    n = len(f0)
    f = f0.copy().astype(float)
    # Normalize to sum 1
    s = f.sum()
    if s > 0:
        f = f / s
    
    best_score = compute_score(f)
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for j in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try scaling f[j] by various factors
            # The optimal f[j] given others can be found by 1D optimization
            # c depends on f[j] quadratically: c[k] = sum_i f[i]f[k-i]
            # Let f = f' + delta*e_j where f' has f[j]=0
            # Then c[k] = c'_k + delta*(f'[k-j] + f'[j-k]) + delta^2
            # where c' is autoconv of f'
            
            # Precompute f' (with f[j] set to 0)
            f_prime = f.copy()
            f_prime[j] = 0.0
            
            # Autoconv of f'
            c_prime = fft_autoconv(f_prime)
            
            # The cross term: sum_i f'[i] * f'[k-i] where one index is j
            # c[k] = c'_k + delta * (sum_i f'[i] * delta_{k-i,j} + sum_i f'[i] * delta_{k-i,j}) 
            # Wait, autoconv c[k] = sum_i f[i] f[k-i]
            # If f = f' + delta*e_j, then f[i]f[k-i] = f'[i]f'[k-i] + delta*f'[i]*delta_{k-i,j} + delta*delta_{i,j}*f'[k-i] + delta^2*delta_{i,j}*delta_{k-i,j}
            # = f'[i]f'[k-i] + delta*f'[k-j] (when k-i=j, i=k-j) + delta*f'[k-j] (when i=j, k-i=k-j) + delta^2 (when i=j and k-i=j, i.e., k=2j)
            # So c[k] = c'_k + 2*delta*f'[k-j] + delta^2 * delta_{k,2j}
            
            # We want to minimize max_k c[k] / (sum f)^2 * n
            # sum f = sum f' + delta = s' + delta where s' = sum f'
            # This is a 1D optimization problem in delta >= 0
            
            # Let's try a few candidate values
            current_fj = f[j]
            # Try: 0, current*0.5, current*0.75, current*1.0, current*1.25, current*1.5, current*2.0
            # Also try values based on the cross term
            candidates = [0.0, current_fj * 0.5, current_fj * 0.75, current_fj, current_fj * 1.25, current_fj * 1.5, current_fj * 2.0]
            # Add some small values
            if current_fj > 0:
                candidates.append(current_fj * 0.1)
                candidates.append(current_fj * 0.25)
            
            best_delta = current_fj
            best_local_score = best_score
            
            for delta in candidates:
                f_new = f_prime.copy()
                f_new[j] = delta
                # Ensure non-negative
                f_new = np.maximum(f_new, 0)
                # Normalize to sum 1
                s_new = f_new.sum()
                if s_new <= 0:
                    continue
                f_new = f_new / s_new
                score_new = compute_score(f_new)
                if score_new < best_local_score:
                    best_local_score = score_new
                    best_delta = delta
            
            if best_delta != current_fj:
                f[j] = best_delta
                # Renormalize
                s = f.sum()
                if s > 0:
                    f = f / s
                best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return f, best_score

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial guess: Constant function
    f_best = np.ones(n) / n  # Normalize to sum 1
    score_best = compute_score(f_best)
    
    # Multi-start with random initializations
    n_starts = 5
    np.random.seed(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization: mix of constant and random
        if i == 0:
            f_init = np.ones(n) / n
        elif i == 1:
            # Bump in the middle
            f_init = np.ones(n) / n
            f_init[n//4:3*n//4] *= 2.0
            f_init /= f_init.sum()
        elif i == 2:
            # Two bumps
            f_init = np.ones(n) / n
            f_init[n//6:n//3] *= 3.0
            f_init[2*n//3:5*n//6] *= 3.0
            f_init /= f_init.sum()
        elif i == 3:
            # Random positive
            f_init = np.random.exponential(1.0, n)
            f_init /= f_init.sum()
        else:
            # Random with smoothing
            f_init = np.random.exponential(1.0, n)
            # Simple moving average
            k = 10
            f_init = np.convolve(f_init, np.ones(k)/k, mode='same')
            f_init = np.maximum(f_init, 0)
            f_init /= f_init.sum()
        
        # Coordinate descent
        f_opt, score_opt = spectral_coordinate_descent(f_init, max_passes=50, deadline=deadline - 15)
        
        if score_opt < score_best:
            f_best = f_opt.copy()
            score_best = score_opt
    
    # Final polish on the best
    if time.time() < deadline - 10:
        f_final, score_final = spectral_coordinate_descent(f_best, max_passes=100, deadline=deadline - 5)
        if score_final < score_best:
            f_best = f_final
            score_best = score_final
    
    # Ensure non-negativity and normalize
    f_best = np.maximum(f_best, 0.0)
    s = f_best.sum()
    if s > 0:
        f_best = f_best / s
    
    return list(f_best)

if __name__ == "__main__":
    # Quick test
    f = construct()
    print(f"Score: {compute_score(np.array(f))}")
