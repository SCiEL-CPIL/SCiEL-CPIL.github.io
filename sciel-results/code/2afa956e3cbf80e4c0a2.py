import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024  # next power of 2 >= 2*N-1 = 1199
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def get_autoconv(h):
        """Return raw autoconvolution array."""
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1]
    
    # Log-sum-exp smooth approximation of max
    def smooth_max(c, temp=0.01):
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
    
    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)
    
    # Evaluate uniform baseline
    h_uniform = np.ones(N)
    best_val = fast_metric(h_uniform)
    best_h = h_uniform.copy()
    
    # Phase 1: Try structured candidates
    candidates = []
    i = np.arange(N)
    center = N / 2 - 0.5
    
    candidates.append(np.ones(N))
    
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    width = 0.3
    sinc_h = np.sinc((i - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)
    
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: IPF-style iterative proportional fitting to flatten autoconvolution
    # Strategy: treat h as a probability distribution, iteratively adjust to reduce
    # L1 concentration of autoconvolution (which correlates with peak reduction)
    if elapsed() < DEADLINE - 60:
        h_ipf = best_h.copy()
        s = np.sum(h_ipf)
        if s > 1e-15:
            h_ipf = h_ipf / s  # normalize to probability
            current_val = fast_metric(h_ipf * s)  # scale-invariant, so just use normalized
            
            ipf_iters = 200
            for it in range(ipf_iters):
                if elapsed() > DEADLINE - 50:
                    break
                
                # Compute autoconvolution of normalized h
                c = get_autoconv(h_ipf)
                s2 = np.sum(h_ipf) ** 2
                if s2 < 1e-15:
                    break
                
                # Target: uniform distribution over lags (flat autoconvolution)
                # L1 norm of deviation from uniform indicates concentration
                c_norm = c / np.sum(c)
                target = np.ones(2*N-1) / (2*N-1)
                
                # Compute gradient-like signal: where autoconv is above target
                # We want to reduce h at positions that contribute to high c values
                # and increase h at positions contributing to low c values
                
                # For each position i, compute its contribution to the autoconvolution
                # c[k] = sum_i h[i]*h[i+k]
                # Contribution of h[i] to peak: sum over k where c[k] is high of h[i+k]
                
                # Simpler approach: use multiplicative update
                # h_new[i] = h[i] * exp(-alpha * (conv_contribution[i] - mean))
                
                # Compute for each i, the weighted sum of c values it participates in
                contrib = np.zeros(N)
                for k in range(1, N):
                    # c[k] involves h[i]*h[i+k] for i=0..N-k-1
                    # and c[2N-1-k] involves h[i]*h[i-k] for i=k..N-1
                    valid = slice(0, N-k)
                    contrib[:N-k] += c[k] * h_ipf[k:2*N-k-1]
                    valid2 = slice(k, N)
                    contrib[k:] += c[2*N-1-k] * h_ipf[:N-k]
                
                # Also c[0] = sum h[i]^2
                contrib += c[0] * h_ipf
                
                # Normalize contribution
                contrib_mean = np.mean(contrib)
                contrib_std = np.std(contrib) + 1e-10
                
                # Multiplicative update: reduce where contribution is high
                alpha = 0.1
                update = np.exp(-alpha * (contrib - contrib_mean) / contrib_std)
                h_new = h_ipf * update
                h_new = np.maximum(h_new, 1e-10)
                
                # Renormalize to maintain sum
                h_new = h_new / np.sum(h_new)
                
                # Check if this improves the metric
                v_new = fast_metric(h_new * s)
                if v_new < current_val:
                    h_ipf = h_new
                    current_val = v_new
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h_ipf * s
                else:
                    # Try smaller step
                    alpha_small = alpha * 0.1
                    update_small = np.exp(-alpha_small * (contrib - contrib_mean) / contrib_std)
                    h_new_small = h_ipf * update_small
                    h_new_small = np.maximum(h_new_small, 1e-10)
                    h_new_small = h_new_small / np.sum(h_new_small)
                    v_small = fast_metric(h_new_small * s)
                    if v_small < current_val:
                        h_ipf = h_new_small
                        current_val = v_small
                        if current_val < best_val:
                            best_val = current_val
                            best_h = h_ipf * s
    
    # Phase 3: Cosine-parameterized Remez-like refinement
    if elapsed() < DEADLINE - 40:
        K = 10  # number of cosine harmonics
        
        def cheb_to_h(coeffs):
            h = np.full(N, coeffs[0])
            for k in range(1, K+1):
                h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
            return np.maximum(h, 0.0)
        
        def fit_cosine(h, K_order):
            A = np.zeros((N, K_order + 1))
            A[:, 0] = 1.0
            for k in range(1, K_order + 1):
                A[:, k] = np.cos(2 * np.pi * k * i / N)
            coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
            return coeffs
        
        def get_conv_from_coeffs(coeffs):
            h = cheb_to_h(coeffs)
            s = np.sum(h)
            if s < 1e-15:
                return np.zeros(2*N-1), 0
            c = get_autoconv(h)
            return c, s
        
        coeffs = fit_cosine(best_h, K)
        
        for remez_iter in range(40):
            if elapsed() > DEADLINE - 30:
                break
            
            c, s = get_conv_from_coeffs(coeffs)
            if s < 1e-15:
                break
            
            max_idx = np.argmax(c)
            max_val = c[max_idx]
            
            h = cheb_to_h(coeffs)
            k = max_idx
            if k >= N:
                k = 2*N - 1 - k
            
            grad = np.zeros(K + 1)
            for j in range(K + 1):
                g = 0.0
                for ii in range(N):
                    if ii + k < N:
                        g += np.cos(2*np.pi*j*ii/N) * h[ii+k] + h[ii] * np.cos(2*np.pi*j*(ii+k)/N)
                    if ii - k >= 0 and k > 0:
                        g += np.cos(2*np.pi*j*ii/N) * h[ii-k] + h[ii] * np.cos(2*np.pi*j*(ii-k)/N)
                grad[j] = g / (s * s)
            
            lr = 0.01 * max_val / (np.max(np.abs(grad)) + 1e-15)
            lr = min(lr, 0.1)
            
            new_coeffs = coeffs - lr * grad
            new_h = cheb_to_h(new_coeffs)
            
            if np.sum(new_h) > 1e-15:
                new_val = fast_metric(new_h)
                if new_val < best_val:
                    best_val = new_val
                    best_h = new_h.copy()
                    coeffs = new_coeffs
                else:
                    new_coeffs = coeffs - lr * 0.1 * grad
                    new_h = cheb_to_h(new_coeffs)
                    if np.sum(new_h) > 1e-15:
                        new_val = fast_metric(new_h)
                        if new_val < best_val:
                            best_val = new_val
                            best_h = new_h.copy()
                            coeffs = new_coeffs
    
    # Phase 4: Gradient-based optimization with smooth max
    def objective(x, temp=0.01):
        if elapsed() > DEADLINE - 15:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)
    
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)
    
    for temp in [0.02, 0.01, 0.005]:
        if elapsed() > DEADLINE - 20:
            break
        try:
            result = minimize(
                lambda x: objective(x, temp), x0, method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 250}
            )
            if result.success or result.fun < 1e9:
                h_opt = np.exp(result.x - np.max(result.x))
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)
        except Exception:
            pass
    
    # Phase 5: Coordinate descent
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    no_improve = 0
    max_no_improve = 4000
    
    while elapsed() < DEADLINE - 25 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1
        
        if no_improve % 150 == 0 and no_improve > 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 6: Fine-tune with small paired moves
    if elapsed() < DEADLINE - 5:
        step_frac = 0.003
        for _ in range(800):
            if elapsed() > DEADLINE - 3:
                break
            ii, jj = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[ii] += delta
            new_h[jj] -= delta
            if new_h[ii] >= 0 and new_h[jj] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
