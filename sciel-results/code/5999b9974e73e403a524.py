import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 115  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for fast convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def logsumexp_objective(f, alpha):
    """
    Objective: log(sum(exp(alpha * (f*f))))
    This approximates max(f*f) as alpha -> inf.
    """
    f = np.asarray(f, dtype=np.float64)
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    # Shift to avoid overflow in exp
    max_conv = np.max(conv)
    if max_conv < 0: max_conv = 0
    # log(sum(exp(alpha * (conv - max_conv)))) + alpha * max_conv
    vals = alpha * (conv - max_conv)
    # Clip to avoid underflow/overflow issues in sum, though exp(-large) is 0
    # We only need the sum of exps
    log_sum = np.log(np.sum(np.exp(vals)))
    return log_sum + alpha * max_conv

def logsumexp_gradient(f, alpha):
    """
    Gradient of log(sum(exp(alpha * (f*f))))
    d/df_j log(sum_k exp(alpha * S_k)) 
    = (1/sum_k exp(alpha S_k)) * sum_k exp(alpha S_k) * alpha * dS_k/df_j
    dS_k/df_j = 2 * f_{k-j} (with bounds)
    """
    f = np.asarray(f, dtype=np.float64)
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    max_conv = np.max(conv)
    if max_conv < 0: max_conv = 0
    
    vals = alpha * (conv - max_conv)
    exp_vals = np.exp(vals)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp < 1e-15:
        return np.zeros(N)
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros(N)
    # We need sum_k weights[k] * alpha * 2 * f_{k-j}
    # This is a convolution-like operation.
    # grad[j] = 2 * alpha * sum_k weights[k] * f_{k-j}
    # Let w = weights. We want (w * f) but with indices k-j.
    # This is cross-correlation of w and f?
    # sum_k w_k f_{k-j}.
    # If we reverse w, it's convolution.
    # Let's just do it efficiently.
    # The range of k is 0 to 2N-2.
    # For a given j, k-j is in [0, N-1] => k is in [j, j+N-1].
    
    # Using FFT for the weighted sum?
    # sum_k w_k f_{k-j}
    # This is the cross-correlation of w and f.
    # np.correlate(w, f, mode='full')?
    # w has length 2N-1. f has length N.
    # We want the part where the result corresponds to shift j.
    
    # Simpler: loop over non-zero weights? No, weights are dense.
    # Use FFT.
    # We want g[j] = sum_k w[k] f[k-j]
    # This is equivalent to convolving w (reversed?) with f?
    # Let's check indices.
    # (f * w_rev)[j] = sum_i f[i] w_rev[j-i] = sum_i f[i] w[(2N-2)-(j-i)] ... messy.
    
    # Let's use the property:
    # grad = 2 * alpha * np.fft.irfft(np.fft.rfft(f) * np.conj(np.fft.rfft(w, n=2*N-1)), n=N) ?
    # No, w is real.
    # Let's just compute it directly if N is small? N=600. 600*1200 = 720k ops. Very fast.
    # But we do this many times.
    # FFT is better.
    
    # Cross-correlation of f and w:
    # C[j] = sum_k f[k] w[k+j] ? No.
    # We want sum_k w[k] f[k-j].
    # Let m = k-j => k = m+j.
    # sum_m w[m+j] f[m].
    # This is the cross-correlation of f and w, shifted.
    
    # np.correlate(f, w, mode='full') gives sum_i f[i] w[i+k].
    # We want sum_k w[k] f[k-j] = sum_k f[k-j] w[k].
    # Let i = k-j => k = i+j.
    # sum_i f[i] w[i+j].
    # This is exactly np.correlate(f, w, mode='full')[j + offset].
    
    # np.correlate(f, w, mode='full') has length N + (2N-1) - 1 = 3N-2.
    # The index j in the correlation sum sum_i f[i] w[i+j] corresponds to:
    # In 'full' mode, the output y has length len(f)+len(w)-1.
    # y[k] = sum_i f[i] w[i+k - (len(w)-1)] ? No.
    # SciPy correlate: c[k] = sum_n a[n] conj(b[n-k]).
    # Here a=f, b=w.
    # c[k] = sum_n f[n] w[n-k].
    # We want sum_m f[m] w[m+j].
    # Set n=m. We want w[n+j] = w[n - (-j)].
    # So k = -j.
    # So we need c[-j].
    # Since c is defined for k from -(len(w)-1) to len(f)-1.
    # c is indexed 0 to len(f)+len(w)-2.
    # c[0] corresponds to k=0?
    # Actually, easier to just use FFT multiplication.
    
    # Let's stick to the direct loop for safety and simplicity, N=600 is small.
    # 600 * 1200 = 720,000 multiplications.
    # In Python, this might be slow (milliseconds).
    # In NumPy vectorized:
    
    # We can construct a matrix? No.
    
    # Let's use FFT.
    # We want g = cross_corr(f, w).
    # g[j] = sum_k f[k] w[k+j] (with valid indices).
    # Using FFT:
    # F = fft(f, n=4*N)
    # W = fft(w, n=4*N)
    # G = F * conj(W)
    # g = ifft(G).real
    # The result g will have the correlation.
    # We need to extract the correct slice.
    
    n_fft = 4 * N
    F = np.fft.rfft(f, n=n_fft)
    W = np.fft.rfft(w, n=n_fft)
    G = F * np.conj(W)
    g_full = np.fft.irfft(G, n=n_fft)
    
    # The correlation sum sum_k f[k] w[k+j] is at index j in the circular correlation?
    # Linear correlation via FFT:
    # The result of ifft(fft(f)*conj(fft(w))) is the circular cross-correlation.
    # For linear cross-correlation, we need to handle the wrap-around.
    # Since we zero-padded to 4N, the linear correlation is contained in the first 3N-1 elements?
    # The correlation of length N and 2N-1 has length 3N-1.
    # The indices in g_full correspond to lags.
    # g_full[k] corresponds to lag k?
    # Let's verify with a small example mentally.
    # f=[1,0], w=[1,0,0].
    # corr[0] = 1*1 = 1.
    # corr[1] = 1*0 + 0*1 = 0.
    # corr[-1] = 0*1 = 0.
    
    # In the FFT result, the positive lags are at the beginning, negative at the end.
    # We want sum_m f[m] w[m+j] for j=0 to N-1.
    # This corresponds to positive lags 0 to N-1.
    # So we take g_full[0:N].
    
    grad = 2.0 * alpha * g_full[0:N]
    
    return grad

def optimize_lse(f_start, alpha=50.0, time_limit=10.0):
    """
    Optimize using L-BFGS-B with penalty for sum constraint.
    """
    f = f_start.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s > 0:
        f = f / s
    
    start_time = time.time()
    mu = 10.0
    
    def obj_pen(x):
        if time.time() - start_time > time_limit:
            return np.inf
        lse_val = logsumexp_objective(x, alpha)
        sum_x = np.sum(x)
        penalty = mu * (sum_x - 1.0) ** 2
        return lse_val + penalty
    
    def grad_pen(x):
        if time.time() - start_time > time.limit: # Typo in thought, fix in code
            return np.zeros(N)
        lse_grad = logsumexp_gradient(x, alpha)
        sum_x = np.sum(x)
        pen_grad = 2.0 * mu * (sum_x - 1.0) * np.ones(N)
        return lse_grad + pen_grad

    # We will run L-BFGS-B a few times, increasing mu if needed
    current_f = f
    for i in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Check constraint
        sum_f = np.sum(current_f)
        if abs(sum_f - 1.0) < 1e-4:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(obj_pen, current_f, jac=grad_pen, method='L-BFGS-B', 
                           bounds=[(0, None)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success or res.status == 0:
                current_f = res.x
                current_f = np.maximum(current_f, 0.0)
            else:
                # If it fails, maybe try a smaller step or just break
                pass
        except Exception:
            break
            
        # Increase penalty if constraint not met
        if abs(np.sum(current_f) - 1.0) > 1e-3:
            mu *= 10.0
        else:
            break

    return current_f

def construct():
    start_time = time.time()
    
    # Define diverse initial shapes
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N) / N)
    
    # 2. Triangular
    x = np.linspace(0, 1, N)
    tri = 1 - np.abs(2 * x - 1)
    starts.append(tri / np.sum(tri))
    
    # 3. Cosine taper
    cos = 0.5 * (1 + np.cos(2 * np.pi * x))
    starts.append(cos / np.sum(cos))
    
    # 4. Plateau
    plat = np.zeros(N)
    plat[N//4 : 3*N//4] = 1.0
    starts.append(plat / np.sum(plat))
    
    # 5. Gaussian-like
    g = np.exp(-((x - 0.5)**2) / (2 * 0.1**2))
    starts.append(g / np.sum(g))
    
    # 6. Random smooth
    np.random.seed(42)
    for _ in range(3):
        r = np.random.rand(N)
        r = np.convolve(r, np.ones(10)/10, mode='same')
        starts.append(r / np.sum(r))
        
    best_f = None
    best_score = np.inf
    
    # Try different alphas
    alphas = [20.0, 50.0, 100.0]
    
    for alpha in alphas:
        for i, start in enumerate(starts):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            # Time per optimization: ~10s
            f_opt = optimize_lse(start, alpha=alpha, time_limit=10.0)
            
            # Normalize
            s = np.sum(f_opt)
            if s < 1e-15:
                continue
            f_opt = f_opt / s
            
            score = compute_score(f_opt)
            if score < best_score:
                best_score = score
                best_f = f_opt
                # Print for debugging
                # print(f"Alpha {alpha}, Start {i}: Score {score}")
                
    if best_f is None:
        return [1.0/N] * N
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
