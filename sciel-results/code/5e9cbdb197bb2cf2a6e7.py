import numpy as np
import time
import random
import scipy.optimize

N = 600
TIME_LIMIT = 115

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def logsumexp_obj_and_grad(x, eps):
    """
    Compute the log-sum-exp approximation of the max autoconvolution
    and its gradient with respect to x.
    f = softmax(x)
    c = f * f
    J = eps * log(sum(exp(c / eps)))
    """
    f = softmax(x)
    c = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    
    # Log-sum-exp trick for stability
    c_max = np.max(c)
    log_sum_exp = c_max + eps * np.log(np.sum(np.exp((c - c_max) / eps)))
    J = log_sum_exp
    
    # Gradient of J w.r.t c
    # p_k = exp(c_k/eps) / sum(exp(c_j/eps))
    # p_k = exp((c_k - c_max)/eps) / sum(exp((c_j - c_max)/eps))
    exp_terms = np.exp((c - c_max) / eps)
    p = exp_terms / np.sum(exp_terms)
    
    # Gradient of c w.r.t f
    # c_k = sum_i f_i f_{k-i}
    # d c_k / d f_j = 2 f_{k-j}
    # G_f = sum_k p_k * (2 f_{k-j})
    # This is 2 * (p * f) where * is convolution?
    # Let's check indices.
    # d c_k / d f_j = 2 f_{k-j} if 0 <= k-j < N
    # G_f[j] = sum_k p_k * 2 f_{k-j}
    # Let q = p.
    # G_f = 2 * (q * f_reversed)?
    # Convolution (q * f)_j = sum_k q_k f_{j-k}.
    # We need sum_k p_k f_{k-j}.
    # This is correlation.
    # In numpy, np.correlate(p, f, mode='full')?
    # Or just use FFT.
    # G_f[j] = 2 * sum_{k} p_k f_{k-j}
    # This is 2 * (p * f)[j] if we define convolution appropriately?
    # (p * f)[j] = sum_k p[k] f[j-k].
    # We want sum_k p[k] f[k-j].
    # This is (p * f_rev)[j]?
    # Let's just compute it directly or via FFT.
    # FFT method:
    # G_f is the gradient of the quadratic form.
    # Actually, G_f[j] = 2 * (p * f)[j] is NOT correct directly due to index reversal in convolution definition vs correlation.
    # Let's use the property: grad_f (sum_k p_k c_k) = sum_k p_k grad_f c_k.
    # grad_f c_k is a vector with 2 f_{k-j} at index j.
    # So G_f[j] = sum_k p_k 2 f_{k-j}.
    # This is 2 * (p * f)[j] if we flip p?
    # Let's compute via FFT to be safe.
    # G_f = 2 * ifft(fft(p) * fft(f)) ?
    # (p * f)_n = sum_k p_k f_{n-k}.
    # We want sum_k p_k f_{k-j}.
    # Let m = k-j => k = m+j.
    # sum_m p_{m+j} f_m.
    # This is correlation of p and f.
    # Correlation can be done by FFT: ifft(fft(p) * conj(fft(f)))?
    # Or simply:
    # G_f = 2 * np.fft.irfft(np.fft.rfft(p) * np.fft.rfft(f, n=2*N-1), n=2*N-1)
    # Wait, the convolution length is 2N-1.
    # p is length 2N-1? No, c is length 2N-1. p is length 2N-1.
    # f is length N.
    # We need G_f of length N.
    
    # Let's do it with linear algebra or simple loops if N is small, but N=600 is large.
    # FFT approach for G_f:
    # G_f[j] = 2 * sum_{k} p_k f_{k-j}
    # This is the cross-correlation of p and f.
    # corr[p, f][j] = sum_k p[k] f[k-j]
    # We can compute this as:
    # fft_p = rfft(p, n=2*N)
    # fft_f = rfft(f, n=2*N)
    # G_f_full = irfft(fft_p * fft_f, n=2*N) ?
    # (p * f)[n] = sum_k p[k] f[n-k].
    # We want sum_k p[k] f[k-j].
    # This is (p * f_rev)[j]?
    # Let f_rev[i] = f[N-1-i].
    # This is getting confusing.
    
    # Alternative:
    # G_f = 2 * (p * f) where * is circular convolution? No.
    # Let's use the explicit formula with FFT of padded signals.
    # We want the vector G of length N such that G[j] = 2 * sum_{k=0}^{2N-2} p_k f_{k-j} (with f out of bounds = 0).
    
    # Let's construct the gradient via:
    # grad_f_c = zeros((2N-1, N))
    # for k in range(2N-1):
    #   for j in range(max(0, k-N+1), min(N, k+1)):
    #     grad_f_c[k, j] = 2 * f[k-j]
    # G_f = p @ grad_f_c
    
    # This is O(N^2) which is 360,000 ops per iteration.
    # L-BFGS might take 100-1000 iterations. 360M ops. Might be slow in Python loops.
    # Vectorized NumPy:
    # We can use the fact that G_f[j] = 2 * sum_k p_k f_{k-j}
    # This is 2 * (p * f)[j] if we consider p and f as sequences and use appropriate convolution.
    # Actually, if we define convolution (a*b)[n] = sum_k a[k] b[n-k],
    # then (p*f)[j] = sum_k p[k] f[j-k].
    # We want sum_k p[k] f[k-j].
    # Let l = k-j => k = l+j.
    # sum_l p[l+j] f[l].
    # This is the correlation of p and f.
    # Correlation can be computed as:
    # corr = ifft(fft(p) * conj(fft(f))) ?
    # Let's check.
    # FFT of correlation is FFT(p) * conj(FFT(f)).
    # So G_f = 2 * irfft(rfft(p, n=2*N) * conj(rfft(f, n=2*N)), n=2*N)
    # We only need the first N elements?
    # The correlation output length is 2N-1.
    # The indices j range from 0 to N-1.
    # The correlation sum is over k.
    # For j=0, sum_k p_k f_k.
    # For j=N-1, sum_k p_k f_{k-(N-1)}.
    
    # Let's test this logic.
    fft_p = np.fft.rfft(p, n=2*N)
    fft_f = np.fft.rfft(f, n=2*N)
    corr_full = np.fft.irfft(fft_p * np.conj(fft_f), n=2*N)
    G_f = 2 * corr_full[:N]
    
    # Now map gradient from f to x
    # J(x) = M(softmax(x))
    # dJ/dx_j = sum_i (dM/df_i) (df_i/dx_j)
    # df_i/dx_j = f_i (delta_ij - f_j)
    # dJ/dx_j = f_j ( sum_i G_f_i - sum_i G_f_i f_i )
    # Let S = sum_i G_f_i
    # Let T = sum_i G_f_i f_i
    # grad_x = f * (S - T) ... wait.
    # sum_i G_f_i f_i (delta_ij - f_j) = G_f_j - f_j sum_i G_f_i f_i
    # So grad_x_j = G_f_j - f_j * (G_f @ f)
    
    Gf_dot_f = np.dot(G_f, f)
    grad_x = G_f - f * Gf_dot_f
    
    return J, grad_x

def optimize_softmax(eps, max_iter=1000, time_limit=20):
    x0 = np.zeros(N)
    start_time = time.time()
    
    # Use L-BFGS-B
    # We need to ensure we don't exceed time limit inside minimize
    # scipy minimize doesn't have a time limit, but we can use a callback to stop
    stop_flag = [False]
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
            return True
        return False
        
    res = scipy.optimize.minimize(
        logsumexp_obj_and_grad,
        x0,
        args=(eps,),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'maxfun': max_iter, 'disp': False},
        callback=callback
    )
    
    f_opt = softmax(res.x)
    score = compute_score(f_opt)
    return f_opt, score

def construct():
    start_time = time.time()
    
    best_f = None
    best_score = np.inf
    
    # Try different epsilons (annealing)
    epsilons = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005]
    
    x_current = np.zeros(N)
    
    for eps in epsilons:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Initialize x from previous best if available, or zero
        # For the first one, use zero. For others, use the x that gave best f?
        # We don't have x from previous, we have f.
        # We can recover x from f? f = softmax(x) => x = log(f) + const.
        # If best_f is available, use x_init = log(best_f + 1e-10)
        
        if best_f is not None:
            x_init = np.log(best_f + 1e-10)
        else:
            x_init = np.zeros(N)
            
        # Optimize
        f_opt, score = optimize_softmax(eps, max_iter=500, time_limit=TIME_LIMIT - 5 - (time.time() - start_time))
        
        if score < best_score:
            best_score = score
            best_f = f_opt

    # Final polish with very small eps
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 5:
        x_init = np.log(best_f + 1e-10)
        f_opt, score = optimize_softmax(0.001, max_iter=200, time_limit=TIME_LIMIT - 5 - (time.time() - start_time))
        if score < best_score:
            best_f = f_opt
            best_score = score

    if best_f is None:
        best_f = np.ones(N) / N
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
