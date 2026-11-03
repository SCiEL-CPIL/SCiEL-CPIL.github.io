"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

New approach: Smoothed Convex Optimization via Softmax Gradient Descent.
Instead of heuristic peak shaving, we minimize a smooth surrogate of the
max autoconvolution value using the Log-Sum-Exp trick. This allows us to
use gradient descent to find a global minimum of the smoothed objective,
which often leads to flatter, more optimal autoconvolution profiles.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def softmax_max(conv, beta):
    """
    Compute the softmax-weighted sum of the convolution, which approximates
    the max as beta increases.
    Returns the softmax probabilities P and the soft-max value.
    """
    # Shift to avoid overflow
    c_max = np.max(conv)
    exp_conv = np.exp(beta * (conv - c_max))
    sum_exp = np.sum(exp_conv)
    if sum_exp == 0:
        return np.zeros_like(conv), c_max
    P = exp_conv / sum_exp
    # The soft-max value is c_max + (1/beta) * log(sum_exp)
    # But for gradient calculation, we usually just use P to weight the derivatives.
    return P, c_max

def gradient_descent_softmax(f_init, start_time, max_time, beta=50.0, lr=0.01, momentum=0.9):
    """
    Minimize the normalized soft-max of the autoconvolution using gradient descent.
    Objective: J(f) = Softmax(beta * conv(f)) / (sum(f))^2
    Since J is scale invariant, we fix sum(f) = N during updates.
    """
    f = f_init.copy()
    s = np.sum(f)
    f = f * (N / s)
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    v = np.zeros(N)  # Momentum
    
    # Adaptive learning rate
    current_lr = lr
    
    for iteration in range(20000):
        if time.time() - start_time > max_time:
            break
        
        # 1. Compute Convolution
        conv = np.convolve(f, f, mode='full')
        
        # 2. Compute Softmax Probabilities
        # We use a moderate beta. Too high -> noisy gradient, too low -> smooth but inaccurate.
        P, c_max = softmax_max(conv, beta)
        
        # 3. Compute Gradient of the numerator: d(softmax)/df[j]
        # d(conv[k])/df[j] = f[k-j] (if valid)
        # grad_num[j] = sum_k P[k] * f[k-j]
        # This is equivalent to the cross-correlation of P and f.
        # Note: conv is 'full' mode, length 2N-1.
        # We need to map conv indices to f indices.
        # conv[k] = sum_i f[i] f[k-i]
        # d conv[k] / d f[j] = f[k-j] if 0 <= k-j < N else 0.
        
        # Efficient calculation:
        # grad_num[j] = sum_k P[k] f[k-j]
        # Let m = k-j => k = j+m.
        # grad_num[j] = sum_m P[j+m] f[m]
        # This is the cross-correlation of P and f, evaluated at lags corresponding to j.
        # Specifically, if we pad P to size 2N-1, we can use np.correlate or similar.
        # However, P is defined on the range of conv (length 2N-1).
        # f is length N.
        
        # Let's compute grad_num directly using FFT for speed? 
        # Or just vectorized loop? N=600 is small. 2N-1 = 1199.
        # A double loop is 600 * 1199 ~ 700k ops. Very fast.
        
        grad_num = np.zeros(N)
        # P is length 2N-1
        # For each j in 0..N-1:
        #   sum over k where 0 <= k-j < N  =>  j <= k < j+N
        #   grad_num[j] += P[k] * f[k-j]
        
        # Vectorized approach:
        # Create a matrix M where M[j, k] = P[k] * f[k-j] if valid else 0.
        # This is basically a Toeplitz-like structure.
        # We can use np.correlate(P, f, mode='valid')? No, P is longer.
        
        # Let's stick to a simple loop for clarity and correctness, it's fast enough.
        # Actually, let's use FFT for the cross-correlation part if possible.
        # grad_num is the cross-correlation of P and f, but P is length 2N-1.
        # Let's just do the loop.
        
        # Optimization: Precompute valid k ranges?
        # Or use np.convolve on reversed f?
        # Let f_rev = f[::-1]
        # np.convolve(P, f_rev, mode='valid')?
        # conv(P, f_rev)[j] = sum_k P[k] f_rev[j-k] = sum_k P[k] f[N-1-(j-k)] = sum_k P[k] f[N-1-j+k]
        # This doesn't map directly to sum_k P[k] f[k-j].
        
        # Let's just use the explicit sum.
        # grad_num[j] = sum_{k=j}^{j+N-1} P[k] f[k-j]
        # We can compute this for all j using a sliding window sum?
        # No, the weights P[k] change.
        
        # Let's use FFT.
        # We want C[j] = sum_k P[k] f[k-j].
        # Let g[m] = f[m] for m in 0..N-1, 0 else.
        # Let h[k] = P[k] for k in 0..2N-2, 0 else.
        # C[j] = sum_k h[k] g[k-j].
        # This is the cross-correlation of h and g.
        # Cross-correlation can be computed via FFT:
        # corr = ifft(fft(h) * conj(fft(g)))
        # We need the part of corr corresponding to lags j=0..N-1.
        
        # Pad h and g to length 4N to avoid circular convolution issues.
        size_fft = 4 * N
        H = np.fft.fft(h, n=size_fft) if False else np.fft.fft(np.concatenate([P, np.zeros(size_fft - len(P))]), n=size_fft)
        G = np.fft.fft(np.concatenate([f, np.zeros(size_fft - len(f))]), n=size_fft)
        
        # Cross correlation: ifft( H * conj(G) )
        corr = np.fft.ifft(H * np.conj(G)).real
        # The linear cross-correlation is in the first 2N-1 elements?
        # Let's verify indices.
        # (h * g)[n] = sum_k h[k] g[n-k]
        # We want sum_k h[k] g[k-j]. Let m = k-j => k = m+j.
        # sum_m h[m+j] g[m].
        # This is (g * h)[j]? No.
        # Standard correlation: R[j] = sum_k h[k] g[k-j].
        # In FFT: ifft( H * conj(G) ) gives sum_k h[k] g[k-j] ?
        # Yes, ifft( H * conj(G) ) is the cross-correlation.
        # The result is circular. We padded to 4N, so linear part is valid.
        # The lag j corresponds to index j in the output?
        # Let's check: ifft( H * conj(G) )[j] = sum_k h[k] g[k-j] (circular).
        # With padding, it becomes linear.
        # So grad_num = corr[:N]
        
        grad_num = corr[:N]
        
        # 4. Compute Gradient of the full objective J = Num / Den
        # Num = SoftmaxVal (approximated by c_max + log(sum_exp)/beta, but for gradient we use P directly on derivatives)
        # Actually, the gradient of the soft-max function S = log(sum exp(beta c)) is beta * P * (dc/df).
        # So d(Num)/df = beta * P (weighted sum of dc/df).
        # Let's refine:
        # Objective L = log(sum exp(beta conv)) - 2 log(sum f)
        # dL/df = beta * P * (d conv / df) - 2 * (1/sum f) * 1
        # d conv / df is grad_num / beta? No.
        # grad_num above was sum P[k] f[k-j].
        # d(conv[k])/df[j] = f[k-j].
        # So sum_k (d S / d conv[k]) * (d conv[k] / df[j])
        # d S / d conv[k] = beta * P[k].
        # So term 1 is beta * grad_num.
        
        term1 = beta * grad_num
        term2 = 2.0 / np.sum(f)
        
        grad = term1 - term2
        
        # 5. Update with Momentum
        v = momentum * v - current_lr * grad
        f_new = f + v
        
        # 6. Project to non-negative
        f_new = np.maximum(f_new, 0.0)
        
        # 7. Normalize
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        # 8. Evaluate
        conv_new = np.convolve(f_new, f_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < best_val:
            best_val = val_new
            best_f = f_new.copy()
            
        # Accept if improved, or use momentum regardless? 
        # Gradient descent usually accepts if loss decreases.
        # If it increases, we might revert or dampen.
        # Let's do a simple line search or just accept if better.
        if val_new < compute_score(f):
            f = f_new
        else:
            # Revert and reduce learning rate
            current_lr *= 0.9
            # Reset momentum
            v *= 0.5
            
        # Increase beta slowly to sharpen the max approximation
        beta *= 1.001
        if beta > 1000:
            beta = 1000

    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial guess: Gaussian bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f_init = np.exp(-10 * (x / 0.25) ** 2)
    
    # Run Gradient Descent
    best_f, best_score = gradient_descent_softmax(
        f_init, start_time, TIME_LIMIT - 5, beta=50.0, lr=0.05, momentum=0.9
    )
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    # If it failed or returned inf, fallback to constant
    if best_score >= np.inf:
        best_f = np.ones(N)
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
