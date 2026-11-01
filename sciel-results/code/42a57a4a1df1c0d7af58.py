"""
Strategy: Gradient-based optimization with smooth approximations.
Instead of discrete bit-flips, we optimize a continuous vector y.
We define a differentiable objective J(y) = sum(exp(beta * |AC_sidelobes(y)|^2)) 
or similar, which approximates the max sidelobe but is smooth.
We use gradient descent to minimize J(y), then threshold y to get x in {-1, 1}.
This allows navigating the rugged landscape more effectively than greedy flips.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_autocorr(x):
    """Compute the full autocorrelation of x using FFT."""
    n = len(x)
    # Zero-pad to avoid circular convolution effects
    # We need linear autocorrelation, so pad to 2*n
    f = np.fft.fft(x, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return ac

def _soft_max_sidelobe(x, beta=10.0):
    """
    Compute a smooth approximation of the max absolute sidelobe.
    Uses the log-sum-exp trick: log(sum(exp(beta * |ac|^2))) / beta
    This is differentiable and approximates max(|ac|^2) as beta -> inf.
    We optimize the square to keep gradients simple, or just use |ac| directly.
    Let's use the soft-max of the squared magnitudes to be consistent with energy.
    """
    ac = _compute_autocorr(x)
    # Sidelobes are indices 1 to n-1
    sidelobes = ac[1:]
    # We want to minimize the peak. 
    # Objective: sum(exp(beta * sidelobes^2))
    # To avoid overflow, subtract max
    s2 = sidelobes * sidelobes
    max_s2 = np.max(s2)
    if max_s2 == 0:
        return 0.0
    # log-sum-exp
    log_sum_exp = max_s2 + np.log(np.sum(np.exp(beta * (s2 - max_s2))))
    return log_sum_exp / beta

def _objective_and_gradient(x, beta=10.0):
    """
    Compute the objective value and its gradient with respect to x.
    We use the chain rule through the FFT.
    """
    n = len(x)
    # Forward FFT
    f = np.fft.fft(x, n=2 * n)
    
    # Power spectrum P = f * conj(f)
    # Autocorrelation ac = IFFT(P).real[:n]
    # We want gradient of L = sum_k exp(beta * |ac[k]|^2) with respect to x.
    
    # Let's compute ac first for the value
    ac_full = np.fft.ifft(f * np.conj(f)).real
    ac = ac_full[:n]
    sidelobes = ac[1:]
    s2 = sidelobes * sidelobes
    max_s2 = np.max(s2)
    
    # Value
    if max_s2 > 0:
        log_sum_exp = max_s2 + np.log(np.sum(np.exp(beta * (s2 - max_s2))))
        val = log_sum_exp / beta
    else:
        val = 0.0
        # Gradient is zero if all sidelobes are zero (unlikely)
        return val, np.zeros_like(x)

    # Gradient calculation
    # L = (1/beta) * log( sum_{k=1}^{n-1} exp(beta * ac[k]^2) )
    # dL/d(ac[k]) = (1/beta) * ( exp(beta * ac[k]^2) * 2 * beta * ac[k] ) / ( sum ... )
    #              = 2 * ac[k] * exp(beta * (ac[k]^2 - max_s2)) / ( sum exp(...) )
    
    exp_terms = np.exp(beta * (s2 - max_s2))
    denom = np.sum(exp_terms)
    
    # dL/d(ac[k]) for k in 1..n-1
    dL_d_ac = np.zeros(n)
    dL_d_ac[1:] = 2.0 * sidelobes * exp_terms / denom
    
    # Now, d(ac)/dx.
    # ac is the real part of the first n elements of IFFT(FFT(x, 2n) * conj(FFT(x, 2n))).
    # This is a bit complex to derive analytically in one go for the full vector.
    # Alternative: Use complex step or finite difference? No, too slow.
    # Let's use the property that ac is a quadratic form x^T A x.
    # But FFT structure is specific.
    
    # Let's try a different differentiable objective that is easier to differentiate.
    # Minimize sum(ac[k]^4) ? Or just sum(ac[k]^2) for k!=0?
    # sum(ac[k]^2) is the total energy of sidelobes. Minimizing this is a standard "white noise" objective.
    # It doesn't directly minimize the PEAK, but often correlates.
    # However, the task asks to improve the "upper bound on C3", which is related to the peak.
    # The soft-max is better for peak.
    
    # Let's compute the gradient of the soft-max properly using automatic differentiation logic.
    # ac = IFFT(P).real[:n]
    # P = |F|^2
    # F = FFT(x, 2n)
    
    # Let's use the linearity of FFT/IFFT.
    # ac = Re[ IFFT( F * conj(F) ) ]
    # Let G = IFFT( F * conj(F) )
    # ac = Re(G[:n])
    
    # We need dL/dx.
    # L = f(ac)
    # dL/dx = sum_k dL/d(ac[k]) * d(ac[k])/dx
    
    # d(ac)/dx is the Jacobian.
    # ac is linear in P? No, P is quadratic in x.
    # This gradient is expensive to compute exactly via FFT chain rule without care.
    
    # Simpler Gradient:
    # Use the fact that for real x, the gradient of sum_k w_k * ac[k] is 2 * (w * x) * (w * x) convolved?
    # Actually, if L = sum_k w_k * ac[k], then dL/dx = 2 * (w * x) where * is circular convolution?
    # For linear autocorrelation, it's slightly different due to zero-padding.
    
    # Let's use a simpler objective: Sum of squares of sidelobes.
    # L = sum_{k=1}^{n-1} ac[k]^2
    # This is equivalent to minimizing the total sidelobe energy.
    # Gradient of L w.r.t x:
    # L = x^T Q x, where Q is a matrix representing the sidelobe energy.
    # In frequency domain:
    # ac = IFFT(|F|^2).real
    # sum ac[k]^2 (k=1..n-1) is related to the power in the non-DC frequencies of the power spectrum?
    # Actually, Parseval's theorem says sum |ac[k]|^2 = (1/2n) sum |P[k]|^2.
    # Minimizing total sidelobe energy is equivalent to making the power spectrum P as flat as possible (except DC).
    # This is a well-known problem: Flat Spectrum Sequence Design.
    # We can minimize L = sum_{k=1}^{n-1} ac[k]^2.
    # Gradient of L w.r.t x:
    # L = sum_{j=1}^{n-1} (sum_{i} x[i] x[i+j])^2  ... this is messy.
    
    # Let's stick to the frequency domain gradient for Total Energy.
    # L_energy = sum_{k=1}^{n-1} ac[k]^2
    # We know that sum_{k=0}^{n-1} ac[k]^2 = (1/N) sum_{m=0}^{N-1} |F[m]|^4 ? No.
    # Let's just use the gradient of the soft-max approximation via finite differences on the gradient? No.
    
    # Let's use the gradient of the "Total Sidelobe Energy" which is easy to compute in frequency domain.
    # Objective: E = sum_{k=1}^{n-1} ac[k]^2
    # We want to minimize E.
    # It can be shown that the gradient of E w.r.t x is:
    # grad_x E = 2 * IFFT( |F|^2 * (1 - delta_0) ) ? No.
    
    # Let's use a known result:
    # If we want to minimize the peak, soft-max is good.
    # If we want to minimize the sum of squares, we can use:
    # grad_x (sum ac[k]^2) = 2 * (ac * x) ? No, that's for circular.
    
    # Let's use a robust method:
    # 1. Compute ac.
    # 2. Compute weights w_k = exp(beta * ac[k]^2) / sum(exp(...)) for k=1..n-1.
    # 3. The gradient of the soft-max log-sum-exp is:
    #    grad_L = sum_k w_k * grad(ac[k])
    #    grad(ac[k]) = 2 * (x * x shifted) ... 
    #    Actually, ac[k] = sum_j x[j] x[j+k].
    #    d(ac[k])/dx[i] = x[i+k] + x[i-k] (with boundary conditions).
    #    So grad_L[i] = sum_{k=1}^{n-1} w_k * (x[i+k] + x[i-k])
    #    This is a convolution!
    #    grad_L = 2 * (w * x) where * is linear convolution, restricted to valid indices.
    
    # Let's implement this.
    # w is a vector of length n-1 (for lags 1..n-1).
    # We need to convolve w with x.
    # Note: ac[k] is symmetric, so w[k] = w[-k].
    # Let's create a full symmetric weight vector W of length 2n-1?
    # Or just use np.convolve.
    
    # w_lags = exp_terms / denom (length n-1)
    # We need to compute sum_{k=1}^{n-1} w[k] * (x[i+k] + x[i-k])
    # This is equivalent to 2 * (x * w_full)[i] where w_full is symmetric.
    
    # Construct w_full for linear convolution
    # w_full[k] = w[k] for k=1..n-1
    # w_full[-k] = w[k] for k=1..n-1
    # w_full[0] = 0 (since we exclude k=0 from sidelobes)
    
    # Using np.convolve(x, w_full, mode='full') gives length 2n-1.
    # The valid part for x indices 0..n-1 corresponds to the middle part.
    
    # Let's build w_full
    w_full = np.zeros(2 * n - 1)
    w_full[n-1:n-1+n-1] = exp_terms  # positive lags
    w_full[n-1-(n-1):n-1] = exp_terms[::-1] # negative lags
    # w_full[n-1] is 0
    
    # Convolve
    # conv = np.convolve(x, w_full, mode='full')
    # The gradient at index i is 2 * conv[i + n - 1] ?
    # Let's check indices.
    # (x * w_full)[m] = sum_j x[j] w_full[m-j]
    # We want sum_k w[k] (x[i+k] + x[i-k])
    # Let m = i + n - 1.
    # w_full[k] is at index n-1+k.
    # w_full[-k] is at index n-1-k.
    # So term x[i+k] w[k] -> x[j] w_full[m-j] where j=i+k, m-j = i+n-1-i-k = n-1-k.
    # Wait, w_full[n-1-k] is w[k]. Correct.
    # Term x[i-k] w[k] -> x[j] w_full[m-j] where j=i-k, m-j = i+n-1-i+k = n-1+k.
    # w_full[n-1+k] is w[k]. Correct.
    # So grad_L[i] = 2 * conv[i + n - 1].
    
    conv = np.convolve(x, w_full, mode='full')
    grad = 2.0 * conv[n-1 : n-1+n]
    
    return val, grad

def construct():
    """
    Construct the signed step function.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Initialize
    # Random start
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], N)
    
    # Continuous variable for optimization
    y = x.copy()
    
    best_x = x.copy()
    best_score = _soft_max_sidelobe(best_x, beta=10.0)
    
    # Gradient Descent Loop
    lr = 0.1
    beta = 10.0
    patience = 0
    
    while time.time() < deadline - 10:
        # Compute objective and gradient
        obj, grad = _objective_and_gradient(y, beta)
        
        # Update y
        y = y - lr * grad
        
        # Threshold to get discrete x
        x = np.sign(y)
        x[x == 0] = 1.0
        
        # Evaluate discrete score
        score = _soft_max_sidelobe(x, beta)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            patience = 0
            # Increase beta slightly to sharpen the peak focus
            beta = min(beta * 1.05, 100.0)
            # Reduce learning rate as we get closer
            lr *= 0.95
        else:
            patience += 1
            # If stuck, perturb y
            if patience % 50 == 0:
                y += rng.normal(0, 0.1, N)
                # Reset beta? Maybe keep it.
            
        # Check time
        if time.time() > deadline - 5:
            break

    # Final local search polish with greedy flips if time permits
    if time.time() < deadline - 2:
        x_current = best_x
        ac = _compute_autocorr(x_current)
        max_side = np.max(np.abs(ac[1:]))
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                new_x = x_current.copy()
                new_x[i] = -new_x[i]
                new_ac = _compute_autocorr(new_x)
                new_max_side = np.max(np.abs(new_ac[1:]))
                
                if new_max_side < max_side - 1e-9:
                    x_current = new_x
                    ac = new_ac
                    max_side = new_max_side
                    improved = True

        if max_side < best_score: # Note: best_score was soft-max, max_side is hard max. 
                                 # They are comparable if beta is high.
                                 # Actually, let's just return the best discrete found.
                                 # We should track the best hard max score too.
                                 pass
                                 
    # We should return the x that minimizes the hard max sidelobe, as that's the metric.
    # Let's re-evaluate best_x and x_current with hard max.
    
    score_best_x = np.max(np.abs(_compute_autocorr(best_x)[1:]))
    score_x_current = np.max(np.abs(_compute_autocorr(x_current)[1:])) if 'x_current' in locals() else float('inf')
    
    if score_x_current < score_best_x:
        return x_current
    else:
        return best_x

# EVOLVE-BLOCK-END
