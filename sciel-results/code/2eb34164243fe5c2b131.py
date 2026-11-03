"""Combined program: Discrete Gradient Descent with Soft-Max Surrogate."""
import numpy as np
import time
import random

N = 400
_M = 1
while _M < 2 * N - 1:
    _M *= 2

_FREQS = np.arange(_M // 2 + 1)
# Precompute basis for FFT? Not strictly needed if we use np.fft directly,
# but we need convolution for gradient.

def _compute_autoconv(h):
    H = np.fft.rfft(h, _M)
    ac = np.fft.irfft(H * H, _M)
    return ac[:2 * N - 1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_autoconv(h)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _soft_max_abs_and_grad(ac, beta):
    """
    Compute soft-max of |ac| and its gradient w.r.t ac.
    S = (1/beta) log sum(exp(beta * |ac|))
    dS/dac_k = sign(ac_k) * exp(beta * |ac_k|) / sum(exp(beta * |ac|))
    """
    abs_ac = np.abs(ac)
    # Shift to avoid overflow
    max_abs = np.max(abs_ac)
    exp_terms = np.exp(beta * (abs_ac - max_abs))
    sum_exp = np.sum(exp_terms)
    
    S = (1.0 / beta) * (np.log(sum_exp) + beta * max_abs)
    
    # Weights for gradient
    weights = exp_terms / sum_exp
    grad_ac = np.sign(ac) * weights
    
    return S, grad_ac

def _gradient_objective(h, beta=10.0):
    """
    Compute gradient of J(h) = soft_max(|ac|) / (sum h)^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        # Return gradient pushing away from 0 sum
        return np.ones(N) * (1.0 if s >= 0 else -1.0)
    
    ac = _compute_autoconv(h)
    S, grad_ac = _soft_max_abs_and_grad(ac, beta)
    
    # grad_ac has length 2*N-1
    # We need grad_S_h = dS/dh
    # S = f(ac). dS/dh_i = sum_k (dS/dac_k) * (dac_k/dh_i)
    # dac_k/dh_i = h_{k-i} + h_{i-k} (with bounds)
    # This is convolution of h with grad_ac (and reversed grad_ac)
    
    # Let g = grad_ac
    # term1 = h * g (linear convolution)
    # term2 = h * g[::-1] (linear convolution)
    # grad_S_h = term1 + term2
    
    # Use FFT for convolution
    # g is length 2N-1. h is length N.
    # Convolution length is N + (2N-1) - 1 = 3N-2.
    # We only need the first N components?
    # Wait, dS/dh_i involves sum over k.
    # h * g gives c_n = sum_i h_i g_{n-i}.
    # We need c_i for i=0..N-1?
    # Let's check indices.
    # ac_k = sum_j h_j h_{k-j}.
    # d(ac_k)/d(h_i) = h_{k-i} + h_{i-k} (if indices valid).
    # dS/dh_i = sum_k grad_ac_k * (h_{k-i} + h_{i-k})
    # = sum_k grad_ac_k h_{k-i} + sum_k grad_ac_k h_{i-k}
    # Let l = k-i => k = l+i. Sum_l grad_ac_{l+i} h_l. This is cross-correlation or convolution?
    # It is (h * grad_ac)[i] if grad_ac is indexed appropriately?
    # Actually, sum_k g_k h_{k-i} is the convolution of h and g evaluated at i?
    # (h * g)[i] = sum_j h_j g_{i-j}.
    # Let j = k-i => k = j+i.
    # Sum_j h_j g_{j+i}. This is not standard convolution.
    # It is cross-correlation: <h, shift(g)>.
    # Alternatively, (h * g_rev)[i] = sum_j h_j g_rev_{i-j} = sum_j h_j g_{j-i}.
    # So sum_k g_k h_{k-i} = sum_j h_j g_{j+i} = (h * g_rev)[i]? No.
    # Let's just use the property:
    # dS/dh = h * grad_ac + h * grad_ac[::-1] ?
    # Let's verify with small N.
    # N=2. h=[h0, h1]. ac=[h0^2, 2h0h1, h1^2].
    # d(ac_1)/d(h0) = 2h1.
    # Formula: h_{1-0} + h_{0-1} (out of bounds) = h1.
    # So d(ac_1)/d(h0) = h1 + h_{-1}(0) = h1.
    # Wait, ac_1 = h0 h1 + h1 h0 = 2 h0 h1.
    # d/dh0 = 2 h1.
    # My formula gave h1. Missing factor?
    # ac_k = sum_{j=0}^{N-1} h_j h_{k-j}.
    # d/dh_i:
    # Term 1: j=i => h_{k-i}.
    # Term 2: k-j=i => j=k-i => h_i h_{k-i} derivative wrt h_i is h_{k-i}.
    # So if i != k-i (i.e., k != 2i), we get h_{k-i} + h_{k-i} = 2 h_{k-i}.
    # If k=2i, we get 2 h_i.
    # So d(ac_k)/d(h_i) = 2 h_{k-i} if k-i in [0, N-1] and k != 2i?
    # Actually, if k=2i, the two terms are the same, so it's 2 h_i.
    # So generally d(ac_k)/d(h_i) = 2 h_{k-i} if 0 <= k-i < N.
    # Wait, is it always 2?
    # ac_k = sum_j h_j h_{k-j}.
    # d/dh_i sum_j h_j h_{k-j} = h_{k-i} + h_{i-k} (where h_x=0 if x not in [0,N-1]).
    # If k-i is valid, h_{k-i} is term 1.
    # If i-k is valid, h_{i-k} is term 2.
    # If k-i is valid, then i-k = -(k-i).
    # So if k-i > 0, h_{i-k} = 0.
    # If k-i < 0, h_{k-i} = 0.
    # If k-i = 0, both are h_0? No, h_{k-i} = h_0, h_{i-k} = h_0.
    # So if k=2i, derivative is 2 h_i.
    # If k != 2i, derivative is h_{k-i} (if valid) or 0.
    
    # So dS/dh_i = sum_k grad_ac_k * d(ac_k)/d(h_i).
    # This is equivalent to convolution?
    # Let's use FFT to compute the gradient directly.
    # dS/dh = IFFT( IFFT(h) * IFFT(grad_ac) )? No.
    
    # Simpler: Just compute it in O(N^2) if N=400?
    # 400^2 = 160,000 ops. Very fast.
    # We can do this in the loop.
    
    grad_S_h = np.zeros(N)
    # grad_ac is length 2N-1
    for i in range(N):
        # sum over k
        # d(ac_k)/d(h_i) is non-zero only for k in [i, i+N-1]
        # Specifically, k-i must be in [0, N-1].
        # So k from i to i+N-1.
        # val = 2 * h_{k-i} if k==2i else h_{k-i}
        # Wait, if k=2i, h_{k-i} = h_i.
        # The derivative is 2 h_i.
        # If k != 2i, derivative is h_{k-i}.
        
        # Vectorized approach:
        # k indices: i .. i+N-1
        # h indices: 0 .. N-1
        # deriv = h[0:N]
        # if 2*i < 2*N-1:
        #    deriv[2*i - i] += h[i]  # add the second term
        #    deriv[2*i - i] = 2 * h[i] # actually just set to 2*h[i]
        
        # Let's just do the sum
        ks = np.arange(i, i + N)
        hs = h[0:N] # h_{k-i} where k-i = 0..N-1
        # Check if k=2i is in range
        # 2i is in [i, i+N-1] if i <= 2i <= i+N-1 => 0<=i and i<=N-1. Always true for i in 0..N-1.
        # So we always have the double term at k=2i.
        # deriv = hs.copy()
        # deriv[2*i - i] = 2 * hs[2*i - i] # hs[i]
        # deriv[i] = 2 * h[i]
        
        # grad_S_h[i] = sum(grad_ac[ks] * deriv)
        
        # This is O(N) per i, so O(N^2) total.
        # N=400, N^2=160k.
        # 10,000 steps => 1.6 billion ops. Might be too slow in Python.
        # Use NumPy vectorization for the whole gradient at once?
        
        pass
        
    # Let's use the convolution property properly.
    # dS/dh_i = sum_k g_k (h_{k-i} + h_{i-k})
    # = sum_k g_k h_{k-i} + sum_k g_k h_{i-k}
    # Let u = k-i. sum_u g_{u+i} h_u.
    # This is the cross-correlation of h and g, shifted.
    # np.correlate(h, g, mode='same')?
    
    # Actually, we can compute:
    # c1 = np.convolve(h, g, mode='full')
    # c2 = np.convolve(h, g[::-1], mode='full')
    # grad_S_h = c1[i] + c2[i]?
    
    # Let's test N=2.
    # h=[1,1]. ac=[1,2,1]. g=[g0,g1,g2].
    # dS/dh0 = g0*1 + g1*2 + g2*0 = g0 + 2g1.
    # dS/dh1 = g0*0 + g1*2 + g2*1 = 2g1 + g2.
    
    # c1 = convolve([1,1], [g0,g1,g2]) = [g0, g0+g1, g1+g2, g2]
    # c2 = convolve([1,1], [g2,g1,g0]) = [g2, g2+g1, g1+g0, g0]
    
    # We want c1[0] + c2[0]? g0 + g2. No.
    # We want g0 + 2g1.
    # c1[1] = g0+g1. c2[1] = g2+g1. Sum = g0+g2+2g1. No.
    
    # Let's look at indices again.
    # dS/dh_i = sum_k g_k h_{k-i} (term 1) + sum_k g_k h_{i-k} (term 2)
    # Term 1: k=i..i+N-1. h_{k-i} = h_0..h_{N-1}.
    # This is dot product of g[i:i+N] and h.
    # Term 2: k=i-N+1..i. h_{i-k} = h_0..h_{N-1}.
    # This is dot product of g[i-N+1:i+1] and h.
    
    # So grad_S_h[i] = np.dot(g[i:i+N], h) + np.dot(g[max(0,i-N+1):i+1], h)
    
    # This can be computed using sliding window dot products, or just O(N^2) loop if optimized in C (NumPy).
    # Since N=400, a Python loop of 400 iterations, each doing a NumPy dot of 400 elements, is 400*400 = 160k mults.
    # In Python, this loop overhead might be significant.
    # 10,000 steps * 160k ops = 1.6e9 ops.
    # If 1 op is 1ns, 1.6s. Acceptable.
    
    grad_S_h = np.zeros(N)
    # g is grad_ac
    for i in range(N):
        # Term 1
        idx1_start = i
        idx1_end = i + N
        if idx1_end > len(g):
             idx1_end = len(g)
        # g[idx1_start:idx1_end] corresponds to h[0 : idx1_end-idx1_start]
        # But h is length N.
        # g[i : i+N] dot h[0:N] is valid if i+N <= 2N-1 => i <= N-1.
        # So always valid.
        t1 = np.dot(g[i:i+N], h)
        
        # Term 2
        # k from i-N+1 to i.
        # h_{i-k}. Let j = i-k. k=i-j.
        # j from 0 to N-1.
        # g[i-j].
        # So dot of g[i-N+1 : i+1] (reversed?) with h?
        # g[i], g[i-1], ..., g[i-N+1].
        # This is g[i-N+1 : i+1][::-1] dot h[0:N].
        idx2_start = max(0, i - N + 1)
        idx2_end = i + 1
        # g[idx2_start : idx2_end] is g[i-N+1...i]
        # We need g[i], g[i-1]...
        # So reverse it.
        t2 = np.dot(g[idx2_start : idx2_end][::-1], h)
        
        grad_S_h[i] = t1 + t2
        
    # Now compute grad_J
    # J = S / D
    # dJ/dh_i = (dS/dh_i * D - S * dD/dh_i) / D^2
    # dD/dh_i = 2 * s
    
    D = s * s
    dD = 2.0 * s
    
    grad_J = (grad_S_h * D - S * dD) / (D * D)
    
    return grad_J

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        h = rng.choice([1.0, -1.0], N)
        s = np.sum(h)
        
        # Avoid sum=0
        if abs(s) < 1.0:
            h[0] = 1.0 if s >= 0 else -1.0
            s = np.sum(h)
            
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        T = 1.0
        beta = 10.0
        
        for step in range(5000):
            if time.time() > deadline - 2:
                break
                
            grad = _gradient_objective(h, beta)
            
            # Discrete Gradient Descent
            # Try to flip indices where gradient suggests improvement
            # With some noise
            
            # Select a random subset of indices to consider flipping
            # Or just pick one random index and decide based on gradient?
            # Directive: "update rule for each sign is derived from the derivative"
            # This implies we might update multiple signs.
            # But flipping all signs where grad suggests it might be unstable.
            # Let's pick a random index i.
            i = rng.integers(0, N)
            
            # Determine if flip is good
            # If h[i] == 1, flip to -1. Delta h = -2.
            # Delta J approx -2 * grad[i]
            # If h[i] == -1, flip to 1. Delta h = 2.
            # Delta J approx 2 * grad[i]
            
            if h[i] == 1.0:
                delta_J = -2.0 * grad[i]
            else:
                delta_J = 2.0 * grad[i]
                
            # Acceptance
            if delta_J < 0 or rng.random() < np.exp(-delta_J / T):
                h[i] = -h[i]
                s = np.sum(h)
                if abs(s) < 1e-9:
                    # Revert or fix
                    h[i] = -h[i]
                    s = np.sum(h)
                    continue
                    
                new_score = _score(h)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
            else:
                pass # Do nothing
                
            T *= 0.999 # Cooling
            
    # Final polish
    if time.time() < deadline - 2:
        h = best_h.copy()
        s = np.sum(h)
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            grad = _gradient_objective(h, 10.0)
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                if h[i] == 1.0:
                    delta_J = -2.0 * grad[i]
                else:
                    delta_J = 2.0 * grad[i]
                    
                if delta_J < 0:
                    h[i] = -h[i]
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        h[i] = -h[i]
                        s = np.sum(h)
                        continue
                    new_score = _score(h)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        h[i] = -h[i]
                        s = np.sum(h)
                        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h
