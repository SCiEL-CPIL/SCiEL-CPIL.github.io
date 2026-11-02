"""
Strategy: Direct Gradient-Based Optimization with Soft-Max Surrogate
Instead of parametric shapes (like two blocks), we optimize the 600 heights directly.
We use a smooth approximation of the maximum function (log-sum-exp) to make the 
objective differentiable, allowing L-BFGS-B to navigate the high-dimensional 
landscape effectively. This allows for complex, non-parametric shapes that 
distribute the autoconvolution mass more evenly, potentially lowering the peak.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
h = 0.5 / N  # Interval length

def _compute_autoconv(a):
    """Compute linear autoconvolution of a."""
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    return conv

def _compute_score_raw(a):
    """Compute the raw normalized max autoconvolution."""
    if np.sum(a) < 1e-12:
        return float('inf')
    conv = _compute_autoconv(a)
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    return max_conv / norm

def _soft_max_obj_and_grad(a, lam):
    """
    Compute the soft-max objective and its gradient.
    Objective: log(sum(exp(lam * conv_i))) / (lam * norm)
    This approximates max(conv_i) / norm as lam -> inf.
    We scale by 1/lam to keep the gradient magnitude reasonable.
    """
    if np.sum(a) < 1e-12:
        return 1e6, np.zeros_like(a)
    
    conv = _compute_autoconv(a)
    
    # For numerical stability in log-sum-exp
    # We want to approximate max(conv).
    # Let M = max(conv).
    # log(sum(exp(lam * (conv - M)))) + lam * M
    # To avoid overflow, we shift by max.
    
    # However, the normalization factor (sum a)^2 depends on a.
    # Let S = sum(a).
    # Score = max(conv) / (h^2 * S^2)
    # We minimize F(a) = log(sum(exp(lam * conv_i / (h^2 * S^2)))) / lam
    # Actually, it's easier to minimize the soft-max of the normalized values.
    
    S = np.sum(a)
    norm_factor = h**2 * S**2
    
    # If S is very small, the score is huge.
    if norm_factor < 1e-12:
        return 1e6, np.zeros_like(a)
        
    # Normalize conv by the current norm to make it scale-invariant-ish for the exp?
    # No, the problem is scale invariant. 
    # Let's fix the scale by normalizing a such that sum(a) = 1 before computing score?
    # No, L-BFGS-B works better if we don't project at every step, but we can 
    # include the normalization in the objective.
    
    # Let's use the objective: J(a) = log( sum( exp( lam * conv_i / (h^2 * S^2) ) ) ) / lam
    # This is approximately max(conv) / (h^2 * S^2) for large lam.
    
    val = conv / norm_factor
    max_val = np.max(val)
    
    # Stability: subtract max_val
    shifted = val - max_val
    exp_sum = np.sum(np.exp(lam * shifted))
    
    if exp_sum < 1e-300:
        obj = max_val
        # Gradient is roughly proportional to conv / norm_factor
        # d/da [ max(val) ] ~ conv / norm_factor (in the max direction)
        # For soft max:
        # p_i = exp(lam * shifted_i) / sum(exp(lam * shifted))
        # d obj / d a_j = sum_i (p_i * d val_i / d a_j)
        
        # We need d conv / d a and d norm / d a.
        # This is complex to derive analytically for FFT-based conv.
        # Let's use numerical gradient? No, L-BFGS-B is fast if we have analytic, 
        # but numerical might be okay for 600 dims if we use a decent step.
        # However, numerical gradient for 600 dims is 1200 FFTs. Too slow.
        
        # Let's try to derive the gradient.
        # conv = a * a (linear conv)
        # conv_k = sum_i a_i a_{k-i}
        # d conv_k / d a_j = a_{k-j} + a_{k-j} (if k-j != j) or 2 a_j (if k-j=j)
        # Actually, d conv_k / d a_j = a_{k-j} + a_{j-k} (since conv is symmetric)
        
        # norm = h^2 S^2
        # d norm / d a_j = 2 h^2 S
        
        # val_k = conv_k / norm
        # d val_k / d a_j = ( (d conv_k / d a_j) * norm - conv_k * (d norm / d a_j) ) / norm^2
        
        # obj = (1/lam) * log(sum(exp(lam * val_k)))
        # d obj / d a_j = sum_k [ p_k * d val_k / d a_j ]
        # where p_k = exp(lam * (val_k - max_val)) / sum(exp(lam * (val_k - max_val)))
        
        p = np.exp(lam * shifted)
        p /= np.sum(p)
        
        # We need to compute sum_k p_k * d val_k / d a_j for all j.
        # d val_k / d a_j = ( (a_{k-j} + a_{j-k}) * norm - conv_k * 2 h^2 S ) / norm^2
        # Note: indices for conv are 0 to 2N-2.
        # a is 0 to N-1.
        # a_{idx} is 0 if idx < 0 or idx >= N.
        
        # Let's vectorize this.
        # For each k, d conv_k / d a is a vector of length N.
        # This is a convolution-like operation.
        
        # Let's use the property:
        # d conv / da is related to cross-correlation.
        # Actually, let's just compute the gradient numerically with a small number of 
        # "smart" steps or use a simpler objective that is easier to differentiate.
        
        # Alternative: Minimize sum of squares of (conv - target) + penalty for max?
        # Or simply: Minimize log(sum(exp(lam * conv))) subject to sum(a)=1, a>=0.
        # If we fix sum(a)=1, norm is constant (h^2).
        # Then val = conv / h^2.
        # d val_k / d a_j = (1/h^2) * (a_{k-j} + a_{j-k})
        
        # This is much simpler!
        # Strategy: Enforce sum(a) = 1 via normalization inside the objective?
        # No, L-BFGS-B doesn't handle equality constraints well.
        # But we can normalize a at the start of each objective call?
        # If we normalize a -> a/S, then the gradient with respect to the unnormalized a
        # is scaled by 1/S.
        # Let a_norm = a / S.
        # obj(a) = SoftMax(conv(a_norm)).
        # d obj / d a = (1/S) * d obj / d a_norm.
        
        # Let's implement this path.
        
        a_norm = a / S
        conv_norm = _compute_autoconv(a_norm)
        # Since sum(a_norm)=1, norm_factor = h^2.
        val = conv_norm / (h**2)
        
        max_val = np.max(val)
        shifted = val - max_val
        exp_terms = np.exp(lam * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp < 1e-300:
            return 1e6, np.zeros_like(a)
            
        obj_val = (max_val + np.log(sum_exp) / lam)
        
        # Probabilities p_k
        p = exp_terms / sum_exp
        
        # Gradient of obj with respect to a_norm
        # d obj / d a_norm_j = sum_k p_k * d val_k / d a_norm_j
        # d val_k / d a_norm_j = (1/h^2) * (a_norm_{k-j} + a_norm_{j-k})
        
        # We need to compute G_j = sum_k p_k * (a_norm_{k-j} + a_norm_{j-k})
        # Let's define an array P of length 2N-1 corresponding to p_k.
        # Note: p is defined on the conv indices 0..2N-2.
        
        # Term 1: sum_k p_k * a_norm_{k-j}
        # Let m = k-j. Then k = m+j.
        # Sum_m p_{m+j} * a_norm_m
        # This is a correlation/convolution of p and a_norm.
        
        # Let's pad p and a_norm to do FFT convolution.
        # p is length 2N-1. a_norm is length N.
        
        # Actually, we can just do this with loops or np.convolve if N is small, 
        # but N=600 is fine for FFT.
        
        # Construct full p array of length 2N-1
        # Note: conv indices 0..2N-2 map to lag 0..2N-2.
        # a_norm indices 0..N-1.
        
        # d conv_k / d a_j = a_{k-j} + a_{j-k}
        # Let's compute the gradient term by term.
        
        # We want grad_j = sum_k p_k * (a_{k-j} + a_{j-k}) / h^2
        # 
        # Let's use FFT to compute the sum efficiently.
        # sum_k p_k a_{k-j}
        # Let P be the FFT of p (padded to size >= 2N-1 + N-1 = 3N-2).
        # Let A be the FFT of a (padded).
        # The product in frequency domain gives the convolution.
        
        # However, we have two terms: a_{k-j} and a_{j-k}.
        # Note that a_{j-k} = a_{-(k-j)}.
        # Since a is real, this is related to the real/imag parts.
        
        # Simpler approach for N=600:
        # The cost of FFT is O(N log N). 600 is small.
        # We can compute the gradient in O(N^2) if necessary? 600^2 = 360k. 
        # Python loops over 360k items are slow (~0.1s-0.5s). 
        # FFT is safer.
        
        # Let's use scipy.signal.convolve or numpy.fft.
        
        # p is length 2N-1.
        # We want to compute C1_j = sum_k p_k a_{k-j}
        # This is cross-correlation of p and a.
        # corr(p, a)[j] = sum_k p[k] a[k-j]
        
        # And C2_j = sum_k p_k a_{j-k}
        # This is cross-correlation of p and a reversed?
        # a_{j-k} = a[-(k-j)].
        
        # Let's just use a direct matrix multiplication if we can vectorize it.
        # Or use FFT.
        
        # Let's try a semi-direct method using np.roll or similar if possible.
        # Actually, for N=600, a simple loop over j (600 iterations) where each iteration 
        # does a dot product of length ~1200 is 720k operations. 
        # In numpy, this is very fast if we use slicing.
        
        grad_an = np.zeros(N)
        
        # p is length 2N-1
        # a_norm is length N
        
        # For each j in 0..N-1:
        # term1 = sum_{k} p_k a_{k-j}
        # k ranges 0..2N-2.
        # a_{k-j} is non-zero if 0 <= k-j < N => j <= k < j+N.
        
        # term2 = sum_{k} p_k a_{j-k}
        # a_{j-k} is non-zero if 0 <= j-k < N => j-N < k <= j.
        
        # We can compute these using slicing.
        
        # Pre-compute nothing, just loop.
        for j in range(N):
            # Term 1: k from j to j+N-1
            # p indices: j to min(j+N-1, 2N-2)
            start1 = j
            end1 = min(j + N, 2*N - 1)
            if start1 < end1:
                # a indices: 0 to end1-start1-1
                p_slice = p[start1:end1]
                a_slice = a_norm[0:end1-start1]
                term1 = np.dot(p_slice, a_slice)
            else:
                term1 = 0.0
            
            # Term 2: k from max(0, j-N+1) to j
            start2 = max(0, j - N + 1)
            end2 = j + 1
            if start2 < end2:
                # a_{j-k}
                # Let m = j-k. When k=start2, m=j-start2. When k=end2-1=j, m=0.
                # So a indices go from j-start2 down to 0.
                # p indices go from start2 to j.
                p_slice2 = p[start2:end2]
                # a_slice2 should be a_norm[j-start2 : -1 : -1] ? No.
                # a_{j-k}. 
                # k = start2 -> a_{j-start2}
                # k = j -> a_0
                # So we need a_norm[j-start2] down to a_norm[0].
                a_idx_start = j - start2
                a_slice2 = a_norm[0:a_idx_start+1][::-1]
                term2 = np.dot(p_slice2, a_slice2)
            else:
                term2 = 0.0
            
            grad_an[j] = (term1 + term2) / (h**2)
            
        # Chain rule: d obj / d a = (1/S) * grad_an
        grad_a = grad_an / S
        
        return obj_val, grad_a

def construct():
    deadline = time.time() + 140.0
    
    # Initialize with uniform distribution
    a0 = np.ones(N) / N
    
    best_score = _compute_score_raw(a0)
    best_a = a0.copy()
    
    # We will run L-BFGS-B with a few different lambda values.
    # Small lambda: smoother, easier to find global structure.
    # Large lambda: sharper, focuses on peak.
    
    lambdas = [50.0, 100.0, 200.0, 500.0]
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
            
        def _obj_grad(x):
            # x is the raw heights
            # We ensure x is non-negative? 
            # L-BFGS-B handles bounds.
            # But our derivation assumed a >= 0 for the normalization S=sum(a).
            # If a has negative values, S might be small or zero.
            # We should constrain a >= 0.
            
            # Compute objective and gradient
            # Note: _soft_max_obj_and_grad expects a array.
            # It computes S = sum(a). If S <= 0, it returns inf.
            # L-BFGS-B with bounds [0, inf] should keep a >= 0.
            
            val, grad = _soft_max_obj_and_grad(x, lam)
            return val, grad
            
        # Bounds: 0 to 1 (arbitrary upper bound, scale is invariant but helps stability)
        bounds = [(0, 1.0)] * N
        
        # Start from current best or a perturbed version?
        # Let's start from the best found so far.
        x0 = best_a.copy()
        
        # Add small noise to escape local minima?
        # For the first run, uniform is good.
        
        try:
            res = minimize(_obj_grad, x0, method='L-BFGS-B', 
                           jac=True, bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000})
            
            if res.fun < best_score:
                # Verify the raw score
                raw_score = _compute_score_raw(res.x)
                if raw_score < best_score:
                    best_score = raw_score
                    best_a = res.x.copy()
        except Exception:
            pass
            
    # Final polish: A few steps of gradient descent on the raw max (using subgradient)
    # or just return the best found.
    # The soft-max with high lambda should be very close to the true max.
    
    # Ensure the returned array is clean
    best_a = np.maximum(best_a, 0)
    
    return list(best_a)

if __name__ == "__main__":
    a = construct()
    print(f"Score: {_compute_score_raw(np.array(a))}")
