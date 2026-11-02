import numpy as np
import time

N = 100

def _compute_overlaps(h):
    """Compute overlaps O_k for all shifts k."""
    # O_k = sum_i h_i * (1 - h_{i+k})
    # We can compute this using FFT or direct convolution.
    # h is length N.
    # 1 - h is length N.
    # Circular convolution of h and (1-h).
    # Let g = 1 - h.
    # O_k = sum_i h_i g_{i+k}. This is the circular cross-correlation.
    # Using FFT:
    H = np.fft.fft(h)
    G = np.fft.fft(1.0 - h)
    # Cross correlation is ifft( H * conj(G) )? 
    # Let's check. 
    # Correlation c_k = sum_i h_i g_{i+k}.
    # FFT of c is H * conj(G)? No, that's for linear. For circular:
    # c_k = (h * g_rev)_k? 
    # Actually, simpler:
    # O_k = sum_i h_i (1 - h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k} = N/2 - C_k.
    # C_k is the circular autocorrelation of h.
    # C_k = sum_i h_i h_{i+k}.
    # FFT of C is |H|^2.
    # So C = ifft(|H|^2).real.
    H = np.fft.fft(h)
    C = np.fft.ifft(np.abs(H)**2).real
    # O_k = N/2 - C_k
    return N/2.0 - C

def _project(y, target_sum):
    """Project y onto {x: sum(x)=target_sum, 0<=x<=1} using bisection."""
    # We want sum(clip(y_i - lambda, 0, 1)) = target_sum
    # Let f(lambda) = sum(clip(y_i - lambda, 0, 1))
    # f is non-increasing in lambda.
    # We want f(lambda) = target_sum.
    
    # Bounds for lambda:
    # min lambda: when all y_i are large, lambda is small (negative).
    # max lambda: when all y_i are small, lambda is large.
    # y_i - lambda <= 1 => lambda >= y_i - 1.
    # y_i - lambda >= 0 => lambda <= y_i.
    
    # Let's find a range.
    # If lambda is very small, sum is sum(min(y_i, 1)) (if y_i > 0) but clipped at 1?
    # Actually clip(y_i - lambda, 0, 1).
    # If lambda -> -inf, y_i - lambda -> inf, clip to 1. Sum -> N.
    # If lambda -> +inf, y_i - lambda -> -inf, clip to 0. Sum -> 0.
    # We want sum = target_sum (which is N/2).
    
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = np.clip(y - mid, 0.0, 1.0).sum()
        if val > target_sum:
            lo = mid
        else:
            hi = mid
            
    lam = (lo + hi) / 2.0
    return np.clip(y - lam, 0.0, 1.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    h = np.full(N, 0.5)
    # Add small noise to break symmetry
    h += np.random.uniform(-0.1, 0.1, N)
    h = _project(h, N/2.0)
    
    best_h = h.copy()
    best_score = _compute_overlaps(h).max()
    
    # Parameters for descent
    eta = 0.1
    beta = 10.0 # Smoothness parameter for max
    
    for step in range(1000):
        if time.time() > deadline:
            break
            
        # Compute overlaps
        O = _compute_overlaps(h)
        
        # Smooth max approximation: log-sum-exp
        # To avoid overflow, subtract max
        O_max = np.max(O)
        O_shifted = O - O_max
        weights = np.exp(beta * O_shifted)
        weights /= np.sum(weights)
        
        # Gradient of smooth max
        # d/dh_k log(sum exp(beta O_j)) = beta * sum_j weights_j * dO_j/dh_k
        # O_j = N/2 - C_j. dO_j/dh = -dC_j/dh.
        # C_j = sum_i h_i h_{i+j}.
        # dC_j/dh_k = h_{k-j} + h_{k+j} (circular).
        # So dO_j/dh_k = -(h_{k-j} + h_{k+j}).
        
        # We need grad_h = - beta * sum_j weights_j * (h_{-j} + h_{j})? 
        # Let's compute the gradient vector directly.
        # grad_O = - (h_convolved_with_weights + h_convolved_with_weights_reversed)
        # Actually, dC_j/dh = circ_shift(h, -j) + circ_shift(h, j).
        # So dO_j/dh = - (circ_shift(h, -j) + circ_shift(h, j)).
        
        # grad_smooth_max = beta * sum_j weights_j * dO_j/dh
        # = - beta * sum_j weights_j * (circ_shift(h, -j) + circ_shift(h, j))
        
        # This can be computed efficiently.
        # Let W = weights.
        # sum_j W_j circ_shift(h, j) is circular convolution of W and h.
        # Let's use FFT.
        W_fft = np.fft.fft(weights)
        H_fft = np.fft.fft(h)
        # conv = ifft(W_fft * H_fft)
        # The term is sum_j W_j (h_{k-j} + h_{k+j})
        # h_{k-j} is shift of h by -j.
        # h_{k+j} is shift of h by j.
        # Since weights are real and symmetric? No, weights depend on O, which is symmetric (autocorrelation), so weights are symmetric: W_j = W_{-j}.
        # So sum_j W_j h_{k-j} = sum_j W_j h_{k+j} (by symmetry of W and reindexing).
        # So the sum is 2 * circular_convolution(W, h).
        
        conv = np.fft.ifft(W_fft * H_fft).real
        grad = -2.0 * beta * conv
        
        # Update
        h_new = h - eta * grad
        
        # Project
        h_new = _project(h_new, N/2.0)
        
        # Check improvement
        O_new = _compute_overlaps(h_new)
        score_new = O_new.max()
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            
        h = h_new
        
        # Decrease learning rate or increase beta?
        # Let's slowly decrease eta to converge.
        eta *= 0.99
        # Increase beta to make the smooth max sharper
        beta *= 1.01
        
    # Final projection to ensure constraints are met exactly
    best_h = _project(best_h, N/2.0)
    return best_h
