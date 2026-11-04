import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    n_fft = 2 * N

    def autoconv(h):
        """Compute autoconvolution h*h via FFT. Returns array of length 2N-1."""
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        return autoconv_circ[:2*N-1] * dx**2

    def objective_value(h):
        """Just the objective value for evaluation."""
        h = np.maximum(h, 0)
        l2 = np.sum(h**2) * dx
        if l2 < 1e-15:
            return 1e10
        h_norm = h / np.sqrt(l2)
        g = autoconv(h_norm)
        return np.max(g)

    def project(h):
        """Project h onto h >= 0 and ||h||_2 = 1."""
        h = np.maximum(h, 0)
        l2 = np.sqrt(np.sum(h**2) * dx)
        if l2 < 1e-15:
            return np.zeros(N)
        return h / l2

    def gradient_top_k(h, K=10):
        """Compute gradient of the max objective using top-K averaging."""
        h = np.maximum(h, 0)
        l2 = np.sqrt(np.sum(h**2) * dx)
        if l2 < 1e-15:
            return np.zeros(N)
        h_norm = h / l2
        g = autoconv(h_norm)
        
        # Find top K indices
        if K >= len(g):
            idx = np.argsort(g)[::-1]
        else:
            idx = np.argpartition(g, -K)[-K:]
            idx = idx[np.argsort(g[idx])[::-1]] # Sort them for consistency, though not strictly needed for sum
        
        # Gradient is sum of shifted h_norm
        # (h*h)_j derivative w.r.t h is 2 * shift(h, j)
        # Note: g is defined on [-N+1, N-1] relative to 0?
        # autoconv returns array of length 2N-1.
        # g[0] corresponds to x = -(N-1)*dx?
        # Let's check the index mapping.
        # h is defined on [0, N-1] in array, representing x in [-1/4, 1/4].
        # autoconv is defined on [-1/2, 1/2].
        # The FFT autoconv_circ[:2*N-1] handles the circular wrap-around.
        # The derivative of the circular convolution is also a shift.
        
        grad = np.zeros(N)
        # The shift amount in array indices for g index j
        # g[j] corresponds to lag j - (N-1) in terms of array indices?
        # Let's verify.
        # h[0] is at -1/4. h[N-1] is at 1/4.
        # (h*h)[0] is sum h[i] h[i-0] -> h[0]h[0]...
        # Actually, standard convolution index:
        # (h*h)[k] = sum_i h[i] h[k-i]
        # In FFT, if we pad to 2N, the result is linear convolution.
        # The output array from ifft has indices 0 to 2N-1.
        # The "center" is at N-1?
        # Let's just use the property that the gradient of the circular convolution
        # is the circular shift.
        
        # Let's compute the gradient directly using the formula:
        # grad_k = sum_{j in topK} 2 * h_norm[(k - j_offset) mod N]
        # What is j_offset?
        # g is length 2N-1.
        # The index j in g corresponds to the lag.
        # Let's map g index to shift.
        # g[0] is lag -(N-1). g[N-1] is lag 0. g[2N-2] is lag N-1.
        # So shift = j - (N-1).
        
        for j in idx:
            shift = int(j - (N - 1))
            # Circular shift of h_norm by -shift?
            # If we want (h*h)_j derivative, it's 2 * h shifted by -j?
            # Let's use np.roll.
            # np.roll(h, shift) moves elements to the right.
            # We want the vector v such that v[k] = h[k - shift].
            # This is np.roll(h, shift).
            grad += 2.0 * np.roll(h_norm, shift)
            
        return grad

    # Generate diverse initial candidates
    centers = np.arange(N) - N // 2
    candidates = []
    
    # Gaussian peaks with various widths
    for sigma in [1, 2, 5, 10, 20, 50, 100, 200]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    # Triangular peaks
    for width in [10, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)
    
    # Box functions
    for half_width in [10, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(centers) <= half_width
        h[mask] = 1.0
        candidates.append(h)
        
    # Parabolic / Super-Gaussian
    for beta in [2, 4, 8, 16]:
        h = np.maximum(0, 1 - (np.abs(centers) / 300.0)**beta)
        candidates.append(h)

    # Evaluate all candidates
    best_score = float('inf')
    best_h = None

    for h in candidates:
        if time.time() > deadline:
            break
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = objective_value(h)
        if score < best_score:
            best_score = score
            best_h = project(h)

    # Projected Gradient Descent Refinement
    if best_h is not None and time.time() < deadline - 20:
        h_curr = best_h.copy()
        score_curr = best_score
        eta = 0.1  # Initial step size
        K = 10     # Top K for smoothing
        
        # Adaptive step size logic
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < deadline - 10:
            grad = gradient_top_k(h_curr, K)
            
            # Gradient step
            h_next = h_curr - eta * grad
            
            # Project
            h_next = project(h_next)
            
            # Evaluate
            score_next = objective_value(h_next)
            
            if score_next < score_curr - 1e-8:
                # Accept
                h_curr = h_next
                score_curr = score_next
                no_improve_count = 0
                # Slightly increase eta if we are improving well?
                # eta *= 1.01
            else:
                # Reject, decrease step size
                eta *= 0.5
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Try a smaller K or break?
                    # Let's try reducing K to get a sharper gradient
                    if K > 1:
                        K = max(1, K // 2)
                        eta *= 0.5
                        no_improve_count = 0
                    else:
                        break
                        
    # Final Polish with L-BFGS-B on smooth objective
    if best_h is not None and time.time() < deadline - 5:
        # Use the best h found by PGD
        h_start = h_curr if 'h_curr' in locals() else best_h
        
        def smooth_objective_with_grad(h, tau=0.01):
            h = np.maximum(h, 0)
            l2 = np.sum(h**2) * dx
            if l2 < 1e-15:
                return 1e10, np.zeros(N)
            h_norm = h / np.sqrt(l2)
            g = autoconv(h_norm)
            
            max_val = np.max(g)
            if max_val <= 0:
                return 0.0, np.zeros(N)
            
            w = np.exp((g - max_val) / tau)
            W = np.sum(w)
            smooth_max = tau * (max_val + np.log(W))
            
            w_padded = np.zeros(n_fft)
            w_padded[:len(w)] = w
            Wp = np.fft.fft(w_padded)
            Hn = np.fft.fft(h_norm)
            grad_hnorm = np.fft.ifft(Wp * Hn).real[:N] / W
            
            sqrt_l2 = np.sqrt(l2)
            d_hnorm_d_h = (1 - h**2 / l2) / sqrt_l2
            grad = grad_hnorm * d_hnorm_d_h
            
            return smooth_max, grad

        try:
            res = minimize(smooth_objective_with_grad, h_start, method='L-BFGS-B', 
                          jac=True,
                          bounds=[(0, None)] * N,
                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
            
            h_opt = np.maximum(res.x, 0)
            score_opt = objective_value(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
