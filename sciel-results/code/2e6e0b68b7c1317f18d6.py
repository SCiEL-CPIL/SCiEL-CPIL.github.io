import numpy as np
import time

N = 600
DELTA = 1.0 / 1200.0

def compute_score(h):
    # h is a numpy array of length N
    # Compute autoconvolution c
    # c_k = sum h_i h_{k-i}
    # We only need the max c_k.
    # Using FFT for speed
    h_padded = np.zeros(2 * N)
    h_padded[:N] = h
    fft_h = np.fft.fft(h_padded)
    fft_conv = fft_h * np.conj(fft_h)
    conv = np.fft.ifft(fft_conv).real
    # The convolution is symmetric. The max is likely at the center or ends.
    # The valid range for c is 0 to 2N-1.
    # Due to FFT circular convolution, we need to unwrap.
    # But since h is zero-padded to 2N, and N=600, 2N=1200.
    # The linear convolution length is 2N-1 = 1199.
    # FFT size 1200 is sufficient for linear convolution if we zero pad to 2N?
    # Actually, linear conv of two N-length vectors has length 2N-1.
    # FFT size M >= 2N-1. 1200 >= 1199. OK.
    # The result conv[0] is c_0. conv[1] is c_1...
    # The peak is max(conv)
    max_c = np.max(conv)
    norm_sq = np.sum(h**2)
    if norm_sq == 0:
        return float('inf')
    return max_c / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try a few initializations
    inits = []
    # 1. Uniform
    inits.append(np.ones(N))
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 * 5))
    # 3. Split (two blocks)
    h_split = np.zeros(N)
    h_split[:N//4] = 1.0
    h_split[3*N//4:] = 1.0
    inits.append(h_split)
    # 4. Random non-negative
    inits.append(np.random.rand(N))
    
    for h0 in inits:
        if time.time() > deadline - 20.0:
            break
            
        h = h0.copy()
        # Normalize to unit L2 norm to stabilize gradient steps?
        # The objective is scale invariant, so we can constrain ||h||_2 = 1
        h = h / np.linalg.norm(h)
        
        # Gradient Descent
        # L = log(max_c) - log(norm_sq)
        # grad L = grad log(max_c) - 2h/norm_sq
        # We want to minimize L, so h = h - eta * grad L
        
        eta = 0.1
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
                
            # Compute gradient of max_c
            # Find index of max c
            h_padded = np.zeros(2 * N)
            h_padded[:N] = h
            fft_h = np.fft.fft(h_padded)
            fft_conv = fft_h * np.conj(fft_h)
            conv = np.fft.ifft(fft_conv).real
            k_max = np.argmax(conv)
            
            # Gradient of c_k with respect to h
            # c_k = sum_i h_i h_{k-i}
            # d c_k / d h_j = h_{k-j} + h_{j-k} (with bounds)
            # We can compute this via correlation
            # grad_c = h * h_rotated + h_rotated * h ?
            # Actually, grad_c[j] = sum_i (delta_{i,j} h_{k-i} + h_i delta_{k-i,j})
            # = h_{k-j} + h_{j-k}
            
            # Construct the gradient vector
            grad_c = np.zeros(N)
            # Term 1: h_{k-j}
            # We need h[k-j] for j in 0..N-1
            # Indices k-j range from k to k-(N-1)
            # We can use np.roll or slicing
            # Let's use a helper to get h[idx] with 0 padding
            def get_h(idx):
                # idx is array
                res = np.zeros_like(idx, dtype=float)
                mask = (idx >= 0) & (idx < N)
                res[mask] = h[idx[mask]]
                return res
            
            idx1 = k_max - np.arange(N)
            idx2 = np.arange(N) - k_max
            g1 = get_h(idx1)
            g2 = get_h(idx2)
            grad_c = g1 + g2
            
            # Normalize gradient of max_c?
            # The objective is log(max_c). Grad is grad_c / max_c.
            max_c_val = np.max(conv)
            if max_c_val == 0:
                break
            grad_log_max = grad_c / max_c_val
            
            # Grad of -log(norm_sq) is -2h / norm_sq
            norm_sq = np.sum(h**2)
            grad_log_norm = 2 * h / norm_sq
            
            grad_total = grad_log_max - grad_log_norm
            
            # Update
            h = h - eta * grad_total
            
            # Project to non-negative
            h[h < 0] = 0
            
            # Re-normalize to unit L2 (optional, keeps scale fixed)
            norm_h = np.linalg.norm(h)
            if norm_h > 0:
                h = h / norm_h
            else:
                break
                
            # Check score
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            # Adaptive learning rate?
            # If score increased, reduce eta
            # Simple: if no improvement for 100 steps, break or reduce eta
            # We don't track history here, so just run fixed steps
            
    # Return the best found
    return list(best_h)
