import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Initialize with a sinusoid + DC offset
    # We want sum h_i = 1
    # h_i = a + b * sin(2*pi*i/P)
    # sum h_i = N*a + b*sum(sin) ≈ N*a (for integer periods)
    # So a ≈ 1/N
    
    # Let's use a multi-frequency sinusoid for a flatter autocorrelation
    # Actually, let's start with a random sequence with a DC offset
    # h_i = 1/N + noise
    
    rng = np.random.default_rng(42)
    
    # Try different initializations
    best_h = None
    best_score = np.inf
    
    for init_type in ['sinusoid', 'random', 'smooth']:
        if time.time() > deadline:
            break
        
        if init_type == 'sinusoid':
            # Multi-frequency sinusoid
            t = np.arange(N)
            h = np.zeros(N)
            for freq in [1, 2, 3, 5, 7]:
                h += np.sin(2 * np.pi * freq * t / N)
            h += 1.0  # DC offset
        elif init_type == 'random':
            h = rng.standard_normal(N)
            h += 1.0  # DC offset
        elif init_type == 'smooth':
            t = np.arange(N)
            h = np.sin(2 * np.pi * t / N) + 1.0
        
        # Normalize to sum = 1
        h = h / np.sum(h)
        
        # Gradient descent
        lr = 0.01
        T = 10.0  # temperature for soft max
        for it in range(2000):
            if time.time() > deadline:
                break
            
            # Compute autocorrelation
            # A_k = sum_i h_i h_{i+k} for k = -(N-1)..N-1
            # Use FFT
            n_fft = 1
            while n_fft < 2*N - 1:
                n_fft *= 2
            H = np.fft.fft(h, n_fft)
            A = np.fft.ifft(H * np.conj(H)).real
            # A[0] = A_0, A[1:N] = A_1..A_{N-1}, A[n_fft-N+1:] = A_{-1}..A_{-(N-1)}
            # By symmetry, A_k = A_{-k}
            # So A_full = [A_{-(N-1)}, ..., A_0, ..., A_{N-1}]
            # A[0] = A_0
            # A[1:N] = A_1..A_{N-1}
            # A[n_fft-N+1:n_fft] = A_{-(N-1)}..A_{-1}
            
            # Get all 2N-1 lags
            A_full = np.concatenate([A[n_fft-N+1:], A[:N]])  # A_{-(N-1)}..A_0..A_{N-1}
            # Actually, let me re-index
            # A[k] for k in 0..n_fft-1 is the circular autocorrelation
            # For linear autocorrelation, we need to extract
            # A_linear[k] for k=0..N-1 is A[k]
            # A_linear[-k] for k=1..N-1 is A[n_fft-k]
            
            A_pos = A[:N]  # A_0..A_{N-1}
            A_neg = A[n_fft-N+1:n_fft][::-1]  # A_{-(N-1)}..A_{-1}
            A_all = np.concatenate([A_neg, A_pos])  # A_{-(N-1)}..A_0..A_{N-1}
            
            # Objective: max |A_all|
            abs_A = np.abs(A_all)
            max_abs = np.max(abs_A)
            
            # Soft max for gradient
            T_local = max(T, max_abs * 2)  # adaptive temperature
            exp_T = np.exp(T_local * abs_A / max_abs)  # normalize to avoid overflow
            w = exp_T / np.sum(exp_T)
            
            # Gradient of soft max: sum_k w_k * sign(A_k) * dA_k/dh
            # dA_k/dh_j = h_{j+k} + h_{j-k}
            # This is a convolution
            # grad = sum_k w_k * sign(A_k) * (shift(h, -k) + shift(h, k))
            # = 2 * sum_k w_k * sign(A_k) * shift(h, k) (by symmetry, w_k = w_{-k})
            
            # Let c_k = 2 * w_k * sign(A_k) for k = -(N-1)..N-1
            # grad = convolve(h, c)
            
            sign_A = np.sign(A_all)
            c = 2 * w * sign_A  # c has 2N-1 elements
            # c[k] corresponds to lag k, where k = -(N-1)..N-1
            # c[0] = A_{-(N-1)}, c[N-1] = A_0, c[2N-2] = A_{N-1}
            
            # convolve(h, c) where h has N elements, c has 2N-1 elements
            # result has N + (2N-1) - 1 = 3N-2 elements, but we only want the middle N
            # Actually, grad_j = sum_k c_k * h_{j+k} (with h out of range = 0)
            # This is a cross-correlation
            # Use FFT
            n_fft2 = 1
            while n_fft2 < N + 2*N - 1:
                n_fft2 *= 2
            H2 = np.fft.fft(h, n_fft2)
            C2 = np.fft.fft(c, n_fft2)
            grad = np.fft.ifft(H2 * C2).real[:N]
            
            # Update
            h -= lr * grad
            h = h / np.sum(h)  # renormalize
            
            # Track best
            score = np.max(np.abs(A_all))
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if best_h is None:
            best_h = h
    
    return best_h
