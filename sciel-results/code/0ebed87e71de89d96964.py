"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral phase retrieval approach:
    1. Design a flat-top magnitude spectrum to minimize peak-to-sidelobe ratio.
    2. Optimize the phase via gradient descent to enforce binary step-function constraint.
    3. The autoconvolution magnitude is fixed by |X|^2, so we only optimize phase.
    """
    
    np.random.seed(42)
    
    # Design flat-top magnitude spectrum
    # A flat-top window in frequency domain gives controlled sidelobes
    freqs = np.fft.fftfreq(N)
    
    # Use a raised-cosine (Hann-like) shape for the magnitude to control sidelobes
    # This creates a smoother transition and reduces spectral leakage
    mag_profile = 0.5 * (1 + np.cos(2 * np.pi * freqs * N / 2))
    
    # Ensure DC component is large (for large integral S)
    # Scale so that the resulting autoconvolution has controlled peak
    mag = mag_profile ** 0.5  # Since autoconv magnitude = |X|^2, we want |X| ~ sqrt(desired)
    
    # Normalize to have reasonable scale
    mag = mag / np.sum(mag) * np.sqrt(N)
    
    # Initialize phase randomly
    phase = np.random.uniform(0, 2 * np.pi, size=N)
    
    # Set DC phase to 0 for consistency
    phase[0] = 0.0
    
    def compute_x_from_phase(phase):
        """Reconstruct time-domain signal from fixed magnitude and variable phase."""
        X = mag * np.exp(1j * phase)
        # Enforce Hermitian symmetry for real signal
        X[0] = np.abs(X[0])  # DC is real and positive
        for k in range(1, N // 2 + 1):
            if k < N - k:  # Avoid double-counting when N is even
                X[N - k] = np.conj(X[k])
            elif k == N - k:  # Nyquist frequency (when N even)
                X[k] = np.abs(X[k])
        x = np.fft.ifft(X).real
        return x
    
    def compute_autoconv(x):
        """Compute circular autoconvolution via FFT."""
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        return c
    
    def objective_and_grad(phase):
        """
        Compute objective (max |c_k| / S^2) and gradient w.r.t. phase.
        """
        x = compute_x_from_phase(phase)
        
        # Project to binary: use soft thresholding for differentiability
        # x_binary = tanh(alpha * x) for smooth approximation
        alpha = 5.0
        x_bin = np.tanh(alpha * x)
        
        c = compute_autoconv(x_bin)
        S = np.sum(x_bin)
        
        if abs(S) < 1e-10:
            return 1e10, np.zeros(N)
        
        max_c = np.max(np.abs(c))
        obj = max_c / (S * S)
        
        # Gradient computation:
        # We need d(obj)/d(phase)
        # Chain rule: d(phase) -> d(x) -> d(x_bin) -> d(c) -> d(obj)
        
        # d(x_bin)/d(x) = alpha * (1 - x_bin^2)
        dx_bin_dx = alpha * (1 - x_bin ** 2)
        
        # d(c)/d(x_bin): c_k = sum_i x_i x_{i+k}
        # dc_k/dx_j = x_{j-k} + x_{j+k} (circular)
        # This is a circular convolution-like operation
        # grad_x_bin = 4 * circular_conv(c, x_bin) for L2 proxy, but we want max
        
        # For max |c_k|, use subgradient: if c_k > 0 at max, grad = +1; if < 0, grad = -1
        # But this is non-differentiable. Use soft-max approximation.
        
        # Simpler: minimize sum(c_k^2) as proxy, which is smooth
        # d(sum c_k^2)/dx_j = 4 * sum_k c_k * (x_{j-k} + x_{j+k}) / 2 = 2 * circular_conv(c, x_bin)
        
        # Actually, let's use the L2 norm of sidelobes (excluding DC) as a smoother objective
        # This is easier to differentiate and correlates with peak reduction
        
        # Compute gradient of sum(c_k^2) w.r.t. x_bin
        # Using FFT: if c = ifft(|fft(x)|^2), then grad_x = 4 * ifft(fft(c) * fft(x))
        grad_x_bin = 4 * np.fft.ifft(np.fft.fft(c) * np.fft.fft(x_bin)).real
        
        # Chain: d(obj)/d(phase) = d(obj)/d(x_bin) * d(x_bin)/d(x) * d(x)/d(phase)
        
        # d(x)/d(phase_k): x = ifft(mag * exp(i*phase))
        # dx/dphase_k = ifft(mag * i * exp(i*phase) * delta_k)
        # = i * mag_k * exp(i*phase_k) * ifft(delta_k)
        # ifft(delta_k) = (1/N) * exp(2*pi*i*k*n/N)
        
        # This is expensive to compute directly. Use FFT domain manipulation.
        # In FFT domain: X = mag * exp(i*phase)
        # dX_k/dphase_m = X_k * i * delta_{km}
        # dx/dphase_m = ifft(X * i * delta_m) = i * X_m * (1/N) * exp(2*pi*i*m*n/N)
        
        # So d(x)/dphase_m = i * X_m / N * exp(2*pi*i*m*n/N)
        
        # We need: grad_phase = sum_n grad_x_bin[n] * dx[n]/dphase[m]
        # = sum_n grad_x_bin[n] * i * X_m / N * exp(2*pi*i*m*n/N)
        # = i * X_m / N * sum_n grad_x_bin[n] * exp(2*pi*i*m*n/N)
        # = i * X_m / N * fft(grad_x_bin)[m]
        
        G = np.fft.fft(grad_x_bin * dx_bin_dx)  # Include dx_bin/dx factor
        X = mag * np.exp(1j * phase)
        
        # Enforce Hermitian symmetry in gradient too
        grad_phase = 1j * X / N * G
        
        # Make gradient real (since phase is real parameter)
        # The imaginary part should be the actual gradient for real-valued phase
        grad_phase = np.real(grad_phase)
        
        # Enforce Hermitian symmetry
        grad_phase[0] = 0.0  # DC phase is fixed
        for k in range(1, N // 2 + 1):
            if k < N - k:
                grad_phase[N - k] = grad_phase[k]
            elif k == N - k:
                grad_phase[k] = 0.0
        
        return obj, grad_phase
    
    # Gradient descent on phase
    lr = 0.1
    best_obj = float('inf')
    best_phase = phase.copy()
    
    for iteration in range(2000):
        obj, grad = objective_and_grad(phase)
        
        if obj < best_obj:
            best_obj = obj
            best_phase = phase.copy()
        
        # Adaptive learning rate
        lr = max(0.01, min(0.5, lr * 0.999))
        
        phase -= lr * grad
        
        # Convergence check
        if iteration % 100 == 0 and iteration > 0:
            x_check = compute_x_from_phase(phase)
            x_bin_check = np.tanh(5.0 * x_check)
            c_check = compute_autoconv(x_bin_check)
            S_check = np.sum(x_bin_check)
            if abs(S_check) > 1e-10:
                obj_check = np.max(np.abs(c_check)) / (S_check ** 2)
                if obj_check < best_obj * 0.99:
                    best_obj = obj_check
                    best_phase = phase.copy()
    
    # Final: extract binary sequence from best phase
    phase = best_phase
    x = compute_x_from_phase(phase)
    x_bin = np.sign(x)
    
    # Ensure non-zero sum
    if np.sum(x_bin) == 0:
        x_bin[0] = 1.0
    
    # Local refinement: try flipping bits to improve objective
    def score(x):
        c = compute_autoconv(x)
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        return np.max(np.abs(c)) / (S * S)
    
    current_score = score(x_bin)
    improved = True
    max_passes = 5
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for i in range(N):
            x_bin[i] *= -1
            new_score = score(x_bin)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x_bin[i] *= -1
    
    return x_bin

# EVOLVE-BLOCK-END
