"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _construct():
    deadline = time.time() + 145.0

    # Strategy: Iterative spectral shaping with random restarts and local refinement.
    # The core idea is to minimize the peak-to-average power ratio (PAPR) of the power spectrum |X|^2.
    # We use a soft-max approximation of the max absolute autoconvolution to get gradients.
    # We optimize the complex FFT coefficients X directly, enforcing Hermitian symmetry to keep x real.
    # To escape local minima, we use random restarts with different initial spectra.

    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(42)

    def _soft_max_abs(conv, beta=100.0):
        """Smooth approximation of max |conv| using log-sum-exp."""
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        if max_val == 0:
            return 0.0
        shifted = abs_conv - max_val
        return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

    def _optimize_spectral(x_init, deadline, rng):
        """
        Optimize the spectrum X to minimize the soft-max of the autoconvolution.
        We use gradient descent on the real and imaginary parts of X.
        """
        x = x_init.copy()
        # Normalize to have sum N for stability
        s = np.sum(x)
        if abs(s) > 1e-9:
            x = x * (N / s)
        
        best_x_local = x.copy()
        best_val_local = _metric(x)
        
        # Initialize X
        X = np.fft.rfft(x, _FFT_SIZE)
        
        beta = 50.0
        step_size = 0.01
        lr_decay = 0.995
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Compute conv and metric
            Y = X * np.conj(X)
            conv = np.fft.irfft(Y, _FFT_SIZE)[:2 * N - 1]
            abs_conv = np.abs(conv)
            max_val = np.max(abs_conv)
            
            if max_val < 1e-9:
                break
                
            # Compute weights w_k for soft-max gradient
            shifted = abs_conv - max_val
            exp_shifted = np.exp(beta * shifted)
            w = exp_shifted / np.sum(exp_shifted)
            
            # G_k = w_k * sign(conv_k)
            G = w * np.sign(conv)
            
            # Gradient of J w.r.t. x
            # J = soft_max(|conv|)
            # conv = IFFT(|X|^2)
            # We want dJ/dx.
            # Using the chain rule: dJ/dx = 2 * IFFT( FFT(x) * conj(FFT(G_padded)) )
            # This is derived from d(conv_k)/dx_j = x_{j-k} + x_{j+k}
            
            G_padded = np.zeros(_FFT_SIZE)
            G_padded[:2 * N - 1] = G
            G_freq = np.fft.rfft(G_padded, _FFT_SIZE)
            
            fx = np.fft.rfft(x, _FFT_SIZE)
            grad_x_freq = 2 * fx * np.conj(G_freq)
            grad_x = np.fft.irfft(grad_x_freq, _FFT_SIZE)[:N]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad_x)
            if grad_norm < 1e-9:
                break
                
            grad_x = grad_x / grad_norm
            
            # Update x
            x_new = x - step_size * grad_x
            
            # Normalize x to keep integral constant (scale invariance)
            S_new = np.sum(x_new)
            if abs(S_new) > 1e-6:
                x_new = x_new * (N / S_new)
            
            val_new = _metric(x_new)
            if val_new < best_val_local:
                best_val_local = val_new
                best_x_local = x_new.copy()
                x = x_new
                X = np.fft.rfft(x, _FFT_SIZE)
                step_size *= lr_decay
            else:
                # Rejection: reduce step size
                step_size *= 0.9
                if step_size < 0.001:
                    break
        
        return best_x_local, best_val_local

    def _local_search(x, deadline, rng, max_rounds=50000):
        """
        Simple local search to refine the solution.
        Tries small random perturbations.
        """
        best_x_local = x.copy()
        best_val_local = _metric(x)
        no_improve = 0
        step_scale = 0.1
        
        for round_idx in range(max_rounds):
            if time.time() > deadline:
                break
            
            if no_improve > 100:
                step_scale *= 0.9
                no_improve = 0
                if step_scale < 0.001:
                    break
            
            x_test = best_x_local.copy()
            # Perturb a few random elements
            k = rng.integers(1, 5)
            indices = rng.choice(N, size=k, replace=False)
            x_test[indices] += rng.normal(0, step_scale, size=k)
            
            # Normalize
            S_new = np.sum(x_test)
            if abs(S_new) > 1e-6:
                x_test = x_test * (N / S_new)
            
            v = _metric(x_test)
            if v < best_val_local:
                best_val_local = v
                best_x_local = x_test
                no_improve = 0
            else:
                no_improve += 1
        
        return best_x_local, best_val_local

    # Main loop with restarts
    num_restarts = 10
    time_per_restart = 10.0
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create a random initial signal
        # Mix of random and structured signals
        if i == 0:
            x_init = np.ones(N)
        elif i % 2 == 0:
            # Random smooth signal
            freqs = np.fft.rfftfreq(N)
            amp = 1.0 / (1.0 + 10.0 * freqs)
            phase = rng.uniform(0, 2 * np.pi, size=len(amp))
            X_init = amp * np.exp(1j * phase)
            X_init[0] = np.abs(X_init[0])  # Real DC
            x_init = np.fft.irfft(X_init, N)
        else:
            # Random binary-like signal
            x_init = rng.choice([-1, 1], size=N).astype(float)
            x_init += rng.normal(0, 0.1, size=N)
        
        # Spectral optimization
        x_opt, val_opt = _optimize_spectral(x_init, deadline - 10, rng)
        
        # Local refinement
        x_ref, val_ref = _local_search(x_opt, deadline - 5, rng, max_rounds=10000)
        
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
