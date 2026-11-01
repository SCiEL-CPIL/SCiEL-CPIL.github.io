import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe.
    Uses Gradient Descent on a continuous relaxation, then thresholds to signs.
    """
    np.random.seed(42)
    
    # Initialize with random continuous values
    f = np.random.uniform(-1, 1, N)
    
    def get_acf(f):
        """
        Computes the autocorrelation vector.
        """
        F = np.fft.fft(f, n=2*N)
        acf = np.fft.ifft(F * np.conj(F)).real
        return acf

    def objective(f):
        """
        Objective function: Sum of squares of non-zero lags (sidelobe energy).
        This is a smooth proxy for the peak sidelobe.
        """
        acf = get_acf(f)
        # Zero out the main lobe (lag 0) and wrap-around artifacts if any
        # acf[0] is lag 0
        # acf[1:N] are lags 1..N-1
        # acf[N+1:2*N] are lags -(N-1)..-1
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        sidelobes = np.concatenate([pos_lags, neg_lags])
        return np.sum(sidelobes ** 2)

    def gradient(f):
        """
        Gradient of the objective function with respect to f.
        J(f) = sum_{k != 0} |acf(k)|^2
        Using FFT, acf = ifft(|F|^2).
        We can compute the gradient numerically or analytically.
        Numerical is safer for implementation brevity, but analytic is faster.
        Let's use a simple numerical gradient check or analytic derivation.
        
        Actually, let's use a standard trick:
        The autocorrelation is a convolution f * f_flipped.
        Minimizing sum of squares of sidelobes is equivalent to minimizing 
        ||P(f * f_flipped)||^2 where P zeros out the center.
        
        Let's stick to numerical gradient for simplicity and robustness, 
        or use a finite difference step if analytic is too complex.
        However, for N=400, analytic gradient via FFT is feasible.
        
        Let's use a simpler approach: 
        We want to minimize J = sum_{k != 0} r[k]^2.
        r[k] = sum_t f[t] f[t+k].
        
        Let's just use a small step gradient descent with numerical gradient 
        if analytic is tricky, but N=400 is small enough for analytic.
        
        Analytic Gradient:
        r = ifft(|F|^2)
        The gradient of r with respect to f is 2 * ifft(F * conj(F) * ...)? 
        Actually, r is real. 
        Let's use the property that the gradient of the energy of the autocorrelation 
        sidelobes can be computed.
        
        Alternative: Use a "soft" peak minimization.
        Let's use the L2 norm of sidelobes as the objective.
        
        To compute the gradient of J = sum_{k != 0} r[k]^2:
        r = conv(f, f)
        dr/df = 2 * conv(1, f) ? No.
        
        Let's use a simple numerical gradient with a small epsilon.
        N=400, so 400 evaluations of FFT per gradient step is 400*400 = 160k ops, which is fine.
        But we can do better.
        
        Let's try a different objective: Minimize the maximum sidelobe using a soft-max.
        But L2 is easier.
        
        Let's implement a simple Gradient Descent with numerical gradient.
        """
        eps = 1e-4
        grad = np.zeros(N)
        base_obj = objective(f)
        for i in range(N):
            f_plus = f.copy()
            f_plus[i] += eps
            obj_plus = objective(f_plus)
            grad[i] = (obj_plus - base_obj) / eps
        return grad

    # Optimization parameters
    lr = 0.01
    max_iters = 500
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > 100:
            break
        g = gradient(f)
        f -= lr * g
        # Keep values bounded to avoid divergence
        f = np.clip(f, -1, 1)
        
    # Threshold to signs
    f_discrete = np.sign(f)
    # Ensure no zeros
    f_discrete[f_discrete == 0] = 1
    
    # Refine with local search (flip signs) to minimize peak sidelobe
    def get_peak_sidelobe(f):
        acf = get_acf(f)
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        return np.max(np.abs(np.concatenate([pos_lags, neg_lags])))
        
    best_f = f_discrete.copy()
    best_peak = get_peak_sidelobe(best_f)
    
    # Local search
    for _ in range(1000):
        if time.time() - start_time > 110:
            break
        i = np.random.randint(N)
        old_val = best_f[i]
        best_f[i] = -old_val
        new_peak = get_peak_sidelobe(best_f)
        if new_peak < best_peak:
            best_peak = new_peak
        else:
            best_f[i] = old_val
            
    return best_f
# EVOLVE-BLOCK-END
