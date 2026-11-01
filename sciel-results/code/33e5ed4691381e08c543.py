"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autocorr_sidelobe_max(x):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(acf[1:]))


def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3


def _objective_and_gradient(x):
    """
    Objective: Sum of squares of sidelobes.
    Gradient: Analytic.
    """
    h = x
    n = len(h)
    ac = np.fft.fft(h, n=2*n)
    power = ac * np.conj(ac)
    acf = np.fft.ifft(power).real[:n]
    sidelobes = acf[1:n]
    obj = np.sum(sidelobes**2)
    
    grad = np.zeros(n)
    for k in range(1, n):
        c_k = sidelobes[k-1]
        if c_k == 0:
            continue
        grad[0:n-k] += 2.0 * c_k * h[k:n]
        grad[k:n] += 2.0 * c_k * h[0:n-k]
        
    return obj, grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with random ±1 sequence
    np.random.seed(42)
    best_x = np.random.choice([-1.0, 1.0], size=N)
    best_score = _autocorr_sidelobe_max(best_x)
    
    # Simulated annealing phase
    current_x = best_x.copy()
    current_score = best_score
    
    T_start = 5.0
    T_end = 0.01
    alpha = 0.9995
    T = T_start
    
    while time.time() < deadline - 30.0:  # Leave 30s for gradient refinement
        idx = np.random.randint(N)
        old_val = current_x[idx]
        current_x[idx] = -old_val
        
        new_score = _autocorr_sidelobe_max(current_x)
        delta = new_score - current_score
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            current_x[idx] = old_val
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    # Gradient-based refinement using smooth approximation
    # Convert to continuous variables for gradient optimization
    # Use tanh to map continuous values to [-1, 1]
    x_cont = np.zeros(N)
    for i in range(N):
        x_cont[i] = np.arctanh(0.999 * best_x[i])  # Map ±1 to large ±values
    
    def _smooth_objective(xc):
        """Smooth approximation of max sidelobe using sum of squares."""
        h = np.tanh(xc)
        obj, _ = _objective_and_gradient(h)
        return obj
    
    def _smooth_gradient(xc):
        """Gradient of smooth objective."""
        h = np.tanh(xc)
        obj, grad_h = _objective_and_gradient(h)
        # Chain rule: d/dxc = d/dh * dh/dxc
        grad_xc = grad_h * (1 - h**2)
        return grad_xc
    
    # Run gradient descent with restarts
    best_x_grad = best_x.copy()
    best_score_grad = best_score
    
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb initial point
        if restart == 0:
            x_start = x_cont.copy()
        else:
            x_start = x_cont + np.random.normal(0, 0.1, N)
        
        result = minimize(
            _smooth_objective,
            x_start,
            jac=_smooth_gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-6}
        )
        
        # Project back to ±1
        h_opt = np.tanh(result.x)
        x_binary = np.sign(h_opt)
        x_binary[x_binary == 0] = 1  # Handle zeros
        
        score_opt = _autocorr_sidelobe_max(x_binary)
        if score_opt < best_score_grad:
            best_score_grad = score_opt
            best_x_grad = x_binary
    
    # Final polish: try single flips until no improvement
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for idx in range(N):
            old_val = best_x_grad[idx]
            best_x_grad[idx] = -old_val
            new_score = _autocorr_sidelobe_max(best_x_grad)
            if new_score < best_score_grad:
                best_score_grad = new_score
                improved = True
            else:
                best_x_grad[idx] = old_val
    
    return best_x_grad
# EVOLVE-BLOCK-END
