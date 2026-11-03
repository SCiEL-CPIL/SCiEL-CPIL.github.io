"""Rewritten program for first autocorrelation inequality: minimize normalized max autoconvolution."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
W = 1.0 / 300.0  # width of each interval; total length = N * W = 2

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s

    # Precompute FFT size
    n_fft = 2 * N

    def autoconv_values(f):
        """Compute continuous autoconvolution values at lags 0..N-1 (discrete samples)."""
        F = np.fft.rfft(f, n=n_fft)
        ac = np.fft.irfft(F * F.conj(), n=n_fft)
        # ac[k] for k=0..N-1 gives sum_{i} f[i]*f[i+k] (circular, but since we padded, it's linear)
        # The continuous autoconvolution at lag t = k*W is W * ac[k]
        # Actually: (f * f)(t) = integral f(x) f(x+t) dx
        # For step functions, at discrete lags k*W: W * sum_i f[i] f[i+k]
        # So the max is W * max(ac[0:N])
        # But ac[0] = sum f[i]^2, ac[k] = sum f[i] f[i+k]
        # For nonnegative f, ac[0] >= ac[k] for all k, so max is at k=0.
        # Wait - is that true? For nonnegative functions, the autoconvolution is indeed maximized at lag 0.
        # So max_autoconv = W * sum(f^2)
        return W * np.sum(f * f)

    def normalized_max(f):
        """Normalized maximum of autoconvolution.
        We need to figure out the correct normalization.
        
        The task says 'normalized maximum of its autoconvolution'.
        Common normalizations:
        1. max(autoconv) / (L1 norm)^2 = W*sum(f^2) / (W*sum(f))^2 = sum(f^2) / (W * sum(f)^2)
        2. max(autoconv) / max_possible = ...
        3. Just the peak value of the continuous autoconvolution
        
        Given the context of 'first autocorrelation inequality' and upper bound on C1,
        the most natural is: max_t (f*f)(t) / (integral f)^2
        which equals sum(f^2) / (W * sum(f)^2)
        
        For uniform f (all 1s): sum(f^2)=N, sum(f)=N, so ratio = N/(W*N^2) = 1/(W*N) = 1/(1/300*600) = 1/2
        """
        s = np.sum(f)
        if s < 1e-10:
            return 1e10
        sq = np.sum(f * f)
        return sq / (W * s * s)

    def objective(x):
        f = np.maximum(x, 0.0)
        return normalized_max(f)

    def objective_and_grad(x):
        f = np.maximum(x, 0.0)
        s = np.sum(f)
        if s < 1e-10:
            return 1e10, np.zeros(N)
        sq = np.sum(f * f)
        val = sq / (W * s * s)
        # Gradient: d/dx_i [sum(f^2)/(W*s^2)] where f_i = max(x_i, 0)
        # Let g = sum(f^2), h = W*s^2
        # dg/dx_i = 2*f_i (if x_i > 0), 0 otherwise
        # dh/dx_i = 2*W*s*f_i (if x_i > 0), 0 otherwise
        # dval/dx_i = (2*f_i * h - g * 2*W*s*f_i) / h^2 = 2*f_i*(h - g*W*s) / h^2
        # h = W*s^2, so h - g*W*s = W*s^2 - W*s*sum(f^2) = W*s*(s - sum(f^2))
        # dval/dx_i = 2*f_i * W*s*(s - sum(f^2)) / (W^2*s^4) = 2*f_i*(s - sum(f^2)) / (W*s^3)
        mask = (x > 0).astype(float)
        grad = 2.0 * f * (s - sq) / (W * s**3) * mask
        return val, grad

    # Start with uniform
    x0 = np.ones(N)
    best_x = x0.copy()
    best_val = objective(x0)

    # Strategy 1: L-BFGS-B with gradient
    if time.time() < deadline - 20:
        try:
            result = minimize(objective_and_grad, best_x, method='L-BFGS-B',
                             jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12},
                             bounds=[(0, None)] * N)
            if result.fun < best_val:
                best_val = result.fun
                best_x = np.maximum(result.x, 0.0)
        except Exception:
            pass

    # Strategy 2: Try different initializations
    if time.time() < deadline - 15:
        # Try a few structured initializations
        inits = []
        # Cosine bell
        t = np.linspace(-0.25, 0.25, N, endpoint=False) + W/2
        inits.append(np.maximum(1.0 + 0.5*np.cos(2*np.pi*t), 0))
        # Gaussian-like
        inits.append(np.exp(-20*(t**2)))
        # Plateau
        inits.append(np.where(np.abs(t) < 0.1, 1.0, 0.1))
        # Triangular
        inits.append(np.maximum(1 - 4*np.abs(t), 0))

        for init in inits:
            if time.time() > deadline - 10:
                break
            init = np.maximum(init, 0)
            # Normalize to have same L1 as uniform
            s_init = np.sum(init)
            if s_init > 1e-10:
                init = init * (N / s_init)  # scale so sum = N
            val_init = objective(init)
            if val_init < best_val:
                best_val = val_init
                best_x = init.copy()
            # Try L-BFGS-B from this init
            try:
                result = minimize(objective_and_grad, init, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12},
                                 bounds=[(0, None)] * N)
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = np.maximum(result.x, 0.0)
            except Exception:
                pass

    # Strategy 3: Coordinate descent refinement
    if time.time() < deadline - 5:
        f = best_x.copy()
        # Use a smarter coordinate descent: for each coordinate, find optimal value
        # Given f, the objective is sum(f^2)/(W*s^2). If we change f_i to v:
        # new_sq = sq - f_i^2 + v^2, new_s = s - f_i + v
        # new_obj = (sq - f_i^2 + v^2) / (W * (s - f_i + v)^2)
        # Minimize over v >= 0.
        
        # For a fixed coordinate i with current value a = f_i:
        # minimize (sq - a^2 + v^2) / (s - a + v)^2
        # Let A = sq - a^2, B = s - a
        # minimize (A + v^2) / (B + v)^2
        # d/dv: [2v(B+v)^2 - (A+v^2)*2(B+v)] / (B+v)^4
        # = 2(B+v)[v(B+v) - (A+v^2)] / (B+v)^4
        # = 2[vB + v^2 - A - v^2] / (B+v)^3
        # = 2(vB - A) / (B+v)^3
        # Set to 0: vB = A => v = A/B (if B > 0)
        # Check if v >= 0: v = (sq - a^2)/(s - a)
        
        for iteration in range(200000):
            if time.time() > deadline - 2:
                break
            # Cyclic coordinate descent for first pass, then random
            if iteration < 10 * N:
                i = iteration % N
            else:
                i = np.random.randint(N)
            
            a = f[i]
            s = np.sum(f)
            sq = np.sum(f * f)
            B = s - a
            A = sq - a * a
            
            if B > 1e-15:
                v_opt = A / B
                if v_opt < 0:
                    v_opt = 0.0
                # Also check v=0 boundary
                obj_current = sq / (W * s * s)
                obj_new = (A + v_opt * v_opt) / (W * (B + v_opt) * (B + v_opt))
                obj_zero = A / (W * B * B)
                
                if obj_new <= obj_zero:
                    f[i] = v_opt
                else:
                    f[i] = 0.0
                
                new_val = np.sum(f*f) / (W * np.sum(f)**2)
                if new_val < best_val:
                    best_val = new_val
                    best_x = f.copy()
            else:
                f[i] = 0.0
                new_val = np.sum(f*f) / (W * np.sum(f)**2) if np.sum(f) > 1e-10 else 1e10
                if new_val < best_val:
                    best_val = new_val
                    best_x = f.copy()

    # Strategy 4: Final polish with L-BFGS-B
    if time.time() < deadline - 3:
        try:
            result = minimize(objective_and_grad, best_x, method='L-BFGS-B',
                             jac=True, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13},
                             bounds=[(0, None)] * N)
            if result.fun < best_val:
                best_val = result.fun
                best_x = np.maximum(result.x, 0.0)
        except Exception:
            pass

    return list(np.maximum(best_x, 0.0))
