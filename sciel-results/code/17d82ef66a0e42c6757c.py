import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, tau=0.1, penalty=1e6):
    """
    Compute smooth approximation of max(|ac|) + penalty * (sum(x)-1)^2.
    Uses soft-max over |ac| values.
    """
    s = np.sum(x)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max approximation of max
    m = np.max(abs_ac)
    if m < 1e-15:
        softmax_val = 0.0
        # Gradient of softmax when max is 0 is roughly uniform or zero depending on implementation
        # d/dx (1/tau) log sum exp(tau |ac_k|)
        # If all 0, exp(0)=1, sum=N, log(N)/tau. Grad is 0?
        # Let's use a safe gradient computation.
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        # Shift for numerical stability
        shifted = (abs_ac - m) / tau
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        softmax_val = (m + tau * np.log(sum_exp)) # This is (1/tau) log sum exp(tau |ac|)
        weights = exp_shifted / sum_exp
        
    # Gradient of softmax(|ac|)
    # d/dx_j sum_k weights_k * |ac_k|
    # = sum_k weights_k * sign(ac_k) * d(ac_k)/dx_j
    # d(ac_k)/dx_j = x_{k-j} + x_{j-k}
    
    grad_ac = np.zeros(N)
    # We need to accumulate contributions from all k
    # This is O(N^2) if done naively.
    # However, we can compute the weighted sum of ac gradients.
    # Let W_k = weights_k * sign(ac_k)
    # grad = sum_k W_k * (x_{k-j} + x_{j-k})
    # This is a convolution-like operation.
    # Let y_k = W_k.
    # term1_j = sum_k y_k x_{k-j} = (y * x)_j ? No, indices are tricky.
    # ac_k = sum_i x_i x_{k-i}
    # d ac_k / d x_j = x_{k-j} + x_{j-k}
    # sum_k W_k x_{k-j} = (W * x)_{?}
    # Let's use FFT for speed.
    # sum_k W_k x_{k-j} is the cross-correlation of W and x.
    # corr_{j} = sum_k W_k x_{k-j}
    # This is equivalent to conv(W, x) at index j?
    # Conv: (W*x)_j = sum_i W_i x_{j-i}.
    # Let i = k-j => k = i+j.
    # sum_i W_{i+j} x_i. Not quite.
    
    # Let's just use the direct sum if N is small enough? N=400. N^2 = 160k.
    # 5000 iterations * 160k = 800M ops. Might be slow in Python.
    # Let's use FFT.
    # We want G_j = sum_k W_k (x_{k-j} + x_{j-k})
    # Note x_{k-j} is non-zero only if 0 <= k-j < N.
    # Let's compute C1_j = sum_k W_k x_{k-j}.
    # This is the cross-correlation of W and x.
    # corr[W, x]_j = sum_k W_k x_{k-j}.
    # In FFT: IFFT(FFT(W) * conj(FFT(x)))?
    # Let's check:
    # FFT(W * x)_k = W_k x_k.
    # We want sum_k W_k x_{k-j}.
    # This is (W * x) in some sense?
    # Actually, simpler:
    # grad = 2 * real( IFFT( W_fft * X_fft ) ) ?
    # Let's derive:
    # ac = IFFT(X^2)
    # d ac / d x is a linear operator.
    # J_{kj} = d ac_k / d x_j = x_{k-j} + x_{j-k}
    # We want v = W^T J = sum_k W_k J_{kj}
    # v_j = sum_k W_k (x_{k-j} + x_{j-k})
    # Let u_k = W_k.
    # v_j = sum_k u_k x_{k-j} + sum_k u_k x_{j-k}
    # The second term: let l = j-k => k = j-l.
    # sum_l u_{j-l} x_l. This is (u * x)_j (convolution).
    # The first term: sum_k u_k x_{k-j}. Let m = k-j => k = m+j.
    # sum_m u_{m+j} x_m. This is (u * x)_{?}
    # Actually, sum_k u_k x_{k-j} is the cross-correlation of u and x at lag j.
    # Correlation C_{ux}(j) = sum_k u_k x_{k-j}.
    # In FFT: IFFT( U_fft * conj(X_fft) ).
    # So v_j = IFFT(U * conj(X))_j + IFFT(U * X)_j.
    # Since x is real, X is Hermitian.
    # Let's compute this.
    
    U = np.fft.rfft(W * np.sign(ac), n=2*N) # W is defined on 0..2N-2, pad to 2N?
    # Wait, W_k is for k=0..2N-2.
    # We need to be careful with FFT sizes.
    # Let's use the same L_fft as before.
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
        
    W_full = np.zeros(L_fft)
    W_full[:L] = W * np.sign(ac)
    U = np.fft.rfft(W_full)
    X = np.fft.rfft(x, n=L_fft)
    
    # Term 1: sum_k W_k x_{k-j} = IFFT(U * conj(X))_j
    t1 = np.fft.irfft(U * np.conj(X), n=L_fft)[:N]
    # Term 2: sum_k W_k x_{j-k} = IFFT(U * X)_j
    t2 = np.fft.irfft(U * X, n=L_fft)[:N]
    
    grad_smooth = t1 + t2
    
    # Gradient of penalty term lambda * (sum(x) - 1)^2
    grad_penalty = 2 * penalty * (s - 1) * np.ones(N)
    
    total_val = softmax_val + penalty * (s - 1)**2
    total_grad = grad_smooth + grad_penalty
    
    return total_val, total_grad

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 140  # safety margin before 150s kill

    best_x = np.ones(N)
    best_x /= np.sum(best_x)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Try a few diverse initializations
    initializations = [
        np.ones(N),
        np.linspace(1.0, 2.0, N),
        np.linspace(2.0, 1.0, N),
        0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, N))),
        0.5 * (1 + np.sin(2 * np.pi * np.linspace(0, 1, N))) + 0.5,
        np.exp(-np.linspace(0, 2, N)),
    ]
    
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        s = np.sum(x0)
        if abs(s) < 1e-6: continue
        x0_norm = x0 / s
        val = _objective(x0_norm)
        if val < best_val:
            best_val = val
            best_x = x0_norm.copy()

    # Use L-BFGS-B to minimize the smooth objective
    # We optimize x directly, with penalty for sum != 1
    
    def obj_and_grad(x):
        val, grad = _smooth_objective_and_grad(x, tau=0.1, penalty=1e6)
        return val, grad

    # Start from best_x
    x_start = best_x.copy()
    
    try:
        res = minimize(
            obj_and_grad, 
            x_start, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        
        if res.success or res.fun < best_val:
            x_opt = res.x
            # Normalize to ensure sum=1 exactly
            s_opt = np.sum(x_opt)
            if abs(s_opt) > 1e-10:
                x_opt = x_opt / s_opt
                val_opt = _objective(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt.copy()
    except Exception:
        pass

    # If we have time left, try a few more random restarts with L-BFGS-B
    if time.time() - START_TIME < TIME_LIMIT - 10:
        for restart in range(10):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            x_rand = rng.normal(size=N)
            s_rand = np.sum(x_rand)
            if abs(s_rand) < 1e-6:
                continue
            x_rand /= s_rand
            val_rand = _objective(x_rand)
            if val_rand < best_val:
                best_val = val_rand
                best_x = x_rand.copy()
                
            try:
                res = minimize(
                    obj_and_grad, 
                    x_rand, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
                )
                if res.success or res.fun < best_val:
                    x_opt = res.x
                    s_opt = np.sum(x_opt)
                    if abs(s_opt) > 1e-10:
                        x_opt = x_opt / s_opt
                        val_opt = _objective(x_opt)
                        if val_opt < best_val:
                            best_val = val_opt
                            best_x = x_opt.copy()
            except Exception:
                pass

    return best_x
