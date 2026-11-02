# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
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
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_objective(params, beta=50.0):
    """
    Objective in spectral domain.
    params: [Re(X0), Im(X0), Re(X1), Im(X1), ..., Re(X_{N/2}), Im(X_{N/2})]
    Note: We use full FFT for simplicity in gradient calculation, then enforce symmetry.
    Actually, let's just use the full complex vector X of length N.
    x = IFFT(X).
    """
    # params is real vector of length 2N
    X = params[0:N] + 1j * params[N:2*N]
    
    # Enforce Hermitian symmetry for real x
    # X[k] = conj(X[N-k])
    # We only optimize the first N//2 + 1 components and mirror them?
    # For L-BFGS-B, it's easier to optimize the full vector and project?
    # Or just optimize the full vector and let IFFT produce complex x, then take real part?
    # If we take real part, the gradient is 0.5 * grad.
    # Better: Optimize only the independent components.
    
    pass # Placeholder, will implement in construct

def _smooth_max_grad(values, beta=50.0):
    """
    Compute soft-max and its gradient w.r.t values.
    values: 1D array
    Returns: soft_max_value, gradient_array
    """
    max_val = np.max(values)
    shifted = beta * (values - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    grad = weights
    return soft_max, grad

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x

    # Spectral Domain Optimization
    # We represent x via its DFT X.
    # x is real, so X is Hermitian.
    # We optimize the real and imaginary parts of X[0:N//2+1].
    # Let M = N//2 + 1.
    # Variables: Re(X[0:M]), Im(X[0:M]).
    # Total variables: 2*M.
    
    M = N // 2 + 1
    def _reconstruct_x(params):
        """Reconstruct x from spectral parameters."""
        # params: [Re(X0)..Re(X_{M-1}), Im(X0)..Im(X_{M-1})]
        Re_X = params[:M]
        Im_X = params[M:2*M]
        
        # X0 is real
        X0 = Re_X[0]
        
        # Construct full X array
        X = np.zeros(N, dtype=complex)
        X[0] = X0
        
        if M > 1:
            # k = 1 to M-1
            X[1:M] = Re_X[1:M] + 1j * Im_X[1:M]
            
            # Fill the rest by symmetry
            # X[N-k] = conj(X[k])
            for k in range(1, M):
                X[N-k] = np.conj(X[k])
                
        # If N is even, X[N/2] is real.
        if N % 2 == 0:
             # k = N/2 is included in M?
             # M = N//2 + 1.
             # Indices 0, ..., N/2.
             # X[N/2] is real.
             # In our param, Im_X[N//2] should be 0.
             # We enforce this by setting it to 0 in reconstruction.
             X[N//2] = Re_X[N//2]
             
        x = np.fft.irfft(X, n=N)
        return x

    def _spectral_obj_grad(params, beta=50.0):
        """
        Objective and gradient in spectral domain.
        """
        x = _reconstruct_x(params)
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(params)
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        soft_max, weights = _smooth_max_grad(abs_ac, beta)
        val = soft_max / (s * s)
        
        # Gradient w.r.t x
        # d(val)/dx = (1/s^2) * d(soft_max)/dx - (2*soft_max/s^3) * 1
        # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx[j]
        # d(ac[n])/dx[j] = x[j+n] + x[j-n] (circular? No, linear autoconv)
        
        # We need d(soft_max)/dx efficiently.
        # Let g[n] = weights[n] * sign(ac[n]).
        # d(soft_max)/dx[j] = sum_n g[n] (x[j+n] + x[j-n])
        # This is a correlation.
        # term1[j] = sum_n g[n] x[j+n]
        # term2[j] = sum_n g[n] x[j-n]
        
        L_fft = 1
        while L_fft < 2 * N:
            L_fft *= 2
            
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        # term1: correlation of g and x
        # sum_n g[n] x[j+n]
        # In freq: IFFT( conj(G) * Xf )?
        # Let's verify.
        # y[j] = sum_n g[n] x[j+n]
        # Y = FFT(y) = sum_j (sum_n g[n] x[j+n]) exp(-2pi i j k / L)
        # This is tricky with linear boundaries.
        # Easier: Use np.correlate or FFT with padding.
        # Let's use FFT method.
        # y1 = ifft( fft(g) * conj(fft(x)) )? No.
        
        # Let's just use the direct sum if N is small, but N=400 is small enough for O(N^2) if needed?
        # 400^2 = 160,000 ops. Very fast.
        # But we want speed.
        
        # Let's use the FFT correlation formula properly.
        # corr[g, x][k] = sum_n g[n] x[n+k]
        # This is cross-correlation.
        # Using FFT:
        # C = ifft( conj(fft(g)) * fft(x) )
        # This gives circular correlation.
        # We need linear.
        # Pad g and x to length >= N + len(g) - 1.
        
        L_corr = 2 * N
        G_p = np.fft.rfft(g, n=L_corr)
        X_p = np.fft.rfft(x, n=L_corr)
        
        # term2[j] = sum_n g[n] x[j-n]
        # This is correlation of g and x with lag j.
        # y2[j] = sum_n g[n] x[j-n]
        # Y2 = ifft( G_p * conj(X_p) )?
        # Let's check.
        # y2 = g * x (convolution)? No.
        # y2[j] = sum_n g[n] x[j-n] is convolution (g * x)[j].
        # So term2 = ifft( G_p * X_p ).
        
        # term1[j] = sum_n g[n] x[j+n]
        # Let m = -n. sum_m g[-m] x[j-m].
        # This is convolution of g_rev and x.
        # g_rev[n] = g[-n].
        # FFT(g_rev) = conj(FFT(g)) (for real g, and appropriate shift).
        # Actually, simpler:
        # term1 = ifft( conj(G_p) * X_p )?
        
        # Let's compute both and add.
        term2 = np.fft.irfft(G_p * X_p, n=L_corr)[:N]
        
        # For term1:
        # x[j+n] = x_rev[-(j+n)]?
        # Let's just use np.correlate for safety and simplicity, N=400 is fast.
        # np.correlate(g, x, mode='full')
        # g has length 2N-1. x has length N.
        # We need sum_n g[n] x[j+n] for j in 0..N-1.
        # This is the cross-correlation.
        
        # To be safe and fast, let's stick to the previous FFT gradient logic which worked,
        # but now we need to map gradient from x to params.
        
        # Gradient w.r.t x:
        # d_soft_max_dx = term1 + term2
        # We computed term2 as convolution.
        # term1 is "anti-convolution" or correlation with sign flip.
        
        # Let's use the explicit loop for gradient w.r.t x if FFT is tricky, 
        # but N=400 allows O(N^2) gradient in ~1ms.
        # Let's use O(N^2) for robustness in the spectral step.
        
        d_soft_max_dx = np.zeros(N)
        # g[n] = weights[n] * sign(ac[n])
        # d(soft_max)/dx[j] = sum_n g[n] * (x[j+n] + x[j-n])
        
        # We can vectorize this.
        # For each n, it contributes g[n] to x[j+n] and x[j-n].
        # This is a sparse matrix multiplication.
        
        # Let's use the FFT method from the parent code which was correct:
        # term1 = ifft( conj(G) * Xf )
        # term2 = ifft( G * Xf )
        
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
            
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_soft_max_dx = term1 + term2
        
        grad_x = (1.0 / (s * s)) * d_soft_max_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
        
        # Map grad_x to grad_params
        # x = IFFT(X)
        # X = FFT(x)
        # dx/dX = IFFT_matrix
        # dX/dparams = ...
        
        # Chain rule:
        # grad_params = J^T grad_x
        # where J = dx/dparams.
        
        # x = IFFT(X)
        # X is constructed from params.
        
        # Let's compute gradient w.r.t X first.
        # grad_X = FFT(grad_x)
        # Because x = IFFT(X) => X = FFT(x).
        # <grad_x, dx> = <grad_X, dX>
        
        grad_X = np.fft.fft(grad_x)
        
        # Now grad_params from grad_X.
        grad_params = np.zeros(2 * M)
        
        # params[0:M] = Re(X[0:M])
        # params[M:2M] = Im(X[0:M])
        
        # dX[0] = dRe(X0)
        # dX[k] = dRe(Xk) + i dIm(Xk) for k=1..M-1
        # dX[N-k] = conj(dX[k])
        
        # grad_params[0] = Re(grad_X[0])
        # grad_params[1] = Re(grad_X[1])
        # grad_params[2] = Re(grad_X[2]) ...
        # grad_params[M] = Im(grad_X[0]) (should be 0)
        # grad_params[M+1] = Im(grad_X[1])
        
        for k in range(M):
            grad_params[k] = np.real(grad_X[k])
            grad_params[M + k] = np.imag(grad_X[k])
            
        # Note: X0 is real, so Im(X0) is fixed to 0.
        # The param for Im(X0) is not used in reconstruction (we set X0=Re_X[0]).
        # So grad_params[M] (which corresponds to Im(X0)) should be 0.
        grad_params[M] = 0.0
        
        # If N is even, X[N/2] is real.
        if N % 2 == 0:
             k_mid = N // 2
             # X[k_mid] = Re_X[k_mid]. Im is fixed 0.
             # So grad_params[M + k_mid] should be 0.
             grad_params[M + k_mid] = 0.0
             # And grad_X[k_mid] is real, so its imaginary part is 0 anyway.
             
        return val, grad_params

    # Initialize spectral params from best_x
    X_init = np.fft.fft(best_x)
    params_init = np.zeros(2 * M)
    for k in range(M):
        params_init[k] = np.real(X_init[k])
        params_init[M + k] = np.imag(X_init[k])
        
    # Optimization
    try:
        result = minimize(_spectral_obj_grad, params_init, method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            x_new = _reconstruct_x(result.x)
            val_new = _objective(x_new)
            if val_new < best_val:
                best_val = val_new
                best_x = x_new
    except Exception:
        pass

    if _check_deadline():
        return best_x

    # Refine with time-domain L-BFGS-B using soft-max
    def _time_domain_obj_grad(x, beta=100.0):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(x)
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        soft_max, weights = _smooth_max_grad(abs_ac, beta)
        val = soft_max / (s * s)
        
        g = weights * np.sign(ac)
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        d_soft_max_dx = term1 + term2
        
        grad = (1.0 / (s * s)) * d_soft_max_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
        return val, grad

    try:
        result = minimize(_time_domain_obj_grad, best_x, method='L-BFGS-B', jac=True,
                          options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            x_new = result.x
            val_new = _objective(x_new)
            if val_new < best_val:
                best_val = val_new
                best_x = x_new
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
