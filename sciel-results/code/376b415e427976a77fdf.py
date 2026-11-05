"""Program using Convex QP formulation to minimize L2 norm of autoconvolution."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    max_val = np.max(conv)
    raw = 2.0 * N * max_val / (s * s)
    return abs(raw)

def _qp_objective_and_grad(h):
    """
    Objective: minimize ||h * h||_2^2 subject to sum(h) = 1.
    Let c = h * h (full autoconvolution, length 2N-1).
    Objective J = sum(c^2) = sum_{k} (sum_{i+j=k} h_i h_j)^2.
    
    This is a quartic form, not quadratic in h.
    However, the directive asks for a "convex Quadratic Program".
    Strictly speaking, minimizing the L2 norm of the autoconvolution is non-convex.
    But often, minimizing the L2 norm of the *Fourier transform* magnitude squared (which is related) or using a surrogate can work.
    
    Actually, let's look at the structure again.
    value = 2*N*max(conv)/sum(h)^2.
    If we fix sum(h)=1, we want to minimize max(conv).
    Minimizing max(conv) is equivalent to minimizing the L_infinity norm of conv.
    This is a non-convex problem generally.
    
    However, the prompt suggests: "Formulate the problem as a convex Quadratic Program (QP) that minimizes the squared L2 norm of the autoconvolution c = h * h subject to the linear constraint sum h_i = 1".
    
    Is min ||c||_2^2 convex in h?
    c_k = sum_{i+j=k} h_i h_j.
    ||c||_2^2 = sum_k (sum_{i+j=k} h_i h_j)^2.
    This is a homogeneous polynomial of degree 4.
    It is not a quadratic program in h.
    
    Perhaps the intent is to minimize ||h||_2^2 subject to constraints on the autoconvolution? Or maybe minimize the autoconvolution peak via a QP surrogate?
    
    Let's re-read: "minimizes the squared L2 norm of the autoconvolution c = h * h".
    If the prompt insists this is a QP, it might be mistaken about the convexity, or it implies a specific relaxation.
    However, I can still use SLSQP to minimize this non-convex objective, as SLSQP can handle non-convex problems (though no global guarantee).
    The key is that minimizing the L2 norm of the autoconvolution tends to spread out the energy of the correlation, potentially lowering the peak compared to a spike.
    
    Let's define:
    f(h) = ||h * h||_2^2
    g(h) = sum(h) - 1 = 0
    
    Gradient of f(h):
    Let c = h * h.
    f = sum(c^2).
    df/dh_k = 2 * sum(c * dc/dh_k).
    dc_m/dh_k = h_{m-k} + h_{m+k} (if indices valid).
    So df/dh_k = 2 * sum_m c_m * (h_{m-k} + h_{m+k}).
    This can be computed via convolution.
    Let u = c * c (autoconvolution of c)? No.
    df/dh = 2 * (c * c) ? Let's check.
    (c * c)_k = sum_m c_m c_{k-m}.
    c_m = sum_i h_i h_{m-i}.
    This path is tricky.
    
    Alternative: Use FFT.
    H = FFT(h).
    C = H * H (element-wise square of FFT of h is NOT the FFT of autoconvolution. FFT(h*h) = H*H).
    Wait, if c = h * h (linear convolution), then C = FFT(c) = H * H.
    f = sum(c^2) = (1/2N) * sum(|C|^2) by Parseval?
    sum(c^2) = sum_{k} c_k^2.
    Parseval: sum |c_k|^2 = (1/M) sum |C_m|^2 where M is FFT size.
    Since c is real, sum c_k^2 = sum |c_k|^2.
    So f = (1/M) sum |H_m * H_m|^2 = (1/M) sum |H_m|^4.
    
    So we want to minimize sum |H_m|^4 subject to sum(h)=1.
    This is a "flattening" objective in the frequency domain.
    Minimizing the L4 norm of the spectrum.
    
    Let's compute the gradient of f = (1/M) sum |H_m|^4 with respect to h.
    H_m = sum_n h_n e^{-i 2 pi m n / M}.
    |H_m|^2 = H_m H_m^*.
    d/dh_k (|H_m|^2) = H_m^* e^{-i 2 pi m k / M} + H_m e^{i 2 pi m k / M} = 2 Re(H_m^* e^{-i 2 pi m k / M}).
    
    f = (1/M) sum_m (|H_m|^2)^2.
    df/dh_k = (1/M) sum_m 2 |H_m|^2 * (2 Re(H_m^* e^{-i 2 pi m k / M}))
             = (4/M) Re( sum_m |H_m|^2 H_m^* e^{-i 2 pi m k / M} )
    
    Let Q_m = |H_m|^2 H_m^*.
    Then sum_m Q_m e^{-i 2 pi m k / M} is the inverse FFT of Q.
    Let q = IFFT(Q).
    Then df/dh_k = (4/M) Re(q_k).
    
    This is very efficient to compute.
    
    So the strategy is:
    1. Minimize f(h) = sum |FFT(h)|^4 subject to sum(h) = 1.
    2. This encourages a flat spectrum, which corresponds to a "white noise" like signal in time, but with positive sum.
    3. A flat spectrum means the autoconvolution c = IFFT(|H|^2) is close to a delta function?
       Wait. c = h * h. FFT(c) = H^2.
       If H is flat (constant magnitude), then H^2 is also flat (constant magnitude, phase doubled).
       If H is flat, c is a delta function?
       No. If H is constant, h is a delta function.
       If h is a delta function, c is a delta function. max(c) is large.
       
       Let's re-evaluate.
       We want to MINIMIZE max(c).
       If h is spread out (like a box), c is a triangle. max(c) is smaller than the delta case?
       For h = ones(N), sum = N. c is triangle with peak N.
       value = 2*N*N / N^2 = 2.
       
       If h is random noise, sum ~ 0.
       If we normalize sum=1, h is small.
       
       Let's look at the objective again.
       value = 2*N*max(c) / S^2.
       
       If we minimize ||c||_2^2 subject to S=1.
       For h = ones(N)/N, S=1. c is triangle with peak 1/N.
       ||c||_2^2 is roughly sum (triangle)^2.
       
       Is minimizing ||c||_2^2 good for minimizing max(c)?
       Usually, yes, if the function is smooth. Spreading the mass of c reduces the peak.
       
       So the QP (or rather, the quartic minimization) approach is:
       Minimize sum |FFT(h)|^4 subject to sum(h)=1.
       
       Let's implement this.
    """
    M = 2 * N - 1
    H = np.fft.rfft(h, M)
    H_abs_sq = np.abs(H) ** 2
    # f = (1/M) * sum(H_abs_sq^2)
    f = np.sum(H_abs_sq ** 2) / M
    
    # Gradient
    # Q_m = |H_m|^2 * conj(H_m)
    Q = H_abs_sq * np.conj(H)
    # q = IFFT(Q)
    # Note: scipy.fft.irfft expects the spectrum.
    # Q is for rfft indices 0..M//2.
    # We need to reconstruct the full spectrum for IFFT?
    # irfft takes the rfft spectrum.
    # But Q is complex. irfft expects a complex array representing the positive frequencies.
    # The output of irfft is real.
    # The formula derived: df/dh_k = (4/M) Re(q_k) where q = IFFT(Q_full).
    # Since h is real, H is symmetric. Q is symmetric?
    # Q_m = |H_m|^2 H_m^*.
    # Q_{M-m} = |H_{M-m}|^2 H_{M-m}^* = |H_m|^2 H_m.
    # So Q is conjugate symmetric.
    # We can use irfft on the rfft part of Q.
    
    q_r = np.fft.irfft(Q, M)
    grad = (4.0 / M) * q_r
    
    return f, grad

def _constraints(h):
    return np.sum(h) - 1.0

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Initial guess: normalized ones
    h0 = np.ones(N) / N
    
    # Run SLSQP to minimize L2 norm of autoconvolution (via L4 norm of spectrum)
    # subject to sum(h) = 1
    
    if _check_time():
        try:
            res = minimize(
                _qp_objective_and_grad,
                h0,
                method='SLSQP',
                jac=True,
                constraints=[{'type': 'eq', 'fun': _constraints}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            h_candidate = res.x
            # Normalize to ensure sum is exactly 1 (SLSQP might be slightly off)
            s = np.sum(h_candidate)
            if abs(s) > 1e-12:
                h_candidate = h_candidate / s
            
            val_candidate = _compute_value(h_candidate)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate
                
        except Exception:
            pass
    
    # If SLSQP failed or didn't improve, try a few other initializations
    if _check_time():
        inits = [
            np.ones(N) / N,
            np.linspace(-1, 1, N),
            np.exp(-10 * (np.linspace(-1, 1, N))**2),
            np.random.default_rng(42).standard_normal(N)
        ]
        
        for init in inits:
            if not _check_time():
                break
            
            h0 = init.copy()
            s = np.sum(h0)
            if abs(s) < 1e-12:
                continue
            h0 = h0 / s
            
            try:
                res = minimize(
                    _qp_objective_and_grad,
                    h0,
                    method='SLSQP',
                    jac=True,
                    constraints=[{'type': 'eq', 'fun': _constraints}],
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                
                h_candidate = res.x
                s = np.sum(h_candidate)
                if abs(s) > 1e-12:
                    h_candidate = h_candidate / s
                
                val_candidate = _compute_value(h_candidate)
                
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
                    
            except Exception:
                continue
    
    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
