"""Structural change: parameterize h_k = |P(omega^k) Q(omega^k)|^2 with low-degree complex polynomials.
This enforces non-negativity, reduces dimensionality, and exploits spectral factorization of autoconvolution."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE - time.time()


def _ratio_from_h(h):
    """Compute raw objective: (L2^2) / (L1 * sup(|autoconv|))."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))  # h is nonnegative by construction
    if l1 <= 0.0:
        return 0.0
    ac = np.convolve(h, h, mode='full')
    s = float(np.max(np.abs(ac)))
    if s <= 0.0:
        return 0.0
    return l2sq / (l1 * s)


def _ratio_from_coeffs(coeffs):
    """
    Given real vector coeffs representing complex coefficients of P and Q,
    compute h_k = |P(w^k) Q(w^k)|^2 and return the ratio.
    
    coeffs layout: [p0_re, p0_im, p1_re, p1_im, ..., q0_re, q0_im, ...]
    P has degree dP, Q has degree dQ.
    """
    dP = 4
    dQ = 4
    nP = dP + 1
    nQ = dQ + 1
    
    # Extract P coefficients
    p_coeffs = np.zeros(nP, dtype=complex)
    for k in range(nP):
        p_coeffs[k] = coeffs[2*k] + 1j * coeffs[2*k+1]
    
    # Extract Q coefficients
    q_coeffs = np.zeros(nQ, dtype=complex)
    for k in range(nQ):
        q_coeffs[k] = coeffs[2*(nP + k)] + 1j * coeffs[2*(nP + k) + 1]
    
    # Evaluate P and Q at omega^k for k=0..N-1
    omega = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # P(w^k) for each k
    p_vals = np.zeros(N, dtype=complex)
    for k in range(N):
        # Horner's method
        val = 0.0 + 0.0j
        for c in p_coeffs[::-1]:
            val = val * omega[k] + c
        p_vals[k] = val
    
    # Q(w^k) for each k
    q_vals = np.zeros(N, dtype=complex)
    for k in range(N):
        val = 0.0 + 0.0j
        for c in q_coeffs[::-1]:
            val = val * omega[k] + c
        q_vals[k] = val
    
    # h_k = |P(w^k) Q(w^k)|^2
    h = np.abs(p_vals * q_vals) ** 2
    
    return _ratio_from_h(h)


def _objective_and_grad(coeffs):
    """
    Returns (-ratio, gradient) for L-BFGS-B.
    Uses finite differences for gradient since analytical is complex.
    """
    dP = 4
    dQ = 4
    nP = dP + 1
    nQ = dQ + 1
    dim = 2 * (nP + nQ)
    
    h0 = _ratio_from_coeffs(coeffs)
    
    # Finite difference gradient
    grad = np.zeros(dim)
    eps = 1e-6
    for i in range(dim):
        coeffs_plus = coeffs.copy()
        coeffs_plus[i] += eps
        h_plus = _ratio_from_coeffs(coeffs_plus)
        grad[i] = -(h_plus - h0) / eps
    
    return -h0, grad


def _ratio_from_coeffs_fd_only(coeffs):
    """Just compute the ratio (no gradient)."""
    return _ratio_from_coeffs(coeffs)


def _gaussian_profile(center, width):
    """Generate a smooth Gaussian-like profile as initial guess."""
    x = np.arange(N, dtype=float)
    v = np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    return v


def _find_initial_coeffs_from_h(h, dP=4, dQ=4):
    """
    Try to find complex polynomial coefficients P, Q such that |P(w^k)Q(w^k)|^2 ≈ h_k.
    Use a simple approach: take square root of h, then try to fit a low-degree polynomial.
    For initialization, we'll just use a random-ish but structured initialization.
    """
    nP = dP + 1
    nQ = dQ + 1
    dim = 2 * (nP + nQ)
    
    # Simple initialization: make P and Q both have a dominant DC term
    coeffs = np.zeros(dim)
    
    # Set P to be approximately constant (all energy at DC)
    coeffs[0] = 1.0  # p0_re
    # Set Q similarly
    coeffs[2 * nP] = 1.0  # q0_re
    
    # Add small perturbations to break symmetry
    rng = np.random.RandomState(42)
    coeffs += rng.randn(dim) * 0.01
    
    return coeffs


def _local_optimize_from_coeffs(coeffs0, max_iter=2000):
    """Run L-BFGS-B from a given initial coefficient vector."""
    try:
        res = minimize(
            _objective_and_grad,
            coeffs0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10},
        )
        return res.x, res.fun
    except Exception:
        return coeffs0, _ratio_from_coeffs_fd_only(coeffs0)


def construct():
    dP = 4
    dQ = 4
    nP = dP + 1
    nQ = dQ + 1
    dim = 2 * (nP + nQ)
    
    best_val = 0.0
    best_coeffs = None
    
    # Phase 1: Multi-start with structured initializations
    starts = []
    
    # Start 1: All DC
    c1 = np.zeros(dim)
    c1[0] = 1.0
    c1[2 * nP] = 1.0
    starts.append(c1)
    
    # Start 2: Slight linear tilt
    c2 = np.zeros(dim)
    c2[0] = 1.0
    c2[2] = 0.1  # p1_re
    c2[2 * nP] = 1.0
    c2[2 * nP + 2] = 0.1  # q1_re
    starts.append(c2)
    
    # Start 3: Random structured
    rng = np.random.RandomState(123)
    for seed_idx in range(5):
        c = np.zeros(dim)
        c[0] = 1.0
        c[2 * nP] = 1.0
        # Small random perturbations
        c += rng.randn(dim) * 0.05
        c[0] = 1.0  # keep DC dominant
        c[2 * nP] = 1.0
        starts.append(c)
    
    # Evaluate and optimize from each start
    for i, c0 in enumerate(starts):
        if _time_left() < 10.0:
            break
        
        # First evaluate the initial point
        val = _ratio_from_coeffs_fd_only(c0)
        if val > best_val:
            best_val = val
            best_coeffs = c0.copy()
        
        # Try local optimization
        if _time_left() < 15.0:
            break
        
        c_opt, val_opt = _local_optimize_from_coeffs(c0, max_iter=1500)
        if val_opt > best_val:
            best_val = val_opt
            best_coeffs = c_opt.copy()
    
    # Phase 2: If we have a good point, do a finer polish
    if best_coeffs is not None and _time_left() > 20.0:
        c_polish, val_polish = _local_optimize_from_coeffs(best_coeffs, max_iter=3000)
        if val_polish > best_val:
            best_val = val_polish
            best_coeffs = c_polish.copy()
    
    # Phase 3: Try a completely different family - use real-valued parameterization
    # h_k = g_k^2 where g is a smooth function parameterized by Fourier coefficients
    if _time_left() > 30.0:
        # Parameterize g_k as a sum of cosines: g_k = sum_j a_j cos(2*pi*j*k/N) + b_j sin(2*pi*j*k/N)
        # Keep h_k = g_k^2 >= 0
        # Limit to low frequencies to keep smoothness
        max_freq = 3
        dim2 = 2 * (max_freq + 1)  # a_0..a_maxfreq, b_1..b_maxfreq (b_0=0)
        
        def _ratio_from_fourier(params):
            g = np.zeros(N)
            for j in range(max_freq + 1):
                a_j = params[j]
                g += a_j * np.cos(2 * np.pi * j * np.arange(N) / N)
                if j > 0:
                    b_j = params[max_freq + 1 + (j - 1)]
                    g += b_j * np.sin(2 * np.pi * j * np.arange(N) / N)
            h = g ** 2
            return _ratio_from_h(h)
        
        def _fourier_obj_and_grad(params):
            h0 = _ratio_from_fourier(params)
            grad = np.zeros(dim2)
            eps = 1e-5
            for i in range(dim2):
                p_plus = params.copy()
                p_plus[i] += eps
                h_plus = _ratio_from_fourier(p_plus)
                grad[i] = -(h_plus - h0) / eps
            return -h0, grad
        
        # Multi-start for Fourier parameterization
        for trial in range(8):
            if _time_left() < 10.0:
                break
            rng2 = np.random.RandomState(100 + trial)
            p0 = rng2.randn(dim2) * 0.5
            p0[0] = 1.0  # DC term
            
            val0 = _ratio_from_fourier(p0)
            if val0 > best_val:
                best_val = val0
                # Convert to h for final output
                g = np.zeros(N)
                for j in range(max_freq + 1):
                    a_j = p0[j]
                    g += a_j * np.cos(2 * np.pi * j * np.arange(N) / N)
                    if j > 0:
                        b_j = p0[max_freq + 1 + (j - 1)]
                        g += b_j * np.sin(2 * np.pi * j * np.arange(N) / N)
                best_h = g ** 2
                best_h = best_h / np.max(best_h) if np.max(best_h) > 0 else best_h
            
            if _time_left() < 15.0:
                break
            
            try:
                res2 = minimize(
                    _fourier_obj_and_grad,
                    p0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10},
                )
                val2 = -res2.fun
                if val2 > best_val:
                    best_val = val2
                    g = np.zeros(N)
                    for j in range(max_freq + 1):
                        a_j = res2.x[j]
                        g += a_j * np.cos(2 * np.pi * j * np.arange(N) / N)
                        if j > 0:
                            b_j = res2.x[max_freq + 1 + (j - 1)]
                            g += b_j * np.sin(2 * np.pi * j * np.arange(N) / N)
                    best_h = g ** 2
                    best_h = best_h / np.max(best_h) if np.max(best_h) > 0 else best_h
            except Exception:
                pass
    
    # Determine final output
    if best_coeffs is not None:
        # Convert complex polynomial coefficients to h
        dP = 4
        dQ = 4
        nP = dP + 1
        nQ = dQ + 1
        
        p_coeffs = np.zeros(nP, dtype=complex)
        for k in range(nP):
            p_coeffs[k] = best_coeffs[2*k] + 1j * best_coeffs[2*k+1]
        
        q_coeffs = np.zeros(nQ, dtype=complex)
        for k in range(nQ):
            q_coeffs[k] = best_coeffs[2*(nP + k)] + 1j * best_coeffs[2*(nP + k) + 1]
        
        omega = np.exp(-2j * np.pi * np.arange(N) / N)
        
        p_vals = np.zeros(N, dtype=complex)
        for k in range(N):
            val = 0.0 + 0.0j
            for c in p_coeffs[::-1]:
                val = val * omega[k] + c
            p_vals[k] = val
        
        q_vals = np.zeros(N, dtype=complex)
        for k in range(N):
            val = 0.0 + 0.0j
            for c in q_coeffs[::-1]:
                val = val * omega[k] + c
            q_vals[k] = val
        
        h = np.abs(p_vals * q_vals) ** 2
    else:
        # Fallback: use the best_h from Fourier if available
        try:
            h = best_h
        except NameError:
            # Last resort: uniform
            h = np.ones(N)
    
    # Ensure nonnegative and normalize
    h = np.maximum(h, 0.0)
    m = np.max(h)
    if m > 0:
        h = h / m
    else:
        h = np.ones(N) / N
    
    return [float(v) for v in h]


if __name__ == '__main__':
    r = construct()
    print(len(r), _ratio_from_h(r))
