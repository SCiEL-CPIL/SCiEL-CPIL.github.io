"""
Strategy: Spectral-phase parameterization.
- Fix |H[k]| = 1 for all k (flat magnitude spectrum), so sum(h) is determined by phase.
- h = IFFT(exp(i*phi)) where phi is a 400-vector of phase angles.
- Autoconvolution c = IFFT(|FFT(h)|^2) = IFFT(FFT(h)*conj(FFT(h)))
  With flat magnitude, |FFT(h)|^2 = 1 for all k, so c = IFFT(ones) = delta (trivial).
  Wait -- that's not right. If |H[k]|=1 for all k, then c[k] = IFFT(|H|^2)[k] = IFFT(1)[k] = delta[k].
  That gives max(c) = 1 and sum(h) = H[0] = exp(i*phi[0])... but h must be real.
  
  Actually for real h, H[k] = conj(H[N-k]), so we can't independently set all phases.
  Let me reconsider.

Better approach: 
- h is real, length N. FFT(h) has Hermitian symmetry.
- We parameterize h via its DFT: H[k] = r[k] * exp(i*theta[k]) with r[k] = r[N-k], theta[k] = -theta[N-k].
- Free parameters: r[0..N/2] and theta[1..N/2-1] (theta[0]=0, theta[N/2]=0 for real).
- Objective: value = |2*N*max(c) / S^2| where c = IFFT(|H|^2), S = sum(h) = H[0].
- With r[k] free, this is a (N/2+1)-dim optimization in the spectral domain.
- Gradient of max(c) w.r.t. H can be computed via FFT: d c[k]/d H[m] involves H.

Actually, simplest effective approach:
- Use h = IFFT(H) where H[k] = exp(i * phi[k]) for k=0..N-1, with phi[0]=0, phi[k]=-phi[N-k].
- This gives |H[k]|=1, so c = IFFT(ones) = [1, 0, 0, ..., 0] (a delta). max(c)=1.
- sum(h) = H[0] = 1. value = |2*N*1/1| = 2N = 800. That's terrible.

The issue: flat magnitude gives a delta autocorrelation, which is actually PERFECT (max sidelobe = 0).
But the "max" in np.max(conv(h,h)) picks up the peak at lag 0, which is N (when h is normalized).
Wait, let me recompute: if h = IFFT(exp(i*phi)) with real constraint, then:
- conv(h,h) has peak at center = sum(h^2) = N (by Parseval, since |H[k]|^2=1).
- sum(h) = H[0] = 1 (if phi[0]=0).
- value = |2*N*N / 1^2| = 2*N^2 = 320000. Terrible.

The problem is the denominator sum(h)^2. We need sum(h) to be large relative to the peak.

New plan: Don't fix magnitude. Instead:
- Parameterize h = a * cos(2*pi*f*x + phase) + b, or use a low-dimensional parametric family.
- OR: Use the insight that value = 2*N*max(c)/S^2. If we scale h -> alpha*h, value is invariant.
  So we can fix S = sum(h) = 1 (or any constant) and minimize max(c).
- With S fixed, minimize max(c) where c = conv(h,h). This is a convex problem!
  min max_k c[k] subject to sum(h) = 1, h real.
  This is a linear program (LP) in the variables h and t:
    min t
    s.t. c[k] <= t for all k
         sum(h) = 1
  where c = conv(h,h) is quadratic in h... so it's a QP, not LP.

  But we can use the fact that c[k] = h^T T_k h where T_k is a Toeplitz matrix.
  min max_k h^T T_k h s.t. 1^T h = 1 is a non-convex QP in general.

  However, if we relax: let x = h (no relaxation needed), and use bisection on t:
  For a given t, check if there exists h with sum(h)=1 and c[k] <= t for all k.
  This is a feasibility problem with quadratic constraints -- still hard.

PRACTICAL APPROACH that's structurally different:
- Use a parametric family: h[j] = A + B * cos(2*pi*m*j/N + phi) for small m.
  This gives a 3-parameter family (A, B, phi) or 4 if m is also free.
- The autocorrelation of a pure cosine is a cosine at the same frequency.
- Optimize over (m, A, B, phi) to minimize value.
- Then refine with a few iterations of projected gradient on the full h.

Even better structural change:
- Recognize that the optimal h for minimizing max(c)/S^2 with S fixed is related to 
  minimum-peak-sidelobe sequences. These are well-studied.
- Use a chirp (linear frequency sweep): h[j] = exp(i*pi*j^2/N) made real.
  The autocorrelation of a chirp is approximately a delta (Barker-like property).
- Optimize the chirp parameters (quadratic coefficient, windowing) via a simple 2-3 dim search.
- Then do a final round of coordinate descent on the full h.

This is materially different: low-dimensional parametric search + spectral insight, vs. 
full 400-dim gradient descent with annealing.
"""

import numpy as np
import time

N = 400
_DEADLINE = 148.0  # wall-clock limit with safety margin


def _exact_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _chirp_h(a, b, window_sigma):
    """
    Generate a real chirp-like signal.
    h[j] = cos(pi * a * j^2 / N + b) * window(j)
    where window is a Gaussian or rectangular.
    """
    j = np.arange(N, dtype=float)
    phase = np.pi * a * j * j / N + b
    h = np.cos(phase)
    if window_sigma > 0:
        center = N / 2.0
        w = np.exp(-0.5 * ((j - center) / window_sigma) ** 2)
        h = h * w
    return h


def _search_parametric():
    """Search over chirp parameters to find a good initial h."""
    best_h = np.ones(N)
    best_val = _exact_value(best_h)
    
    # Grid search over chirp parameters
    for a in np.linspace(0.0, 4.0, 40):
        for b in np.linspace(0.0, 2 * np.pi, 20):
            for ws in [0.0, N / 4.0, N / 6.0, N / 3.0]:
                h = _chirp_h(a, b, ws)
                s = np.sum(h)
                if abs(s) < 1e-10:
                    continue
                # Normalize so sum = 1 (scale invariance)
                h_norm = h / s
                v = _exact_value(h_norm)
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_h = h_norm.copy()
    
    return best_h, best_val


def _conjugate_gradient_phase(h, max_iter=500, tol=1e-8):
    """
    Conjugate gradient on the phase of the FFT of h.
    We fix the magnitude spectrum and optimize phases to minimize max(c).
    
    Actually, let's do a simpler but effective approach:
    Iterative reweighting / subgradient method on the full h with 
    conjugate gradient acceleration.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-10:
        return h
    
    # Normalize: sum(h) = 1
    h = h / s
    
    conv = np.convolve(h, h)
    M = np.max(conv)
    
    # Subgradient of max(c) w.r.t. h
    mask = (conv == M)
    n_max = np.sum(mask)
    if n_max == 0:
        return h
    grad_M = mask.astype(float) / n_max
    
    # dM/dh[j] = 2 * sum_{k} grad_M[k] * h[k-j] (with bounds)
    # Use FFT for speed: dM/dh = 2 * IFFT(FFT(grad_M_padded) * conj(FFT(h_padded)))
    # Actually: dM/dh = 2 * correlate(grad_M, h) truncated to N
    
    # Efficient computation via FFT:
    size = 2 * N
    H = np.fft.fft(h, size)
    G = np.fft.fft(grad_M, size)
    # dM/dh[j] = 2 * sum_k grad_M[k] * h[k-j]
    # This is 2 * (grad_M * h)[j] where * is cross-correlation
    # = 2 * IFFT(conj(FFT(grad_M)) * FFT(h)) ... let me be careful
    # cross-correlation: (a * b)[j] = sum_k a[k] * b[k-j]
    # FFT: IFFT(conj(FFT(a)) * FFT(b)) gives (a * b)[j] for circular corr
    # For linear, we need zero-padding to size >= len(a)+len(b)-1
    
    # grad_M has length 2N-1, h has length N. Need size >= 3N-2. Use 4N for safety.
    size = 4 * N
    H = np.fft.fft(h, size)
    G = np.fft.fft(grad_M, size)
    # (grad_M * h)[j] = IFFT(conj(G) * H)[j] for j=0..N-1 (linear cross-corr)
    dM_dh_full = np.fft.ifft(np.conj(G) * H).real
    dM_dh = 2.0 * dM_dh_full[:N]
    
    # Gradient of value = |2*N*M/S^2| w.r.t. h, with S=1 fixed (we renormalize after each step)
    # If we fix S=1, value = 2*N*M, gradient = 2*N * dM_dh
    # But we need to account for the constraint sum(h)=1.
    # Projected gradient: h_new = h - lr * (dM_dh - mean(dM_dh)) to keep sum unchanged
    # Then renormalize.
    
    best_h = h.copy()
    best_val = _exact_value(best_h)
    
    lr = 0.01
    prev_grad = None
    beta = 0.0
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        conv = np.convolve(h, h)
        M = np.max(conv)
        mask = (conv == M)
        n_max = np.sum(mask)
        if n_max == 0:
            break
        grad_M = mask.astype(float) / n_max
        
        size = 4 * N
        H = np.fft.fft(h, size)
        G = np.fft.fft(grad_M, size)
        dM_dh_full = np.fft.ifft(np.conj(G) * H).real
        dM_dh = 2.0 * dM_dh_full[:N]
        
        # Project to maintain sum(h) = 1
        grad = dM_dh - np.mean(dM_dh)
        
        # Conjugate gradient
        if prev_grad is not None:
            num = np.dot(grad, grad)
            den = np.dot(prev_grad, prev_grad)
            if den > 1e-15:
                beta = num / den
            else:
                beta = 0.0
        direction = grad + beta * prev_grad
        prev_grad = grad
        
        # Line search (simple backtracking)
        step = lr
        for _ in range(20):
            h_try = h - step * direction
            s_try = np.sum(h_try)
            if abs(s_try) < 1e-10:
                step *= 0.5
                continue
            h_try = h_try / s_try
            v_try = _exact_value(h_try)
            if np.isfinite(v_try) and v_try < best_val:
                best_val = v_try
                best_h = h_try.copy()
                h = h_try
                break
            step *= 0.5
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    
    return best_h


def _coordinate_descent(h, n_iters=5000):
    """Simple coordinate descent refinement."""
    rng = np.random.default_rng(123)
    best_h = h.copy()
    best_val = _exact_value(best_h)
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 3:
            break
        
        idx = rng.integers(0, N)
        s = np.sum(best_h)
        if abs(s) < 1e-10:
            break
        
        # Try perturbations
        for delta in [0.005, -0.005, 0.02, -0.02, 0.05, -0.05]:
            h_try = best_h.copy()
            h_try[idx] += delta
            s_try = np.sum(h_try)
            if abs(s_try) < 1e-10:
                continue
            v_try = _exact_value(h_try)
            if np.isfinite(v_try) and v_try < best_val - 1e-12:
                best_val = v_try
                best_h = h_try
                break
    
    return best_h


def construct():
    """
    Construct step heights using spectral-phase / parametric-chirp strategy.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Phase 1: Parametric chirp search (fast, low-dim)
    best_h, best_val = _search_parametric()
    
    # Phase 2: Conjugate gradient refinement on full h
    if time.time() < deadline - 20:
        refined = _conjugate_gradient_phase(best_h, max_iter=300)
        v_ref = _exact_value(refined)
        if np.isfinite(v_ref) and v_ref < best_val:
            best_val = v_ref
            best_h = refined
    
    # Phase 3: More CG iterations from best point
    if time.time() < deadline - 10:
        refined2 = _conjugate_gradient_phase(best_h, max_iter=200)
        v_ref2 = _exact_value(refined2)
        if np.isfinite(v_ref2) and v_ref2 < best_val:
            best_val = v_ref2
            best_h = refined2
    
    # Phase 4: Coordinate descent polish
    if time.time() < deadline - 3:
        polished = _coordinate_descent(best_h, n_iters=3000)
        v_pol = _exact_value(polished)
        if np.isfinite(v_pol) and v_pol < best_val:
            best_val = v_pol
            best_h = polished
    
    # Final validation
    final_val = _exact_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
