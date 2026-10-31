import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _logsumexp_surrogate(h, beta=50.0):
    """
    Smooth surrogate for max|autoconv| using log-sum-exp.
    This makes the objective differentiable, enabling gradient-based optimization.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    # Log-sum-exp for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0
    lse = max_val + np.log(np.sum(np.exp(beta * (abs_ac - max_val)))) / beta
    return lse

def _surrogate_objective(x, beta=50.0):
    """
    Differentiable objective: log-sum-exp(autoconv) / (sum h)^2
    """
    h = np.tanh(x)  # Map real values to (-1, 1)
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    lse = _logsumexp_surrogate(h, beta)
    return lse / (s * s)

def _surrogate_gradient(x, beta=50.0, eps=1e-7):
    """
    Numerical gradient of the surrogate objective.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_surrogate_objective(x_plus, beta) - _surrogate_objective(x_minus, beta)) / (2 * eps)
    return grad

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0

def _generate_mersenne_legendre(q, N):
    """
    Generate a sequence from the Legendre symbol of a Mersenne prime p = 2^q - 1.
    Mersenne primes have special algebraic structure with provably optimal
    asymptotic autocorrelation bounds.
    """
    p = (1 << q) - 1
    # Verify it's actually prime (for small q this is known)
    # For q=2: p=3, q=3: p=7, q=5: p=31, q=7: p=127, q=13: p=8191
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _generate_legendre_sequence(p, N):
    """Generate a sequence of length N based on the Legendre symbol of a prime p."""
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _local_sign_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses incremental autoconvolution updates (O(N) per flip).
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1

    ac = np.convolve(h, h, mode='full')
    max_abs_ac = float(np.max(np.abs(ac)))
    s = float(np.sum(h))
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            old = h[i]
            delta = -2.0 * old

            s_new = s + delta
            if abs(s_new) < 1e-6:
                continue

            ac_new = ac.copy()
            ac_new[i:i+n] += delta * h
            ac_new[n-1-i:2*n-1-i] += delta * h
            max_abs_new = float(np.max(np.abs(ac_new)))
            val_new = max_abs_new / (s_new * s_new)

            if val_new < best_val:
                h[i] = -old
                ac = ac_new
                s = s_new
                best_val = val_new
                max_abs_ac = max_abs_new
                improved = True
        if not improved:
            break
    return h, best_val

def _gradient_based_optimize(h0, beta=50.0, max_iter=100, deadline=None):
    """
    Use L-BFGS-B to optimize the log-sum-exp surrogate objective.
    Start from h0 (±1 values), map through arctanh to get initial x.
    """
    n = len(h0)
    # Map h0 to x such that tanh(x) ≈ h0
    # tanh(x) = h0 => x = arctanh(h0)
    # But h0 is ±1, so arctanh is undefined. Use large values instead.
    x0 = np.zeros(n)
    for i in range(n):
        if h0[i] > 0:
            x0[i] = 3.0  # tanh(3) ≈ 0.995
        else:
            x0[i] = -3.0
    
    result = minimize(
        _surrogate_objective,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        jac=_surrogate_gradient,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if deadline is not None and time.time() > deadline:
        # Return the best we have so far
        h_opt = np.sign(np.tanh(result.x))
        h_opt[h_opt == 0] = 1.0
        return h_opt, _score(h_opt)
    
    h_opt = np.sign(np.tanh(result.x))
    h_opt[h_opt == 0] = 1.0
    return h_opt, _score(h_opt)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Mersenne primes: 2^q - 1 where q is prime
    # q=2: p=3, q=3: p=7, q=5: p=31, q=7: p=127, q=13: p=8191
    mersenne_qs = [2, 3, 5, 7, 13]
    
    # Other primes for variety
    other_primes = [397, 401, 389, 409, 383, 419, 431, 433, 439, 443]
    
    n_restarts = 200
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        if r < len(mersenne_qs):
            # Mersenne prime Legendre sequences (primary seeds)
            q = mersenne_qs[r]
            h = _generate_mersenne_legendre(q, N)
        elif r < len(mersenne_qs) + len(other_primes):
            # Other primes
            p = other_primes[r - len(mersenne_qs)]
            h = _generate_legendre_sequence(p, N)
        elif r % 3 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 3 == 1:
            # Low frequency cosine
            k = (r // 3) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        else:
            # Block structure
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # First do local sign flip
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
        
        # Then try gradient-based optimization if time permits
        if time.time() < deadline - 10.0:
            h_grad, val_grad = _gradient_based_optimize(h_opt, beta=50.0, max_iter=50, deadline=deadline)
            if val_grad < val_opt:
                h_opt = h_grad
                val_opt = val_grad
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Final polish with both methods
    if time.time() < deadline - 5.0:
        best_h, best_val = _local_sign_flip(best_h, max_passes=10, deadline=deadline)
        if time.time() < deadline - 3.0:
            h_grad, val_grad = _gradient_based_optimize(best_h, beta=50.0, max_iter=30, deadline=deadline)
            if val_grad < best_val:
                best_h = h_grad
                best_val = val_grad

    return best_h

# EVOLVE-BLOCK-END
