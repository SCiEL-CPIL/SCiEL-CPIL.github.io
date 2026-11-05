import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _evaluate(h):
    """Compute the objective value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (s * s))

def _smooth_objective(x, beta):
    """
    Smooth surrogate: use log-sum-exp to approximate max(conv).
    We work with log-sum-exp of the convolution entries.
    Since we want to minimize max(conv)/sum(h)^2, and by scale invariance
    we can fix sum(h)=1. Then we minimize max(conv).
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(x, x)
    conv_max = np.max(conv)
    logsumexp = conv_max + np.log(np.sum(np.exp(beta * (conv - conv_max)))) / beta
    return logsumexp

def _smooth_grad(x, beta):
    """Gradient of the smooth objective."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    conv = np.convolve(x, x)
    conv_max = np.max(conv)
    exp_vals = np.exp(beta * (conv - conv_max))
    denom = np.sum(exp_vals)
    weights = exp_vals / denom
    
    # grad[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * x[j] where i+j < 2N-1
    grad = np.zeros(N)
    for i in range(N):
        max_j = min(N, 2*N-1-i)
        if max_j > 0:
            grad[i] = 2.0 * np.dot(weights[i:i+max_j], x[:max_j])
    return grad

def _quadratic_residue_seq(p, shift=0):
    """
    Generate a +/-1 sequence based on quadratic residues modulo prime p.
    h[k] = 1 if (k+shift)^2 mod p is a quadratic residue (including 0), -1 otherwise.
    We pad/truncate to length N.
    """
    # Find smallest prime >= p
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0: return False
        return True
    prime = p
    while not is_prime(prime):
        prime += 1
    
    h = np.zeros(N)
    for k in range(N):
        val = (k + shift) % prime
        # Legendre symbol
        if val == 0:
            h[k] = 1.0
        else:
            # Check if val is a quadratic residue mod prime
            # val^((prime-1)/2) mod prime == 1 if residue, -1 if non-residue
            if pow(val, (prime-1)//2, prime) == 1:
                h[k] = 1.0
            else:
                h[k] = -1.0
    return h

def _spike_candidate():
    """Deterministic high-quality candidate: two-spike + small negative tail."""
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    def _update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # --- Phase 1: Generate diverse initializations ---
    initializations = []
    
    # 1. Constant (baseline)
    initializations.append(np.ones(N, dtype=np.float64))
    
    # 2. Quadratic Residue Sequences (Levenshtein-like)
    # Try primes near 400
    for p in [397, 401, 409, 419]:
        for shift in range(0, 10):
            initializations.append(_quadratic_residue_seq(p, shift))
    
    # 3. Spectral: sin waves
    for freq in [1, 2, 3, 5, 8, 13, 21]:
        t = np.arange(N)
        initializations.append(np.sin(2 * np.pi * freq * t / N))
        initializations.append(np.cos(2 * np.pi * freq * t / N))
    
    # 4. Alternating signs
    initializations.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 5. Block patterns
    for block_size in [10, 20, 50, 100]:
        h = np.ones(N)
        for i in range(0, N, block_size):
            if (i // block_size) % 2 == 1:
                h[i:i+block_size] = -1.0
        initializations.append(h)
    
    # 6. Spike candidates (from inspiration)
    initializations.append(_spike_candidate())
    
    # 7. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        initializations.append(rng.standard_normal(N))
    
    # --- Phase 2: Evaluate all initializations and keep best ---
    for init in initializations:
        val = _evaluate(init)
        _update_best(init, val)
        
        # Also try negation (since abs(max(conv)) is invariant to sign flip of h? 
        # No, conv(-h) = conv(h), so max is same. But sum changes sign.
        # value = |2*N*max(conv)/sum^2|. sum^2 is same. So negation gives same value.
        # So no need to try negation.
    
    # --- Phase 3: Optimize best candidate ---
    # Sort initializations by value to start with best ones? 
    # We already found the best_h. Let's polish it.
    
    # We might also want to try optimizing a few of the top candidates.
    # But to save time, let's just polish the current best.
    
    h0 = best_h.copy()
    s = np.sum(h0)
    if abs(s) < 1e-12:
        return best_h # Fallback to seed if invalid, though seed is valid
    
    # Normalize for optimization? 
    # The objective is scale invariant. 
    # _smooth_objective uses log-sum-exp which is not scale invariant in the same way?
    # log(sum(exp(beta * conv))) is not scale invariant. 
    # If we scale h by c, conv scales by c^2. 
    # logsumexp(beta * c^2 * conv_orig) is not simply related.
    # However, we can just optimize the raw h.
    
    # Let's try a few beta values.
    for beta in [10, 50, 200, 800]:
        if time.time() > deadline - 10:
            break
            
        try:
            result = minimize(
                _smooth_objective,
                h0,
                args=(beta,),
                method='L-BFGS-B',
                jac=_smooth_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e10:
                h_opt = result.x
                val = _evaluate(h_opt)
                _update_best(h_opt, val)
                h0 = h_opt.copy() # Continue from improved point
                
        except Exception:
            pass
    
    # --- Phase 4: Final polish with high beta ---
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        for beta in [2000, 5000]:
            if time.time() > deadline - 2:
                break
            try:
                result = minimize(
                    _smooth_objective,
                    h0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_smooth_grad,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < 1e10:
                    h_opt = result.x
                    val = _evaluate(h_opt)
                    _update_best(h_opt, val)
                    h0 = h_opt.copy()
            except Exception:
                pass
    
    return best_h
