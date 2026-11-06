"""Optimal plus-minus-one polynomial search for 24 coefficients."""
import numpy as np
import time
import random

N = 24
SQRT_N = np.sqrt(N)
DEADLINE = 150.0  # wall-clock seconds; keep safety margin
SAFETY_MARGIN = 15.0  # return at least this much before the kill

# Precompute grid for fast evaluation
FAST_N_SAMPLES = 512
FAST_THETA = np.linspace(0.0, 2.0 * np.pi, FAST_N_SAMPLES, endpoint=False)
FAST_Z = np.exp(1j * FAST_THETA)

def eval_p(coeffs, z):
    """Evaluate P(z) = sum c_k z^k for a given z (or array of z)."""
    return np.polynomial.polynomial.polyval(z, coeffs)

def get_max_norm(coeffs, n_samples=4096):
    """Estimate max |P(z)| over |z|=1 via dense sampling + golden refinement."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    vals = np.abs(eval_p(coeffs, z))
    order = np.argsort(vals)[::-1]
    top_k = 12
    best = 0.0
    for idx in order[:top_k]:
        a = theta[idx] - 2.0 * np.pi / n_samples
        b = theta[idx] + 2.0 * np.pi / n_samples
        best = max(best, _golden_refine(coeffs, a, b))
    for idx in order[:top_k]:
        v = vals[idx]
        if v > best:
            best = v
    return best

def _golden_refine(coeffs, a, b, iters=60):
    """Golden-section search for max |P(z)| on theta in [a, b]."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = np.abs(eval_p(coeffs, np.exp(1j * c)))
    fd = np.abs(eval_p(coeffs, np.exp(1j * d)))
    for _ in range(iters):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = np.abs(eval_p(coeffs, np.exp(1j * d)))
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = np.abs(eval_p(coeffs, np.exp(1j * c)))
    return max(fc, fd)

def score(coeffs):
    """Score = max |P(z)| / sqrt(N). Lower is better."""
    return get_max_norm(coeffs) / SQRT_N

def fast_score(coeffs):
    """Fast approximate score using precomputed grid."""
    vals = np.abs(eval_p(coeffs, FAST_Z))
    return np.max(vals) / SQRT_N

def local_search(coeffs, deadline):
    """Hill-climbing local search flipping coefficients to reduce score."""
    current = list(coeffs)
    current_score = score(current)
    n = len(current)
    rng = random.Random(12345)
    while time.time() < deadline:
        improved = False
        best_idx = -1
        best_score = current_score
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline:
                break
            current[i] = -current[i]
            s = score(current)
            current[i] = -current[i]
            if s < best_score - 1e-12:
                best_score = s
                best_idx = i
        if best_idx >= 0:
            current[best_idx] = -current[best_idx]
            current_score = best_score
            improved = True
        if not improved:
            break
    return current, current_score

def simulated_annealing(initial_coeffs, deadline, T0=1.0, T_end=0.01, alpha=0.99):
    """Simulated annealing to escape local minima."""
    current = list(initial_coeffs)
    current_score = fast_score(current)
    best = list(current)
    best_score = current_score
    
    T = T0
    n = N
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_end:
        # Pick a random flip
        i = rng.randint(0, n - 1)
        current[i] = -current[i]
        new_score = fast_score(current)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            current[i] = -current[i] # revert
            
        T *= alpha
        
    return best, best_score

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    best_coeffs = None
    best_score = float('inf')

    # Phase 1: Simulated Annealing with fast scoring
    # Allocate ~70% of time to SA
    sa_deadline = start + (DEADLINE - SAFETY_MARGIN) * 0.7
    
    rng = random.Random(7)
    # Run SA from a few random starts
    for _ in range(5):
        if time.time() > sa_deadline:
            break
        init = [1 if rng.random() < 0.5 else -1 for _ in range(N)]
        sa_coeffs, sa_score = simulated_annealing(init, sa_deadline)
        # Refine the SA result with precise score to compare fairly
        precise_score = score(sa_coeffs)
        if precise_score < best_score:
            best_score = precise_score
            best_coeffs = sa_coeffs

    # Phase 2: Local Search Polish with precise scoring
    # Use remaining time
    if best_coeffs is not None:
        final_coeffs, final_score = local_search(best_coeffs, deadline)
        if final_score < best_score:
            best_score = final_score
            best_coeffs = final_coeffs
    else:
        # Fallback
        best_coeffs = [1] * N
        best_score = score(best_coeffs)

    return best_coeffs
