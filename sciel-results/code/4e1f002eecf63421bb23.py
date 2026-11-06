"""Optimal plus-minus-one polynomial search for 24 coefficients."""
import numpy as np
import time
import random

N = 24
SQRT_N = np.sqrt(N)
DEADLINE = 150.0  # wall-clock seconds; keep safety margin
SAFETY_MARGIN = 15.0  # return at least this much before the kill


def eval_p(coeffs, z):
    """Evaluate P(z) = sum c_k z^k for a given z (or array of z)."""
    # Use Horner-like via polynomial evaluation
    return np.polynomial.polynomial.polyval(z, coeffs)


def get_max_norm(coeffs, n_samples=4096):
    """Estimate max |P(z)| over |z|=1 via dense sampling + golden refinement."""
    # Dense sampling
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    vals = np.abs(eval_p(coeffs, z))
    # Find top peaks and refine each with golden-section search
    # Sort indices by value descending, take top few
    order = np.argsort(vals)[::-1]
    top_k = 12
    best = 0.0
    for idx in order[:top_k]:
        # Golden-section refinement around theta[idx]
        a = theta[idx] - 2.0 * np.pi / n_samples
        b = theta[idx] + 2.0 * np.pi / n_samples
        best = max(best, _golden_refine(coeffs, a, b))
    # Also check a few global top points directly
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


def local_search(coeffs, deadline):
    """Hill-climbing local search flipping coefficients to reduce score."""
    current = list(coeffs)
    current_score = score(current)
    n = len(current)
    rng = random.Random(12345)
    # Keep a list of recently-flipped indices to avoid immediate undo (optional)
    while time.time() < deadline:
        improved = False
        # Try all single flips; pick the best improving one
        best_idx = -1
        best_score = current_score
        # Shuffle order for diversity
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline:
                break
            current[i] = -current[i]
            s = score(current)
            current[i] = -current[i]  # revert
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


def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    best_coeffs = None
    best_score = float('inf')

    # Initial candidates: a few structured starts + random
    candidates = []
    # All +1
    candidates.append([1] * N)
    # Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(N)])
    # A few random starts
    rng = random.Random(7)
    for _ in range(8):
        candidates.append([1 if rng.random() < 0.5 else -1 for _ in range(N)])

    for c in candidates:
        if time.time() > deadline:
            break
        c_list, s = local_search(c, deadline)
        if s < best_score:
            best_score = s
            best_coeffs = c_list

    if best_coeffs is None:
        return [1] * N
    return best_coeffs
