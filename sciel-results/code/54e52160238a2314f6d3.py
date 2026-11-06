"""QAOA p=2 graph-dependent angle optimizer using BFGS with spectral initialization."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 15.0  # seconds to spare before 150 s wall-clock kill
_START_TIME = time.time()


def _remaining_time():
    return 150.0 - _DEADLINE_MARGIN - (time.time() - _START_TIME)


def _density(n, edges):
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible


def _build_c_diag(n, edges):
    """Precompute cut value per basis state (vectorized)."""
    num_states = 1 << n
    cd = np.zeros(num_states, dtype=float)
    if not edges:
        return cd
    # Vectorize: for each edge (u,v), compute bits and add if different
    states = np.arange(num_states, dtype=np.int64)
    for (u, v) in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        cd += (bits_u != bits_v).astype(float)
    return cd


def _expected_cut(n, edges, gammas, betas, c_diag):
    """Fast expected cut via exact state simulation with vectorized mixer."""
    num_states = 1 << n
    psi = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]

        # Cost layer: diagonal
        psi = psi * np.exp(-1j * gamma * c_diag)

        # Mixer layer: product over i of (cos(beta) I - i sin(beta) X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            mask = 1 << i
            # Split into pairs: even/odd with respect to bit i
            # Use reshape trick: reshape to have bit i as last axis
            # Actually, let's use a simpler vectorized approach
            idx0 = np.where((np.arange(num_states) & mask) == 0)[0]
            idx1 = idx0 | mask
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


def _spectral_init(n, edges, p):
    """Spectral/degree-based initialization for QAOA angles."""
    if n < 2:
        return np.zeros((p, 2))

    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    # Compute degree statistics
    degrees = np.zeros(n, dtype=int)
    for (u, v) in edges:
        degrees[u] += 1
        degrees[v] += 1
    avg_deg = np.mean(degrees) if n > 0 else 0
    max_deg = np.max(degrees) if n > 0 else 0
    deg_ratio = max_deg / (avg_deg + 1e-9) if avg_deg > 0 else 1.0

    # Base parameters from spectral considerations
    # For p=2, good starting points are around:
    # gamma ~ O(1/n), beta ~ O(1/sqrt(n)) for sparse graphs
    # For dense graphs, larger values work
    scale = 1.0 / (1.0 + 0.5 * d)

    gamma1 = (0.3 + 0.4 * d) * scale
    beta1 = (0.25 - 0.05 * d) * scale
    gamma2 = (0.5 + 0.5 * d) * scale
    beta2 = (0.2 - 0.05 * d) * scale

    # Adjust based on degree heterogeneity
    if deg_ratio > 1.5:
        gamma2 *= 1.1
        beta2 *= 0.9

    result = np.zeros((p, 2))
    if p >= 1:
        result[0, 0] = gamma1
        result[0, 1] = beta1
    if p >= 2:
        result[1, 0] = gamma2
        result[1, 1] = beta2
    for k in range(2, p):
        if k % 2 == 0:
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


def _heuristic_init(n, edges, p):
    """Heuristic initialization from graph structure."""
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    gamma1 = 0.25 + 0.55 * d
    beta1 = 0.30 - 0.10 * d
    gamma2 = 0.45 + 0.60 * d
    beta2 = 0.22 - 0.08 * d

    result = np.zeros((p, 2))
    if p >= 1:
        result[0, 0] = gamma1
        result[0, 1] = beta1
    if p >= 2:
        result[1, 0] = gamma2
        result[1, 1] = beta2
    for k in range(2, p):
        if k % 2 == 0:
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


def _optimize_graph(n, edges, p, c_diag, opt_cut):
    """Optimize angles for a single graph using BFGS with multiple starts."""
    if opt_cut == 0 or n == 0 or len(edges) == 0:
        return _heuristic_init(n, edges, p)

    if n > 12:
        return _heuristic_init(n, edges, p)

    def objective(params):
        """Negative ratio for minimization."""
        if len(params) != p * 2:
            return 0.0
        gammas = params[0::2]
        betas = params[1::2]
        try:
            val = _expected_cut(n, edges, gammas, betas, c_diag)
            return -val / opt_cut
        except Exception:
            return 0.0

    # Generate multiple starting points
    starts = []
    starts.append(_spectral_init(n, edges, p).flatten())
    starts.append(_heuristic_init(n, edges, p).flatten())

    # Add some structured variations
    base = _spectral_init(n, edges, p).flatten()
    for scale in (0.8, 1.2, 1.5):
        starts.append(base * scale)
    for offset in (0.1, -0.1, 0.2):
        starts.append(base + offset)

    # Remove duplicates and invalid starts
    unique_starts = []
    seen = set()
    for s in starts:
        s_clipped = np.clip(s, -np.pi, np.pi)
        key = tuple(np.round(s_clipped, 3))
        if key not in seen:
            seen.add(key)
            unique_starts.append(s_clipped)

    best_params = unique_starts[0] if unique_starts else base
    best_score = objective(best_params)

    # BFGS optimization from each start
    for start in unique_starts:
        if _remaining_time() < 5.0:
            break
        try:
            res = minimize(
                objective,
                start,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-4}
            )
            if res.success or res.fun < best_score:
                best_params = np.clip(res.x, -np.pi, np.pi)
                best_score = res.fun
        except Exception:
            continue

    # Convert back to (p, 2) array
    result = np.zeros((p, 2))
    for k in range(p):
        result[k, 0] = best_params[2 * k]
        result[k, 1] = best_params[2 * k + 1]
    return result


def angles(n, edges, p):
    """Entry point. Returns (p, 2) array of (gamma, beta) in radians."""
    try:
        c_diag = _build_c_diag(n, edges)
        opt_cut = int(np.max(c_diag)) if c_diag.size > 0 else 0
    except Exception:
        return _heuristic_init(n, edges, p)

    if opt_cut == 0:
        return _heuristic_init(n, edges, p)

    try:
        result = _optimize_graph(n, edges, p, c_diag, opt_cut)
        return np.clip(result, -np.pi, np.pi)
    except Exception:
        return np.clip(_heuristic_init(n, edges, p), -np.pi, np.pi)
