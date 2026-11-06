"""QAOA p=2 graph-dependent angle optimizer: vectorized simulation + multi-start Nelder-Mead."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 12.0


def _density(n, edges):
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible


def _initial_angles(n, edges, p):
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    # Refined heuristics based on QAOA angle trends for p=2
    gamma1 = 0.22 + 0.58 * d
    beta1 = 0.32 - 0.12 * d
    gamma2 = 0.48 + 0.52 * d
    beta2 = 0.24 - 0.09 * d

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


def _precompute(n, edges):
    """Precompute c_diag, mixer index pairs, and edge bitmasks for fast evaluation."""
    num_states = 1 << n

    # c_diag: cut value per basis state
    c_diag = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        bits = [(state >> i) & 1 for i in range(n)]
        cut = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut += 1
        c_diag[state] = cut

    # Mixer index pairs
    mixer_pairs = []
    for i in range(n):
        mask = 1 << i
        idx0 = np.array([s0 for s0 in range(num_states) if (s0 & mask) == 0], dtype=np.int64)
        idx1 = idx0 | mask
        mixer_pairs.append((idx0, idx1))

    # Edge bitmasks for vectorized cost phase
    # For each edge (u,v), we need to know which states have bit_u == bit_v
    # We precompute a phase array: for each state, the total phase from all edges
    # Actually, we'll compute the cost phase per state directly
    # cost_phase[state] = -gamma * sum_{(u,v) in E} (1 if bit_u != bit_v else -1) / 2
    # But gamma varies, so we precompute the "cut indicator" per state
    # c_diag already gives us the cut count; the phase is -gamma * c_diag
    # Wait: C = sum_{(u,v)} (1 - Z_u Z_v)/2
    # Z_u Z_v = 1 if same, -1 if different
    # So (1 - Z_u Z_v)/2 = 0 if same, 1 if different = cut indicator
    # So exp(-i gamma C) applies phase -i*gamma*c_diag[state]
    # That's exactly what we have: psi *= exp(-i*gamma*c_diag)

    return c_diag, mixer_pairs


def _expected_cut_fast(n, edges, gammas, betas, c_diag, mixer_pairs):
    """Fast expected cut via vectorized state simulation."""
    num_states = 1 << n
    psi = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]

        # Cost layer: diagonal phase
        psi = psi * np.exp(-1j * gamma * c_diag)

        # Mixer layer
        c = np.cos(beta)
        s = np.sin(beta)
        for idx0, idx1 in mixer_pairs:
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


def _nelder_mead_refine(params0, objective, deadline, maxfev=200, xatol=1e-7, fatol=1e-9):
    """Nelder-Mead refinement from a starting point."""
    try:
        res = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': maxfev,
                'xatol': xatol,
                'fatol': fatol,
                'adaptive': True,
            }
        )
        if res.success or res.fun < objective(params0):
            return np.clip(res.x, -np.pi, np.pi)
    except Exception:
        pass
    return np.clip(params0, -np.pi, np.pi)


def angles(n, edges, p):
    """Entry point. Returns (p, 2) array of (gamma, beta) in radians."""
    t_start = time.time()
    deadline = t_start + 150.0 - _DEADLINE_MARGIN

    try:
        init = _initial_angles(n, edges, p)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))

    if n == 0 or len(edges) == 0:
        return init

    try:
        c_diag, mixer_pairs = _precompute(n, edges)
        opt_cut = int(np.max(c_diag)) if c_diag.size else 0
    except Exception:
        return init

    if opt_cut == 0:
        return init

    if n > 12:
        return init

    def objective(params):
        a = params.reshape(p, 2)
        a = np.clip(a, -np.pi, np.pi)
        try:
            v = _expected_cut_fast(n, edges, a[:, 0], a[:, 1], c_diag, mixer_pairs)
            return -v / opt_cut
        except Exception:
            return 1e10

    # Generate diverse starting points
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    starts = [init.flatten().copy()]

    # Add structured grid + random starts for better coverage
    g1_c = 0.22 + 0.58 * d
    b1_c = 0.32 - 0.12 * d
    g2_c = 0.48 + 0.52 * d
    b2_c = 0.24 - 0.09 * d

    # Structured grid: 3x3 around heuristic center
    for dg1 in [-0.3, 0.0, 0.3]:
        for db1 in [-0.15, 0.0, 0.15]:
            for dg2 in [-0.3, 0.0, 0.3]:
                for db2 in [-0.15, 0.0, 0.15]:
                    g1 = g1_c + dg1
                    b1 = b1_c + db1
                    g2 = g2_c + dg2
                    b2 = b2_c + db2
                    s = np.array([g1, b1, g2, b2])
                    if p >= 1:
                        s = s[:2]
                    else:
                        s = s[:0]
                    if p >= 2:
                        s = np.concatenate([s, [g2, b2]])
                    for k in range(2, p):
                        if k % 2 == 0:
                            s = np.concatenate([s, [g2, b2]])
                        else:
                            s = np.concatenate([s, [g1, b1]])
                    starts.append(np.clip(s, 0.0, np.pi))

    # Random starts for additional diversity
    rng = np.random.RandomState(713)
    n_random_starts = 12
    for _ in range(n_random_starts):
        g1 = rng.uniform(0.05, 1.5)
        b1 = rng.uniform(0.05, 1.2)
        g2 = rng.uniform(0.1, 2.0)
        b2 = rng.uniform(0.05, 1.0)
        s = np.array([g1, b1, g2, b2])
        if p >= 1:
            s = s[:2]
        else:
            s = s[:0]
        if p >= 2:
            s = np.concatenate([s, [g2, b2]])
        for k in range(2, p):
            if k % 2 == 0:
                s = np.concatenate([s, [g2, b2]])
            else:
                s = np.concatenate([s, [g1, b1]])
        starts.append(np.clip(s, 0.0, np.pi))

    # Evaluate all starts and keep the best few
    scored_starts = []
    for s in starts:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        val = objective(s)
        scored_starts.append((val, s))

    scored_starts.sort(key=lambda x: x[0])  # ascending (most negative = best)

    # Refine top candidates with more budget
    best_params = scored_starts[0][1] if scored_starts else init.flatten()
    best_val = scored_starts[0][0] if scored_starts else objective(best_params)

    n_refine = min(5, len(scored_starts))
    for i in range(n_refine):
        if time.time() > deadline - _DEADLINE_MARGIN - 8.0:
            break
        val, s = scored_starts[i]
        refined = _nelder_mead_refine(s, objective, deadline, maxfev=180, xatol=1e-8, fatol=1e-10)
        r_val = objective(refined)
        if r_val < best_val:
            best_val = r_val
            best_params = refined

    # Final high-precision polish on the best
    if time.time() < deadline - _DEADLINE_MARGIN - 5.0:
        polished = _nelder_mead_refine(best_params, objective, deadline, maxfev=250, xatol=1e-10, fatol=1e-12)
        p_val = objective(polished)
        if p_val < best_val:
            best_params = polished

    return np.clip(best_params.reshape(p, 2), 0.0, np.pi)
