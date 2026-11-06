"""QAOA p=2 angle design: exact expectation + multi-start Nelder-Mead with graph-aware bounds."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 140s out of 150s limit, leaving 10s safety margin


def _remaining():
    return _DEADLINE - time.time()


def _build_graph(n, edges):
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        if u == v:
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _precompute_graph_data(n, edges):
    """Precompute all data needed for repeated expected-cut evaluations."""
    m = len(edges)
    c = 1 << n
    bits = np.arange(c, dtype=np.int64)
    
    # For each basis state z, compute z_i = (-1)^{bit_i} for each qubit i
    # z_val[i, z] = (-1)^{bit_i of z}
    z_vals = np.ones((n, c), dtype=np.float64)
    for i in range(n):
        z_vals[i] = np.where((bits >> i) & 1, -1.0, 1.0)
    
    # zz[k, z] = z_u * z_v for edge k = (u, v)
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    zz = z_vals[u] * z_vals[v]  # shape (m, c)
    
    # edge_sum[z] = sum over edges of z_u * z_v for state z
    edge_sum = np.sum(zz, axis=0)  # shape (c,)
    
    # popcount[z] = number of 1-bits in z
    popcount = np.zeros(c, dtype=np.int32)
    for i in range(n):
        popcount += (bits >> i) & 1
    
    return {
        'n': n,
        'm': m,
        'c': c,
        'zz': zz,
        'edge_sum': edge_sum,
        'popcount': popcount,
    }


def _apply_sx_rotation(amp, b, n, c, popcount):
    """Apply exp(-i*b*Sx) to amplitude vector.
    
    exp(-i*b*Sx) = prod_i exp(-i*b*X_i)
    For each qubit i: exp(-i*b*X_i) = cos(b)*I - i*sin(b)*X_i
    
    This is a tensor product operation. We apply it qubit by qubit.
    amp has shape (c,) where c = 2^n.
    """
    if abs(b) < 1e-15:
        return amp.copy()
    
    cb = np.cos(b)
    sb = np.sin(b)
    
    # For each qubit i, apply: new_amp = cb*amp - i*sb*flip_i(amp)
    # where flip_i flips bit i of the index
    result = amp.astype(np.complex128).copy()
    for i in range(n):
        # Create index arrays for flipping bit i
        # indices where bit i is 0
        mask = (np.arange(c) >> i) & 1
        idx0 = np.where(mask == 0)[0]
        idx1 = np.where(mask == 1)[0]
        # idx0 and idx1 are paired: idx0[k] and idx0[k] + (1<<i)
        # Flip: new[idx0] = cb*amp[idx0] - i*sb*amp[idx1]
        #        new[idx1] = cb*amp[idx1] - i*sb*amp[idx0]
        old0 = result[idx0].copy()
        old1 = result[idx1].copy()
        result[idx0] = cb * old0 - 1j * sb * old1
        result[idx1] = cb * old1 - 1j * sb * old0
    return result


def _expected_cut(n, edges, g1, b1, g2, b2, data):
    """Compute expected MaxCut for p=2 QAOA using exact state vector simulation."""
    m = data['m']
    c = data['c']
    zz = data['zz']
    edge_sum = data['edge_sum']
    popcount = data['popcount']
    
    if m == 0:
        return 0.0
    
    # Start: |+>^n, uniform amplitudes
    amp = np.ones(c, dtype=np.complex128) / np.sqrt(c)
    
    # Step 1: Apply exp(-i*b1*Sx)
    amp = _apply_sx_rotation(amp, b1, n, c, popcount)
    
    # Step 2: Apply exp(-i*g1*C) where C = sum_edges (1 - Z_u Z_v)/2
    # C|z> = (m - sum_edges z_u z_v)/2 |z> = (m - edge_sum[z])/2 |z>
    if abs(g1) > 1e-15:
        phase1 = np.exp(-1j * g1 * (m - edge_sum) / 2.0)
        amp *= phase1
    
    # Step 3: Apply exp(-i*b2*Sx)
    amp = _apply_sx_rotation(amp, b2, n, c, popcount)
    
    # Step 4: Apply exp(-i*g2*C)
    if abs(g2) > 1e-15:
        phase2 = np.exp(-1j * g2 * (m - edge_sum) / 2.0)
        amp *= phase2
    
    # Probability distribution
    prob = np.abs(amp) ** 2
    prob = prob / np.sum(prob)  # normalize for numerical stability
    
    # Expected cut = sum_edges (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> for edge k = sum_z prob[z] * zz[k,z]
    corr = zz @ prob  # shape (m,)
    expected_cut = 0.5 * np.sum(1.0 - corr)
    
    return float(expected_cut)


def _exhaustive_maxcut(n, edges):
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, data):
    """Negative expected cut for minimization."""
    return -_expected_cut(data['n'], None, params[0], params[1], params[2], params[3], data)


def angles(n, edges, p):
    adj = _build_graph(n, edges)
    m = len(edges)
    
    # Precompute graph data
    data = _precompute_graph_data(n, edges)
    
    # Compute graph properties for scaling
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        w = np.linalg.eigvalsh(L)
        lmax = float(w[-1]) if len(w) else 0.0
    except Exception:
        lmax = 0.0

    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    # Graph-aware parameter ranges - use broader ranges
    if lmax > 1e-8:
        gamma_max = min(2.5, 0.8 * lmax)
    else:
        gamma_max = 0.8
    beta_max = 1.5

    # Multi-start strategy with diverse initial points
    initial_points = []
    
    # Strategy 1: Coarse grid
    n_grid = 5
    gamma_vals = np.linspace(0.05, gamma_max, n_grid)
    beta_vals = np.linspace(0.05, beta_max, n_grid)
    
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    initial_points.append([g1, b1, g2, b2])
    
    # Strategy 2: Random sampling
    rng = np.random.default_rng(42)
    for _ in range(80):
        g1 = rng.uniform(0.05, gamma_max)
        b1 = rng.uniform(0.05, beta_max)
        g2 = rng.uniform(0.05, gamma_max)
        b2 = rng.uniform(0.05, beta_max)
        initial_points.append([g1, b1, g2, b2])
    
    # Strategy 3: Heuristic starting points
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
        g1 = gamma_max * ratio
        g2 = gamma_max * (1 - ratio)
        b1 = beta_max * 0.4
        b2 = beta_max * 0.6
        initial_points.append([g1, b1, g2, b2])
    
    # Strategy 4: Symmetric and near-symmetric points
    for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            initial_points.append([g, b, g, b])
            initial_points.append([g, b, g*0.8, b*0.8])
            initial_points.append([g*1.2, b*0.7, g*0.8, b*1.3])
    
    # Evaluate all initial points and keep top candidates
    scored_points = []
    for point in initial_points:
        if _remaining() < 15.0:
            break
        try:
            val = -_objective(point, data)
            scored_points.append((val, point))
        except Exception:
            continue
    
    # Sort by value and keep top candidates for refinement
    scored_points.sort(key=lambda x: x[0], reverse=True)
    top_k = scored_points[:min(12, len(scored_points))]
    
    best_val = top_k[0][0] if top_k else -np.inf
    best_x = np.array(top_k[0][1]) if top_k else np.array([0.2, 0.2, 0.2, 0.2])
    
    # Refine top candidates with Nelder-Mead
    for val, point in top_k:
        if _remaining() < 20.0:
            break
        try:
            res = minimize(
                _objective, point, args=(data,),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision refinement with L-BFGS-B
    if _remaining() > 25.0:
        try:
            res = minimize(
                _objective, best_x, args=(data,),
                method='L-BFGS-B',
                options={'maxiter': 800, 'gtol': 1e-12, 'ftol': 1e-14},
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional refinement passes from perturbed best points
    if _remaining() > 25.0:
        rng2 = np.random.default_rng(123)
        for _ in range(10):
            if _remaining() < 15.0:
                break
            scale = 0.02 + 0.01 * (10 - _)
            perturbed = best_x + rng2.normal(0, scale, size=4)
            perturbed = np.clip(perturbed, 0.01, max(gamma_max, beta_max))
            try:
                res = minimize(
                    _objective, perturbed, args=(data,),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True},
                )
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final polish with very tight tolerance
    if _remaining() > 15.0:
        try:
            res = minimize(
                _objective, best_x, args=(data,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    out = np.zeros((p, 2), dtype=float)
    out[0, 0] = float(best_x[0])
    out[0, 1] = float(best_x[1])
    if p > 1:
        out[1, 0] = float(best_x[2])
        out[1, 1] = float(best_x[3])
    return out
# EVOLVE-BLOCK-END
