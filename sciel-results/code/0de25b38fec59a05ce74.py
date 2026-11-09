"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Deadline guard
_DEADLINE = 140.0  # seconds wall-clock limit (10s safety margin under 150s kill)

# Cache for exact optima and expected cuts
_optima_cache = {}
_angles_cache = {}


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _exact_maxcut(n, adj):
    """Compute exact MaxCut via brute force for small n."""
    key = (n, tuple(tuple(row) for row in adj))
    if key in _optima_cache:
        return _optima_cache[key]
    
    best = 0
    # For n up to ~20, brute force is feasible
    # Use bit manipulation for speed
    for mask in range(1 << n):
        cut = 0
        for i in range(n):
            if mask & (1 << i):
                # Count neighbors NOT in same partition
                for j in range(i + 1, n):
                    if adj[i, j] and not (mask & (1 << j)):
                        cut += 1
            else:
                for j in range(i + 1, n):
                    if adj[i, j] and (mask & (1 << j)):
                        cut += 1
        if cut > best:
            best = cut
    _optima_cache[key] = float(best)
    return float(best)


def _expected_cut(n, adj, gammas, betas):
    """
    Compute expected cut size for QAOA with given parameters.
    Uses the exact quantum expectation formula for MaxCut.
    
    For a single edge (i,j), the probability that i and j are in different
    partitions after applying the QAOA circuit is:
    
    P_diff(i,j) = 0.5 - 0.5 * <Z_i Z_j>
    
    where <Z_i Z_j> can be computed from the unitary evolution.
    
    For p=2, we can compute this analytically using the structure of the circuit.
    """
    # For small n, we can compute the full state vector
    # But for n=10, 2^10 = 1024 states, which is manageable
    # For n=10 with p=2, the circuit has 4 layers of gates
    
    # Actually, let's use the analytical formula for QAOA on MaxCut
    # The expectation value <Z_i Z_j> for an edge can be computed via
    # the overlap of certain operators.
    
    # For efficiency with small n, let's just simulate the state vector
    # if n is small enough, otherwise use approximation
    
    if n <= 12:
        return _expected_cut_statevector(n, adj, gammas, betas)
    else:
        # Fallback: use a simpler approximation or local search
        return _expected_cut_approx(n, adj, gammas, betas)


def _expected_cut_statevector(n, adj, gammas, betas):
    """Compute expected cut using full state vector simulation for small n."""
    dim = 1 << n
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i * gamma * C) where C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
        # = sum_{(i,j) in E} (1/2) - (1/2) Z_i Z_j
        # The constant term just adds a global phase, so we can ignore it
        # C_effective = -1/2 * sum_{(i,j) in E} Z_i Z_j
        # So exp(-i * gamma * C) = exp(-i * gamma * (-1/2) * sum Z_i Z_j)
        # = exp(i * gamma/2 * sum Z_i Z_j)
        
        # For each edge, apply exp(i * gamma/2 * Z_i Z_j)
        # Z_i Z_j is diagonal with eigenvalues +1 (same bit) or -1 (different bits)
        # So exp(i * gamma/2 * Z_i Z_j) is diagonal with entries:
        #   exp(i * gamma/2) if bits same
        #   exp(-i * gamma/2) if bits different
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    # Apply diagonal gate exp(i * gamma/2 * Z_i Z_j)
                    for mask in range(dim):
                        bit_i = (mask >> i) & 1
                        bit_j = (mask >> j) & 1
                        if bit_i == bit_j:
                            state[mask] *= np.exp(1j * gamma / 2)
                        else:
                            state[mask] *= np.exp(-1j * gamma / 2)
        
        # Apply exp(-i * beta * sum X)
        # This is a product of single-qubit rotations: exp(-i * beta * X_k) for each k
        # X = [[0,1],[1,0]], exp(-i*beta*X) = [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        # Apply to each qubit
        
        # For each qubit k, apply the rotation
        # We need to apply this to the state vector
        for k in range(n):
            # Apply single-qubit gate to qubit k
            c = np.cos(beta)
            s = np.sin(beta)
            # Gate: [[c, -i*s], [-i*s, c]]
            # For each pair of basis states differing only in bit k:
            # new_state[|0...>] = c * old[|0...>] - 1j*s * old[|1...>]
            # new_state[|1...>] = -1j*s * old[|0...>] + c * old[|1...>]
            
            # Iterate over all masks where bit k is 0
            for mask in range(dim):
                if not (mask & (1 << k)):
                    mask_flip = mask | (1 << k)
                    v0 = state[mask]
                    v1 = state[mask_flip]
                    state[mask] = c * v0 - 1j * s * v1
                    state[mask_flip] = -1j * s * v0 + c * v1
    
    # Compute expected cut: sum over edges of P(i,j differ)
    # P(i,j differ) = 0.5 - 0.5 * <Z_i Z_j>
    # <Z_i Z_j> = sum_mask |state[mask]|^2 * Z_i(mask) * Z_j(mask)
    # where Z_k(mask) = +1 if bit k is 0, -1 if bit k is 1
    
    prob = np.abs(state) ** 2
    expected_cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                # Compute <Z_i Z_j>
                zz = 0.0
                for mask in range(dim):
                    zi = 1.0 if (mask >> i) & 1 == 0 else -1.0
                    zj = 1.0 if (mask >> j) & 1 == 0 else -1.0
                    zz += prob[mask] * zi * zj
                expected_cut += 0.5 - 0.5 * zz
    
    return expected_cut


def _expected_cut_approx(n, adj, gammas, betas):
    """Approximate expected cut for larger n using local structure."""
    # For now, fall back to a simple heuristic
    # This is a placeholder - for n <= 12 we use exact simulation
    # For larger n, we'd need a better approximation
    # Since our graphs are n=6,8,10, this shouldn't be called
    return _expected_cut_statevector(n, adj, gammas, betas)


def _objective(params, n, adj, opt):
    """Compute the objective: expected_cut / exact_optimum."""
    gammas = params[0::2]
    betas = params[1::2]
    try:
        exp_cut = _expected_cut(n, adj, gammas, betas)
        if opt > 0:
            return exp_cut / opt
        return 0.0
    except Exception:
        return 0.0


def _optimize_angles(n, adj, opt, p, time_budget):
    """
    Optimize QAOA parameters using Nelder-Mead or similar.
    Returns (gammas, betas) arrays of length p.
    """
    from scipy.optimize import minimize
    
    def neg_obj(params):
        return -_objective(params, n, adj, opt)
    
    # Initial guess: standard QAOA starting point
    x0 = np.zeros(2 * p)
    for i in range(p):
        x0[2*i] = 0.5 * np.pi / (p + 1)  # gamma
        x0[2*i+1] = 0.5 * np.pi / (p + 1)  # beta
    
    best_result = None
    best_val = neg_obj(x0)
    
    # Try multiple restarts
    start_time = time.time()
    restarts = 0
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Random restart
            x_init = np.random.uniform(0, np.pi, size=2*p)
        
        try:
            result = minimize(
                neg_obj, x_init, method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_result = result.x.copy()
        except Exception:
            pass
        
        restarts += 1
        if time.time() - start_time > time_budget:
            break
    
    if best_result is None:
        best_result = x0
    
    gammas = best_result[0::2]
    betas = best_result[1::2]
    return gammas, betas


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of (u, v) tuples
        p: number of QAOA layers (should be 2)
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Build adjacency
    adj = _build_adjacency(n, edges)
    
    # Compute exact optimum
    opt = _exact_maxcut(n, adj)
    
    # Check if we've already computed this
    key = (n, tuple(sorted(edges)), p)
    if key in _angles_cache:
        return _angles_cache[key]
    
    # Allocate time budget for optimization
    time_budget = min(120.0, max(5.0, _DEADLINE - time.time() - 5.0))
    
    # Optimize parameters
    gammas, betas = _optimize_angles(n, adj, opt, p, time_budget)
    
    # Build result array
    result = np.zeros((p, 2))
    for i in range(p):
        result[i, 0] = gammas[i]
        result[i, 1] = betas[i]
    
    # Cache the result
    _angles_cache[key] = result
    
    return result


# Pre-compute common cases to speed up evaluation
def _precompute():
    """Pre-compute angles for common graph sizes."""
    pass

# EVOLVE-BLOCK-END
