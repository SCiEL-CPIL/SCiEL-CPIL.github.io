"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_cost_op(n, edges):
    """Build the MaxCut cost operator as a sparse matrix."""
    dim = 1 << n
    # C = (1/2) * sum_{(i,j) in E} (I - Z_i Z_j)
    # We'll build it as a diagonal matrix since Z_i Z_j is diagonal
    diag = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                diag[state] += 0.5  # contribution to cut
    return np.diag(diag)


def _apply_x_layer(state, beta):
    """Apply exp(-i beta sum X) to state vector."""
    # sum X = X_1 + X_2 + ... + X_n
    # exp(-i beta sum X) = (cos(beta) I - i sin(beta) sum X)
    # But sum X is not unitary in the same simple way...
    # Actually, sum X is not a commuting set, so we need to be careful.
    # 
    # Wait: the problem says "exp(-i beta sum X)" where sum X means sum of X_i operators.
    # These X_i operators commute with each other (X_i X_j = X_j X_i for i≠j),
    # so exp(-i beta sum_i X_i) = prod_i exp(-i beta X_i)
    # And exp(-i beta X) = cos(beta) I - i sin(beta) X
    #
    # So the full operator is prod_i (cos(beta) I - i sin(beta) X_i)
    # We can apply this by iterating over each qubit.
    
    dim = len(state)
    n = int(np.log2(dim))
    c = np.cos(beta)
    s = np.sin(beta)
    
    for i in range(n):
        # Apply (c I - i s X_i) to state
        # X_i flips bit i
        new_state = np.zeros_like(state)
        for state_idx in range(dim):
            flipped = state_idx ^ (1 << i)
            new_state[state_idx] += c * state[state_idx]
            new_state[flipped] += -1j * s * state[state_idx]
        state = new_state
    return state


def _apply_cost_layer(state, gamma, edges, n):
    """Apply exp(-i gamma C) to state vector, where C is MaxCut cost operator."""
    dim = len(state)
    # C is diagonal: C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    # Z_i Z_j has eigenvalue +1 if bits i,j same, -1 if different
    # So (I - Z_i Z_j)/2 has eigenvalue 0 if same, 1 if different
    # C = sum over edges of [1 if cut, 0 if not]
    
    # Compute eigenvalues of C for each computational basis state
    c_vals = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        for state_idx in range(dim):
            bi = (state_idx >> i) & 1
            bj = (state_idx >> j) & 1
            if bi != bj:
                c_vals[state_idx] += 1.0
    
    # exp(-i gamma C) is diagonal with entries exp(-i gamma * c_vals[state])
    phase = np.exp(-1j * gamma * c_vals)
    return state * phase


def _expected_cut(state, edges, n):
    """Compute expected cut value <state|C|state>."""
    dim = len(state)
    probs = np.abs(state) ** 2
    c_vals = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        for state_idx in range(dim):
            bi = (state_idx >> i) & 1
            bj = (state_idx >> j) & 1
            if bi != bj:
                c_vals[state_idx] += 1.0
    return float(np.real(np.sum(probs * c_vals)))


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    for state in range(dim):
        cut = 0
        for (i, j) in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut += 1
        if cut > best:
            best = cut
    return best


def _forward_state(n, edges, angles):
    """Simulate the QAOA circuit forward to get final state."""
    dim = 1 << n
    # Start in |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        state = _apply_cost_layer(state, gamma, edges, n)
        state = _apply_x_layer(state, beta)
    
    return state


def _objective(params, n, edges, p):
    """Objective: minimize negative expected cut."""
    angles = params.reshape(p, 2)
    state = _forward_state(n, edges, angles)
    return -_expected_cut(state, edges, n)


def _gradient(params, n, edges, p, eps=1e-6):
    """Compute gradient via finite differences."""
    grad = np.zeros_like(params)
    f0 = _objective(params, n, edges, p)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, n, edges, p)
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus, n, edges, p)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _optimize_graph(n, edges, p, max_iter=50):
    """Optimize QAOA angles for a given graph."""
    # Initial guess: small angles
    x0 = np.array([0.1, 0.1] * p, dtype=np.float64)
    
    bounds = [(-np.pi, np.pi)] * (2 * p)
    
    result = minimize(
        _objective,
        x0,
        args=(n, edges, p),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    return result.x.reshape(p, 2)


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Check time budget
    if _time_left() < 5.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        # Optimize angles for this specific graph
        opt_angles = _optimize_graph(n, edges, p, max_iter=30)
        
        # Validate: ensure angles are reasonable
        if not np.all(np.isfinite(opt_angles)):
            return np.tile([0.35, 0.25], (p, 1))
        
        return opt_angles
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
