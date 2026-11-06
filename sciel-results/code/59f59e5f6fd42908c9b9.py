"""QAOA angle design: graph-dependent two-layer parameters via multi-start Nelder-Mead."""
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
_DEADLINE = 140.0  # seconds (10s margin before 150s kill)
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _build_cost_diag(n, edges):
    """
    Build the diagonal of the cost Hamiltonian C for unweighted MaxCut.
    C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
    So <Z|C|Z> = sum_{(i,j) in E} (1 - z_i z_j)/2 where z_i = +/-1.
    We'll work with the bit representation where z_i = 1 if bit=0, -1 if bit=1.
    Actually, let's use: for a basis state |b> with bits b_i in {0,1},
    Z_i |b> = (-1)^{b_i} |b>, so z_i = (-1)^{b_i}.
    C |b> = sum_{(i,j)} (1 - z_i z_j)/2 |b>
    = sum_{(i,j)} (1 - (-1)^{b_i + b_j})/2 |b>
    
    For the QAOA evolution, we need to apply exp(-i gamma C) to the state.
    Since C is diagonal in the computational basis, this is just a phase rotation
    on each basis state.
    
    For n up to 10, we can store the full 2^n vector.
    """
    m = len(edges)
    if m == 0:
        return np.zeros(2**n)
    
    # Build adjacency info for fast diagonal computation
    # For each edge (i,j), contribution is (1 - (-1)^(b_i + b_j))/2
    # = 0 if b_i == b_j, 1 if b_i != b_j
    # So C_diag[b] = number of edges with endpoints in different partitions = cut size of b
    
    # Vectorized approach: for each edge, compute which basis states have that edge cut
    dim = 2**n
    diag = np.zeros(dim, dtype=np.float64)
    
    for (i, j) in edges:
        # Bit i is set in states where (state >> i) & 1 == 1
        # Bit j is set in states where (state >> j) & 1 == 1
        # Edge is cut when bits differ: XOR
        # Create array of states
        states = np.arange(dim)
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        cut_mask = bit_i ^ bit_j  # 1 where edge is cut
        diag += cut_mask
    
    return diag


def _expected_cut(n, edges, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA.
    Layer 1: exp(-i gamma1 * C) then exp(-i beta1 * sum X)
    Layer 2: exp(-i gamma2 * C) then exp(-i beta2 * sum X)
    
    Start state: |+>^n = (1/sqrt(2^n)) * sum_b |b>
    
    For n <= 10, we can use full state vector (2^n amplitudes).
    """
    dim = 2**n
    
    # Initial state: uniform superposition
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cost diagonal
    cost_diag = _build_cost_diag(n, edges)
    
    # Precompute bit arrays for mixer
    states = np.arange(dim)
    
    def apply_cost(gamma):
        """Apply exp(-i gamma * C) - diagonal phase rotation"""
        psi[:] *= np.exp(-1j * gamma * cost_diag)
    
    def apply_mixer(beta):
        """Apply exp(-i beta * sum_i X_i) - this is a tensor product of single-qubit rotations"""
        # exp(-i beta X) on a single qubit:
        # [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        # For the full state, this is a tensor product.
        # We can apply it qubit by qubit.
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        # Reshape psi to (2, 2, ..., 2) with n axes
        psi_shape = [2] * n
        psi_arr = psi.reshape(psi_shape)
        
        for q in range(n):
            # Move axis q to front, apply 2x2 rotation, move back
            psi_arr = np.moveaxis(psi_arr, q, 0)
            # Apply [[cb, -1j*sb], [-1j*sb, cb]]
            new_psi = cb * psi_arr[0] - 1j * sb * psi_arr[1]
            new_psi1 = -1j * sb * psi_arr[0] + cb * psi_arr[1]
            # Reconstruct
            new_shape = [2] + list(psi_arr.shape[1:])
            psi_arr = np.stack([new_psi, new_psi1], axis=0)
            psi_arr = np.moveaxis(psi_arr, 0, q)
        
        psi[:] = psi_arr.reshape(dim)
    
    # Apply layers
    apply_cost(gamma1)
    apply_mixer(beta1)
    apply_cost(gamma2)
    apply_mixer(beta2)
    
    # Expected cut = <psi|C|psi> = sum_b |psi_b|^2 * cost_diag[b]
    probs = np.abs(psi)**2
    return np.dot(probs, cost_diag)


def _exact_maxcut(n, edges):
    """Exact MaxCut via brute force for small n."""
    if not edges:
        return 0.0
    
    best = 0
    for mask in range(2**n):
        cut = 0
        for (i, j) in edges:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut += 1
        if cut > best:
            best = cut
    return float(best)


def _objective(params, n, edges, opt_cut):
    """Negative ratio for minimization."""
    gamma1, beta1, gamma2, beta2 = params
    # Clip to reasonable ranges
    gamma1 = np.clip(gamma1, -np.pi, np.pi)
    beta1 = np.clip(beta1, -np.pi, np.pi)
    gamma2 = np.clip(gamma2, -np.pi, np.pi)
    beta2 = np.clip(beta2, -np.pi, np.pi)
    
    exp_cut = _expected_cut(n, edges, gamma1, beta1, gamma2, beta2)
    ratio = exp_cut / opt_cut if opt_cut > 0 else 0.0
    return -ratio


def _optimize_for_graph(n, edges):
    """Find best QAOA angles for a specific graph."""
    opt_cut = _exact_maxcut(n, edges)
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Multi-start Nelder-Mead
    best_params = None
    best_score = -np.inf
    
    # Initial guesses: use common good starting points
    starts = [
        [0.5, 0.5, 0.5, 0.5],
        [0.35, 0.25, 0.35, 0.25],
        [0.8, 0.4, 0.6, 0.3],
        [1.0, 0.5, 0.8, 0.4],
        [0.2, 0.3, 0.4, 0.2],
        [0.6, 0.6, 0.4, 0.4],
        [1.2, 0.3, 0.9, 0.2],
        [0.4, 0.7, 0.5, 0.6],
    ]
    
    # Add some random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        starts.append(rng.uniform(0, np.pi, size=4).tolist())
    
    for x0 in starts:
        if _remaining() < 5.0:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, edges, opt_cut),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish if we have time
    if best_params is not None and _remaining() > 10.0:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, edges, opt_cut),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True,
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of (i, j) edges
        p: number of layers (should be 2)
    
    Returns:
        np.array of shape (p, 2), rows (gamma, beta) in radians.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if n > 12:
        # Too large for full state vector; use heuristic
        return np.array([[0.5, 0.5], [0.4, 0.3]])
    
    try:
        result = _optimize_for_graph(n, edges)
    except Exception:
        result = np.array([[0.35, 0.25], [0.35, 0.25]])
    
    return result
