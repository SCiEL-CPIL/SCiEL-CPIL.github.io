"""QAOA angle design: graph-dependent two-layer parameters via grid search + Nelder-Mead refinement."""
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
_DEADLINE = 130.0  # seconds (20s margin before 150s kill)
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


def _expected_cut(n, edges, gamma1, beta1, gamma2, beta2, cost_diag=None):
    """
    Compute expected cut for 2-layer QAOA with optional precomputed cost diagonal.
    """
    dim = 2**n
    
    if cost_diag is None:
        cost_diag = _build_cost_diag(n, edges)
    
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    def apply_cost(gamma):
        psi[:] *= np.exp(-1j * gamma * cost_diag)
    
    def apply_mixer(beta):
        cb = np.cos(beta)
        sb = np.sin(beta)
        psi_arr = psi.reshape([2] * n)
        for q in range(n):
            psi_arr = np.moveaxis(psi_arr, q, 0)
            new_psi = cb * psi_arr[0] - 1j * sb * psi_arr[1]
            new_psi1 = -1j * sb * psi_arr[0] + cb * psi_arr[1]
            psi_arr = np.stack([new_psi, new_psi1], axis=0)
            psi_arr = np.moveaxis(psi_arr, 0, q)
        psi[:] = psi_arr.reshape(dim)
    
    apply_cost(gamma1)
    apply_mixer(beta1)
    apply_cost(gamma2)
    apply_mixer(beta2)
    
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





def _optimize_for_graph(n, edges):
    """Find best QAOA angles via grid search + Nelder-Mead refinement."""
    opt_cut = _exact_maxcut(n, edges)
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    cost_diag = _build_cost_diag(n, edges)
    
    def ratio(params):
        g1, b1, g2, b2 = params
        exp_cut = _expected_cut(n, edges, g1, b1, g2, b2, cost_diag)
        return exp_cut / opt_cut
    
    # Stage 1: Coarse grid search over [0, pi]^4
    n_grid = 6
    grid_vals = np.linspace(0.0, np.pi, n_grid)
    
    best_score = -np.inf
    best_params = None
    
    for g1 in grid_vals:
        for b1 in grid_vals:
            for g2 in grid_vals:
                for b2 in grid_vals:
                    if _remaining() < 5.0:
                        break
                    r = ratio([g1, b1, g2, b2])
                    if r > best_score:
                        best_score = r
                        best_params = [g1, b1, g2, b2]
    
    if _remaining() < 5.0:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    
    # Stage 2: Fine grid around the best coarse-grid point
    box_size = np.pi / 3.0
    n_fine = 8
    center = np.array(best_params)
    center = np.clip(center, box_size/2, np.pi - box_size/2)
    
    f1 = np.linspace(center[0] - box_size/2, center[0] + box_size/2, n_fine)
    f2 = np.linspace(center[1] - box_size/2, center[1] + box_size/2, n_fine)
    f3 = np.linspace(center[2] - box_size/2, center[2] + box_size/2, n_fine)
    f4 = np.linspace(center[3] - box_size/2, center[3] + box_size/2, n_fine)
    
    for g1 in f1:
        for b1 in f2:
            for g2 in f3:
                for b2 in f4:
                    if _remaining() < 5.0:
                        break
                    r = ratio([g1, b1, g2, b2])
                    if r > best_score:
                        best_score = r
                        best_params = [g1, b1, g2, b2]
    
    if _remaining() < 10.0:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    
    # Stage 3: High-precision Nelder-Mead polish
    def objective(params):
        return -ratio(params)
    
    try:
        result = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-12,
                'fatol': 1e-15,
                'adaptive': True,
            }
        )
        if -result.fun > best_score:
            best_score = -result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
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
        return np.array([[0.5, 0.5], [0.4, 0.3]])
    
    try:
        result = _optimize_for_graph(n, edges)
    except Exception:
        result = np.array([[0.35, 0.25], [0.35, 0.25]])
    
    return result
