"""QAOA angle design: graph-dependent two-layer parameters via multi-start Nelder-Mead."""
import numpy as np
from scipy.optimize import minimize
import time
import random

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
    """
    m = len(edges)
    if m == 0:
        return np.zeros(2**n)
    
    dim = 2**n
    diag = np.zeros(dim, dtype=np.float64)
    
    for (i, j) in edges:
        states = np.arange(dim)
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        cut_mask = bit_i ^ bit_j  # 1 where edge is cut
        diag += cut_mask
    
    return diag


def _expected_cut(n, edges, gamma1, beta1, gamma2, beta2, cost_diag=None):
    """
    Compute expected cut for 2-layer QAOA with optional precomputed cost diagonal.
    Optimized for small n.
    """
    dim = 2**n
    
    if cost_diag is None:
        cost_diag = _build_cost_diag(n, edges)
    
    # Initial state: |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Helper to apply cost Hamiltonian
    def apply_cost(gamma):
        psi[:] *= np.exp(-1j * gamma * cost_diag)
        
    # Helper to apply mixer Hamiltonian (sum X)
    # This is a tensor product of single-qubit rotations
    def apply_mixer(beta):
        cb = np.cos(beta)
        sb = np.sin(beta)
        # Rotation matrix for single qubit: [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        # We apply this along each axis of the tensor representation of psi
        
        # Reshape to [2, 2, ..., 2]
        psi_tensor = psi.reshape([2] * n)
        
        for q in range(n):
            # Move axis q to the front
            psi_tensor = np.moveaxis(psi_tensor, q, 0)
            
            # Apply rotation: new[0] = cb*old[0] - i*sb*old[1]
            #             new[1] = -i*sb*old[0] + cb*old[1]
            # We can do this with slicing
            new_psi = np.empty_like(psi_tensor)
            new_psi[0] = cb * psi_tensor[0] - 1j * sb * psi_tensor[1]
            new_psi[1] = -1j * sb * psi_tensor[0] + cb * psi_tensor[1]
            
            psi_tensor = new_psi
            # Move axis back
            psi_tensor = np.moveaxis(psi_tensor, 0, q)
            
        psi[:] = psi_tensor.reshape(dim)

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
    """Find best QAOA angles via multi-start Nelder-Mead."""
    opt_cut = _exact_maxcut(n, edges)
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    cost_diag = _build_cost_diag(n, edges)
    
    def ratio(params):
        g1, b1, g2, b2 = params
        exp_cut = _expected_cut(n, edges, g1, b1, g2, b2, cost_diag)
        return exp_cut / opt_cut
        
    def objective(params):
        return -ratio(params)

    best_score = -np.inf
    best_params = None
    
    # Generate starting points
    starts = []
    
    # 1. Coarse grid search to find a good region
    # Use a small grid to save time, e.g., 4x4x4x4 = 256 evaluations
    grid_vals = np.linspace(0.0, np.pi, 4)
    for g1 in grid_vals:
        for b1 in grid_vals:
            for g2 in grid_vals:
                for b2 in grid_vals:
                    if _remaining() < 10.0:
                        break
                    r = ratio([g1, b1, g2, b2])
                    if r > best_score:
                        best_score = r
                        best_params = [g1, b1, g2, b2]
                    starts.append([g1, b1, g2, b2])
    
    # 2. Add some random starts
    random.seed(42) # Fixed seed for reproducibility
    for _ in range(10):
        starts.append([random.uniform(0, np.pi) for _ in range(4)])
        
    # 3. Add heuristic starts
    starts.append([0.25*np.pi, 0.15*np.pi, 0.25*np.pi, 0.15*np.pi])
    starts.append([0.35*np.pi, 0.25*np.pi, 0.35*np.pi, 0.25*np.pi])

    # Run Nelder-Mead from the best few starts
    # Sort starts by their score in the coarse grid (if they were in the grid)
    # For simplicity, just run from the best grid point and a few randoms
    candidates = [best_params]
    # Add a few random ones
    for s in starts[1:6]:
        candidates.append(s)
        
    for start in candidates:
        if _remaining() < 5.0:
            break
            
        try:
            result = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if -result.fun > best_score:
                best_score = -result.fun
                best_params = result.x.copy()
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if _remaining() > 5.0 and best_params is not None:
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
