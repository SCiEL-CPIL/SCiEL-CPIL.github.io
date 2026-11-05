"""
QAOA angle design for MaxCut with p=2 layers.
Graph-dependent optimization using precomputed eigen-decompositions and L-BFGS-B.
"""
import numpy as np
from scipy.optimize import minimize
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0


def _remaining_time():
    return _TIME_LIMIT - _SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_cost_eigs(n, edges):
    """Build cost Hamiltonian C = sum_{(u,v)} Z_u Z_v and return its eigenvalues.
    Since C is diagonal in the computational basis, eigenvalues are just the diagonal."""
    dim = 1 << n
    diag = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            diag[state] += 1.0 if bu == bv else -1.0
    return diag


def _build_mixer_eig(n):
    """Build sum X_i Hamiltonian and return eigenvalues and eigenvectors."""
    dim = 1 << n
    # sum X_i has eigenvalues -n, -(n-2), ..., n-2, n each with multiplicity C(n, k)
    # but for n up to 10, just build the matrix and diagonalize
    M = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            new_state = state ^ (1 << i)
            M[state, new_state] += 1.0
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvals, eigvecs


def _qaoa_expected_cut(n, edges, gammas, betas, cost_diag, mixer_eigvals, mixer_eigvecs, n_edges):
    """
    Compute expected cut for given QAOA parameters.
    Uses state-vector simulation with precomputed eigen-decompositions.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma * C): C is diagonal
        psi = psi * np.exp(-1j * gamma * cost_diag)
        # Apply exp(-i beta * M): use eigen decomposition
        coeffs = mixer_eigvecs.conj().T @ psi
        coeffs = coeffs * np.exp(-1j * beta * mixer_eigvals)
        psi = mixer_eigvecs @ coeffs
    
    # Expected cut = (n_edges - <C>) / 2 where <C> = sum |psi_i|^2 * cost_diag_i
    prob = np.abs(psi) ** 2
    exp_C = np.sum(prob * cost_diag)
    cut = (n_edges - exp_C) / 2.0
    return cut


def _objective(params, n, edges, cost_diag, mixer_eigvals, mixer_eigvecs, n_edges):
    """Minimize negative expected cut."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    cut = _qaoa_expected_cut(n, edges, gammas, betas, cost_diag, mixer_eigvals, mixer_eigvecs, n_edges)
    return -cut


def _optimize_angles(n, edges, p, cost_diag, mixer_eigvals, mixer_eigvecs, n_edges):
    """Optimize QAOA angles using L-BFGS-B with multiple restarts."""
    best_angles = None
    best_cut = -np.inf
    
    initial_guesses = [
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.2, 0.3, 0.2, 0.3]),
        np.array([0.4, 0.2, 0.4, 0.2]),
        np.array([0.7, 0.1, 0.3, 0.6]),
        np.array([0.1, 0.7, 0.6, 0.3]),
    ]
    
    rng = np.random.default_rng(12345)
    for _ in range(6):
        g = rng.uniform(0.0, np.pi, 2)
        b = rng.uniform(0.0, np.pi, 2)
        initial_guesses.append(np.concatenate([g, b]))
    
    bounds = [(0.0, np.pi)] * (2 * p)
    
    for x0 in initial_guesses:
        if _remaining_time() < 3.0:
            break
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, edges, cost_diag, mixer_eigvals, mixer_eigvecs, n_edges),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cut_val = -result.fun
            if cut_val > best_cut:
                best_cut = cut_val
                best_angles = result.x.copy()
        except Exception:
            continue
    
    if best_angles is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    gammas = best_angles[:p]
    betas = best_angles[p:]
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """Design graph-dependent QAOA angles for MaxCut."""
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if _remaining_time() < 2.0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    n_edges = len(edges)
    if n_edges == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Precompute
    cost_diag = _build_cost_eigs(n, edges)
    mixer_eigvals, mixer_eigvecs = _build_mixer_eig(n)
    
    # Optimize
    result = _optimize_angles(n, edges, p, cost_diag, mixer_eigvals, mixer_eigvecs, n_edges)
    
    return result
