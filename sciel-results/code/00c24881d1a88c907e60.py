"""Optimized QAOA angle design with graph-dependent precomputation and L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools
from functools import lru_cache

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0  # seconds
_SAFETY_MARGIN = 10.0  # return best before this


def _remaining_time():
    return _TIME_LIMIT - _SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_hamiltonian(n, edges):
    """Build the MaxCut cost Hamiltonian matrix (sparse-friendly via dense for small n)."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    for u, v in edges:
        # Z_u Z_v = 1 if same spin, -1 if different
        # We want to maximize cut, so cost term is -Z_u Z_v (reward cut edges)
        # Actually for MaxCut, the cost Hamiltonian C = sum_{(u,v)} Z_u Z_v
        # We want to maximize <C>, which is equivalent to minimizing -<C>
        # In QAOA, we apply exp(-i gamma C), so C should be the MaxCut operator
        # C = sum_{(u,v) in E} Z_u Z_v
        # Z_u Z_v has eigenvalue +1 when bits are same, -1 when different
        # So -Z_u Z_v has eigenvalue -1 when same, +1 when different -> this rewards cuts
        # The standard QAOA for MaxCut uses C = sum Z_u Z_v and maximizes <C>
        # <C> = sum_{edges} (P_same - P_diff) = |E| - 2 * cut_size
        # So maximizing <C> minimizes cut_size... that's wrong.
        # Let me reconsider: for MaxCut we want to maximize the number of cut edges.
        # The cost function for MaxCut is typically H_C = sum_{(u,v)} (1 - Z_u Z_v)/2 = |E|/2 - sum Z_u Z_v / 2
        # Or equivalently, we can use H_C = -sum Z_u Z_v and maximize <H_C>
        # Let's use C = sum_{(u,v)} Z_u Z_v and note that <C> = |E| - 2*cut
        # So maximizing cut = minimizing <C> = maximizing -<C>
        # In QAOA, the standard formulation uses U_C = exp(-i gamma * C) where C is the cost Hamiltonian
        # and we maximize <psi| C |psi>. But for MaxCut, we want to maximize cut, not <C>.
        # Actually, the standard QAOA for MaxCut uses the cost Hamiltonian H_C = sum_{(u,v)} Z_u Z_v
        # and the objective is to maximize <H_C>. Since <H_C> = |E| - 2*cut, maximizing <H_C>
        # minimizes cut. That's the opposite of what we want!
        # The correct approach: use H_C = -sum Z_u Z_v so that <H_C> = 2*cut - |E|, and maximize <H_C>.
        # Or equivalently, use H_C = sum Z_u Z_v and minimize <H_C>.
        # In the QAOA circuit, we apply exp(-i gamma H_C). The sign of gamma can be adjusted.
        # Let's use the standard: H_C = sum_{(u,v)} Z_u Z_v, and we'll maximize -<H_C> by using
        # the negative in our objective function for optimization.
        
        # Build Z_u Z_v operator
        ZuZv = np.zeros((2**n, 2**n), dtype=np.complex128)
        for state in range(2**n):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu == bv:
                ZuZv[state, state] = 1.0
            else:
                ZuZv[state, state] = -1.0
        H += ZuZv
    return H


def _build_mixer(n):
    """Build the transverse field mixer Hamiltonian sum X_i."""
    M = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n):
        for state in range(2**n):
            new_state = state ^ (1 << i)
            M[state, new_state] += 1.0
    return M


def _apply_unitary(psi, H, angle):
    """Apply exp(-i * angle * H) to state vector psi."""
    # For small matrices, use eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H)
    # exp(-i * angle * H) = V diag(exp(-i * angle * eigvals)) V^dagger
    phase = np.exp(-1j * angle * eigvals)
    # psi' = V @ (phase * (V^dagger @ psi))
    coeffs = eigvecs.conj().T @ psi
    new_coeffs = phase * coeffs
    return eigvecs @ new_coeffs


def _qaoa_state(n, edges, gammas, betas, H, M):
    """Compute the final state vector for given QAOA parameters."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    for gamma, beta in zip(gammas, betas):
        psi = _apply_unitary(psi, H, gamma)
        psi = _apply_unitary(psi, M, beta)
    return psi


def _expected_cut(psi, H, n_edges):
    """Compute expected cut size from state vector. <H> = n_edges - 2*cut, so cut = (n_edges - <H>)/2."""
    # H = sum Z_u Z_v, so <H> = sum <Z_u Z_v> = n_edges - 2*cut
    h_val = np.real(np.vdot(psi, H @ psi))
    cut = (n_edges - h_val) / 2.0
    return cut


def _exact_maxcut(n, edges):
    """Brute-force exact MaxCut for small graphs."""
    best = 0
    for mask in range(2**n):
        cut = 0
        for u, v in edges:
            bu = (mask >> u) & 1
            bv = (mask >> v) & 1
            if bu != bv:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, edges, H, M, n_edges):
    """Objective to minimize: negative of expected cut ratio proxy (just expected cut)."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    psi = _qaoa_state(n, edges, gammas, betas, H, M)
    cut = _expected_cut(psi, H, n_edges)
    return -cut  # minimize negative cut = maximize cut


def _optimize_angles(n, edges, p, H, M, n_edges, exact_opt):
    """Optimize QAOA angles using L-BFGS-B with multiple restarts."""
    best_angles = None
    best_cut = -np.inf
    
    # Initial guesses
    initial_guesses = [
        np.array([0.35, 0.25] * p),  # seed program
        np.array([0.1, 0.1] * p),
        np.array([0.5, 0.5] * p),
        np.array([0.2, 0.3] * p),
        np.array([0.4, 0.2] * p),
    ]
    
    # Add some random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        g = rng.uniform(0.0, 1.0, p)
        b = rng.uniform(0.0, 1.0, p)
        initial_guesses.append(np.concatenate([g, b]))
    
    for x0 in initial_guesses:
        if _remaining_time() < 5.0:
            break
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, edges, H, M, n_edges),
                method='L-BFGS-B',
                bounds=[(0.0, np.pi)] * (2 * p),
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if result.success or result.fun < -best_cut:
                cut_val = -result.fun
                if cut_val > best_cut:
                    best_cut = cut_val
                    best_angles = result.x
        except Exception:
            continue
    
    if best_angles is None:
        # Fallback to seed program values
        return np.tile([0.35, 0.25], (p, 1))
    
    # Reshape to (p, 2) array with columns (gamma, beta)
    p_actual = len(best_angles) // 2
    gammas = best_angles[:p_actual]
    betas = best_angles[p_actual:]
    angles_arr = np.column_stack([gammas, betas])
    return angles_arr


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    if p != 2:
        # Fallback for unexpected p
        return np.tile([0.35, 0.25], (p, 1))
    
    if _remaining_time() < 2.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    n_edges = len(edges)
    if n_edges == 0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Build Hamiltonians
    H = _build_hamiltonian(n, edges)
    M = _build_mixer(n)
    
    # Compute exact MaxCut for reference (not directly used in optimization but could help)
    exact_opt = _exact_maxcut(n, edges)
    
    # Optimize
    angles_arr = _optimize_angles(n, edges, p, H, M, n_edges, exact_opt)
    
    if _remaining_time() < 1.0:
        return angles_arr
    
    return angles_arr
