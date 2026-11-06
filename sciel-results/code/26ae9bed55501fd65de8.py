"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for classical optimum and precomputed structures
_classical_opt_cache = {}
_graph_cache = {}


def _get_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    key = (n, tuple(sorted(tuple(sorted(e)) for e in edges)))
    if key in _graph_cache:
        return _graph_cache[key]
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    _graph_cache[key] = adj
    return adj


def _classical_maxcut(n, edges):
    """Exhaustive search for exact MaxCut on small graphs."""
    key = (n, tuple(sorted(tuple(sorted(e)) for e in edges)))
    if key in _classical_opt_cache:
        return _classical_opt_cache[key]
    
    # For n <= 20, exhaustive search is feasible
    best_cut = 0
    num_vertices = n
    # Use bit manipulation for speed
    for mask in range(1 << num_vertices):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
    _classical_opt_cache[key] = best_cut
    return best_cut


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for QAOA with given parameters.
    Uses the formula: E[C] = sum_{(u,v) in E} 0.5 * (1 - <sigma_z^u sigma_z^v>)
    
    For QAOA with p layers, the expectation <sigma_z^u sigma_z^v> can be computed
    via state evolution. For small n and p=2, we can use matrix methods.
    """
    # For efficiency, use the tensor product state approach
    # Initial state: |+>^n
    # After each layer: U_beta = exp(-i beta sum X_i), U_gamma = exp(-i gamma C)
    
    # For n up to 10, 2^n states are manageable (2^10 = 1024)
    dim = 1 << n
    
    # Initial state vector: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute X operator and C operator
    # X = sum_i sigma_x^i
    X = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        # sigma_x on qubit i: flip bit i
        bit = 1 << i
        for state in range(dim):
            flipped = state ^ bit
            X[flipped, state] += 1.0
    
    # C = sum_{(u,v) in E} (1 - sigma_z^u sigma_z^v) / 2
    # Actually, C = sum_{(u,v) in E} sigma_z^u sigma_z^v for MaxCut (up to constant)
    # The standard QAOA cost for MaxCut: C = sum_{(u,v) in E} (1 - sigma_z^u sigma_z^v)/2
    # But for the expectation, we need <psi| C |psi>
    # Let's use: C = sum_{(u,v) in E} 0.5 * (1 - sigma_z^u sigma_z^v)
    
    # Build sigma_z^u sigma_z^v for each edge
    C = np.zeros((dim, dim), dtype=np.complex128)
    for u, v in edges:
        bit_u = 1 << u
        bit_v = 1 << v
        for state in range(dim):
            sz_u = 1.0 if ((state >> u) & 1) == 0 else -1.0
            sz_v = 1.0 if ((state >> v) & 1) == 0 else -1.0
            C[state, state] += 0.5 * (1.0 - sz_u * sz_v)
    
    p = len(gammas)
    for k in range(p):
        # Apply U_gamma = exp(-i gamma_k * C)
        gamma_k = gammas[k]
        # Diagonal unitary
        psi = psi * np.exp(-1j * gamma_k * np.diag(C))
        
        # Apply U_beta = exp(-i beta_k * X)
        beta_k = betas[k]
        # Need to compute exp(-i beta_k * X) @ psi
        # For small beta or small dim, use matrix exponential
        # But X is sparse-like; for n<=10, dim=1024, this is feasible
        # Use scipy or numpy for matrix exponential
        from scipy.linalg import expm
        U_beta = expm(-1j * beta_k * X)
        psi = U_beta @ psi
    
    # Expected cut = <psi| C |psi>
    expected = np.vdot(psi, C @ psi).real
    return max(0.0, expected)


def _expected_cut_fast(n, edges, gammas, betas):
    """
    Faster computation using the known analytical form for small p.
    For p=1 or p=2, we can use correlation functions.
    """
    # For general approach, use the state vector method but optimize
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute X
    X = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        bit = 1 << i
        for state in range(dim):
            flipped = state ^ bit
            X[flipped, state] += 1.0
    
    # Precompute C diagonal
    C_diag = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        bit_u = 1 << u
        bit_v = 1 << v
        for state in range(dim):
            sz_u = 1.0 if ((state >> u) & 1) == 0 else -1.0
            sz_v = 1.0 if ((state >> v) & 1) == 0 else -1.0
            C_diag[state] += 0.5 * (1.0 - sz_u * sz_v)
    
    p = len(gammas)
    from scipy.linalg import expm
    
    for k in range(p):
        # Apply gamma
        psi = psi * np.exp(-1j * gammas[k] * C_diag)
        
        # Apply beta
        U_beta = expm(-1j * betas[k] * X)
        psi = U_beta @ psi
    
    expected = np.vdot(psi, C_diag * psi).real
    return max(0.0, expected)


def _objective(params, n, edges, classical_opt):
    """Objective function to maximize: expected_cut / classical_opt."""
    gammas = params[:2]
    betas = params[2:]
    ec = _expected_cut_fast(n, edges, gammas, betas)
    if classical_opt <= 0:
        return 0.0
    return ec / classical_opt


def _spectral_init(n, edges):
    """Spectral-based initialization for QAOA angles."""
    adj = _get_adjacency(n, edges)
    # Laplacian
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Eigenvector corresponding to smallest non-zero eigenvalue (Fiedler vector)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Skip the zero eigenvalue (constant vector)
        idx = 1
        fiedler = eigvecs[:, idx]
    except:
        fiedler = np.ones(n) / np.sqrt(n)
    
    # Use spectral information for initialization
    # Heuristic: gamma proportional to spectral gap, beta proportional to degree
    spectral_gap = eigvals[1] if len(eigvals) > 1 else 1.0
    avg_degree = degree.mean() if n > 0 else 1.0
    
    gamma_init = np.clip(0.5 * spectral_gap, 0.1, np.pi / 2)
    beta_init = np.clip(0.3 / (avg_degree + 1), 0.1, np.pi / 2)
    
    return np.array([gamma_init, gamma_init, beta_init, beta_init])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of vertices
        edges: list of edges (u, v)
        p: number of QAOA layers (expected to be 2)
    
    Returns:
        numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    if p == 0:
        return np.array([]).reshape(0, 2)
    
    # Get classical optimum
    classical_opt = _classical_maxcut(n, edges)
    
    if classical_opt <= 0:
        return np.zeros((p, 2))
    
    # Build adjacency
    adj = _get_adjacency(n, edges)
    
    # Spectral initialization
    init_params = _spectral_init(n, edges)
    
    # Multi-start optimization
    best_score = 0.0
    best_params = init_params.copy()
    
    # Generate multiple starting points
    np.random.seed(42)
    start_points = [init_params.copy()]
    
    # Add random perturbations
    for _ in range(5):
        perturbation = np.random.uniform(-0.5, 0.5, size=4)
        sp = init_params + perturbation
        sp = np.clip(sp, 0.01, np.pi - 0.01)
        start_points.append(sp)
    
    # Also add some structured starting points
    structured_starts = [
        np.array([0.5, 0.5, 0.3, 0.3]),
        np.array([1.0, 0.5, 0.5, 0.3]),
        np.array([0.3, 0.3, 0.5, 0.5]),
        np.array([0.8, 0.4, 0.4, 0.6]),
    ]
    for sp in structured_starts:
        start_points.append(np.clip(sp, 0.01, np.pi - 0.01))
    
    # Try each starting point with BFGS
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                lambda x: -_objective(x, n, edges, classical_opt),
                sp,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-5}
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with higher precision
    if time.time() < deadline - 10:
        try:
            result = minimize(
                lambda x: -_objective(x, n, edges, classical_opt),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract gammas and betas
    gammas = best_params[:2]
    betas = best_params[2:]
    
    # Ensure they are in reasonable range
    gammas = np.clip(gammas, 0.0, np.pi)
    betas = np.clip(betas, 0.0, np.pi)
    
    # Return as (p, 2) array
    result = np.column_stack((gammas, betas))
    
    return result
# EVOLVE-BLOCK-END
