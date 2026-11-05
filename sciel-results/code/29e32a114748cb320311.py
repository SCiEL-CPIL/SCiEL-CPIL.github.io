"""QAOA parameter design: graph-dependent two-layer angles for MaxCut."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Cache for exact MaxCut and objective computations
_exact_cache = {}
_obj_cache = {}


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    key = (n, tuple(sorted(edges)))
    if key in _exact_cache:
        return _exact_cache[key]
    
    best = 0
    # For n up to 10, brute force 2^n assignments
    # Use bit manipulation for speed
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    _exact_cache[key] = best
    return best


def _qaoa_expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value of QAOA with given parameters.
    
    For QAOA with p layers:
    - Start in |+>^n
    - For each layer i: apply exp(-i gamma_i * C) then exp(-i beta_i * sum X_j)
    
    The expected cut is sum over edges of <s_u * s_v> where s_j = +/-1.
    
    For small n (<=10), we can compute this exactly using the full state vector.
    """
    # Build the cost Hamiltonian matrix C (diagonal in computational basis)
    # C = sum_{(u,v) in edges} (1 - Z_u Z_v) / 2
    # Actually for MaxCut, C = sum_{(u,v)} (I - Z_u Z_v)/2
    # The expectation of C in state |psi> is <psi|C|psi>
    
    # For n <= 10, we can work with the full 2^n dimensional state vector
    dim = 1 << n
    
    # Initialize state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_u Z_v matrices or just compute the cost diagonal
    # C is diagonal: C[mask] = number of edges cut by assignment mask
    # where mask bit j = 0 means s_j = +1, bit j = 1 means s_j = -1
    # Actually let's use: s_j = +1 if bit is 0, -1 if bit is 1
    # Edge (u,v) is cut if s_u != s_v, i.e., bits differ
    
    cost_diag = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        cost_diag[mask] = cut
    
    # Mixer: sum X_j. In the computational basis, X flips bit j.
    # exp(-i beta * sum X_j) is a unitary that couples states differing by one bit.
    # For small n, we can construct the full mixer matrix or use the fact that
    # X_j = |0><1| + |1><0| on qubit j.
    
    # For efficiency, let's precompute the mixer matrix for each beta value
    # Actually, since we need to evaluate for many parameter sets, let's be smart.
    
    # The mixer U_M(beta) = exp(-i beta * H_M) where H_M = sum_j X_j
    # H_M has eigenvalues in {-n, -n+2, ..., n} with degeneracies C(n, (n-k)/2) for eigenvalue k
    # But for small n, let's just construct the matrix.
    
    # H_M in computational basis: H_M[mask, mask'] = 1 if mask' differs from mask in exactly one bit
    # Actually H_M = sum_j X_j, so <mask|H_M|mask'> = number of bits that differ between mask and mask'
    # Wait no: X_j flips bit j, so <mask|X_j|mask'> = 1 if mask' = mask with bit j flipped, 0 otherwise.
    # So <mask|H_M|mask'> = (number of bits where mask and mask' differ) if that number is 1, else 0?
    # No: H_M = X_0 + X_1 + ... + X_{n-1}
    # <mask|X_j|mask'> = 1 if mask' = mask ^ (1<<j), else 0
    # So <mask|H_M|mask'> = 1 if mask' = mask ^ (1<<j) for some j, i.e., Hamming distance 1.
    # But each such j gives exactly 1, so <mask|H_M|mask'> = 1 if Hamming distance is 1, else 0.
    
    # For n=6, dim=64; n=8, dim=256; n=10, dim=1024. These are manageable.
    
    # Let's precompute the mixer matrix
    H_M = np.zeros((dim, dim), dtype=np.float64)
    for mask in range(dim):
        for j in range(n):
            mask_flipped = mask ^ (1 << j)
            H_M[mask, mask_flipped] = 1.0
    
    # Now apply the QAOA layers
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma * C): C is diagonal, so this is just phase multiplication
        psi = psi * np.exp(-1j * gamma * cost_diag)
        
        # Apply exp(-i beta * H_M): need to diagonalize H_M or use matrix exponential
        # For small dim, use scipy's expm or eigendecomposition
        # H_M is symmetric real, so use eigh
        eigvals, eigvecs = np.linalg.eigh(H_M)
        U_M = (eigvecs * np.exp(-1j * beta * eigvals)) @ eigvecs.conj().T
        psi = U_M @ psi
    
    # Expected cut = <psi|C|psi> = sum_mask |psi[mask]|^2 * cost_diag[mask]
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * cost_diag)
    
    return expected_cut


def _objective(params, n, edges, exact_opt):
    """Objective function to minimize (negative of ratio)."""
    gammas = params[:2]
    betas = params[2:]
    
    # Keep angles in reasonable range
    gammas = np.clip(gammas, -np.pi, np.pi)
    betas = np.clip(betas, -np.pi, np.pi)
    
    try:
        exp_cut = _qaoa_expected_cut(n, edges, gammas, betas)
    except Exception:
        return 1e6
    
    ratio = exp_cut / exact_opt if exact_opt > 0 else 0
    return -ratio


def _optimize_for_graph(n, edges, p=2, time_budget=10.0):
    """Optimize QAOA parameters for a specific graph."""
    start_time = time.time()
    
    exact_opt = _exact_maxcut(n, edges)
    
    # Initial guess based on known good values for QAOA
    # For p=2, common good starting points are around gamma ~ 0.3-0.5, beta ~ 0.2-0.4
    x0 = np.array([0.35, 0.35, 0.25, 0.25])
    
    best_params = x0.copy()
    best_score = _objective(x0, n, edges, exact_opt)
    
    # Try multiple random starts with differential evolution or L-BFGS-B
    n_starts = 5
    
    for i in range(n_starts):
        if time.time() - start_time > time_budget:
            break
        
        if i == 0:
            x_start = x0
        else:
            # Random initialization within [-pi, pi]
            x_start = np.random.uniform(-np.pi, np.pi, size=4)
        
        try:
            result = minimize(
                _objective,
                x_start,
                args=(n, edges, exact_opt),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * 4,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # If we have time, try differential evolution for global search
    if time.time() - start_time < time_budget * 0.5:
        try:
            de_result = differential_evolution(
                _objective,
                bounds=[(-np.pi, np.pi)] * 4,
                args=(n, edges, exact_opt),
                maxiter=20,
                popsize=10,
                tol=1e-8,
                seed=42,
                polish=True
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    gammas = best_params[:2]
    betas = best_params[2:]
    
    return np.array([[gammas[0], betas[0]], [gammas[1], betas[1]]])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits/nodes
        edges: list of edges (u, v)
        p: number of QAOA layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    # Deadline guard: 150s total, but we want to be safe
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if p != 2:
        # Fallback for unexpected p
        return np.tile([0.35, 0.25], (p, 1))
    
    # Optimize for this specific graph
    try:
        result = _optimize_for_graph(n, edges, p=2, time_budget=10.0)
        return result
    except Exception:
        # Fallback to reasonable defaults
        return np.array([[0.35, 0.25], [0.35, 0.25]])
