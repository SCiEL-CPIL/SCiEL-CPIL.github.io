"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
import itertools
import sys


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _expected_cut_value(gamma1, beta1, gamma2, beta2, n, edges):
    """
    Compute the expected cut value for a 2-layer QAOA circuit.
    
    Circuit: |+>^n -> exp(-i*gamma1*C) -> exp(-i*beta1*sum X) -> exp(-i*gamma2*C) -> exp(-i*beta2*sum X)
    
    The expected cut is E[sum_{(u,v) in E} (1 - Z_u Z_v) / 2]
    
    We compute this exactly using the tensor network / state vector approach for small n.
    For n up to 10, we can use the density matrix or direct state vector.
    """
    # For small n (<=10), we can compute the expectation values directly
    # The key insight: after the circuit, we need <Z_u Z_v> for each edge (u,v)
    
    # State vector approach: start with |+>^n, apply gates
    # |+>^n = (1/sqrt(2^n)) * sum over all bitstrings
    
    # The CNOT-based MaxCut Hamiltonian: C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # But for QAOA, we apply exp(-i*gamma*C) where C is the MaxCut cost function
    
    # Actually, let's think about this more carefully.
    # The QAOA circuit applies:
    # 1. exp(-i * gamma1 * C) where C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    # 2. exp(-i * beta1 * B) where B = sum_j X_j
    # 3. exp(-i * gamma2 * C)
    # 4. exp(-i * beta2 * B)
    
    # For small n, we can compute the state vector directly.
    # n <= 10 means 2^10 = 1024 basis states, which is manageable.
    
    if n > 12:
        # Fallback: use a simpler approximation
        return _expected_cut_approx(gamma1, beta1, gamma2, beta2, n, edges)
    
    dim = 1 << n
    # Initial state: |+>^n = (1/sqrt(dim)) * ones(dim)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_u Z_v for each edge
    # Z_u Z_v acting on basis state |x> gives (-1)^{x_u + x_v} |x>
    # So exp(-i*gamma*C) = exp(-i*gamma * sum_{(u,v)} (1 - Z_u Z_v)/2)
    # = exp(-i*gamma * |E|/2) * exp(i*gamma/2 * sum_{(u,v)} Z_u Z_v)
    # = phase * prod_{(u,v)} exp(i*gamma/2 * Z_u Z_v)
    # = phase * prod_{(u,v)} diag(exp(i*gamma/2 * (-1)^{x_u+x_v}))
    
    # For each basis state x, the phase from exp(-i*gamma*C) is:
    # exp(-i*gamma * cut(x)/2) where cut(x) = number of edges cut by assignment x
    # Wait, let me be more careful.
    # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    # For a computational basis state |x>, Z_u Z_v |x> = (-1)^{x_u + x_v} |x>
    # So C|x> = sum_{(u,v)} (1 - (-1)^{x_u+x_v})/2 |x> = cut(x) |x>
    # where cut(x) = number of edges (u,v) such that x_u != x_v
    
    # So exp(-i*gamma*C)|x> = exp(-i*gamma*cut(x)) |x>
    
    # Similarly, exp(-i*beta*B) where B = sum_j X_j
    # X_j flips bit j, so this is a multi-qubit X rotation.
    # exp(-i*beta * sum_j X_j) = prod_j exp(-i*beta * X_j)
    # Each exp(-i*beta*X_j) = cos(beta)*I - i*sin(beta)*X_j
    # This is not diagonal in the computational basis.
    
    # Let's just apply the gates sequentially using the state vector.
    
    # Step 1: Apply exp(-i*gamma1*C)
    # This is diagonal in the computational basis
    cut_values = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        cut_count = 0
        for u, v in edges:
            if ((x >> u) & 1) != ((x >> v) & 1):
                cut_count += 1
        cut_values[x] = cut_count
    
    psi *= np.exp(-1j * gamma1 * cut_values)
    
    # Step 2: Apply exp(-i*beta1 * sum_j X_j)
    # This equals prod_j (cos(beta1)*I_j - i*sin(beta1)*X_j)
    # We can apply this by iterating over each qubit
    c = np.cos(beta1)
    s = np.sin(beta1)
    
    for j in range(n):
        # Apply (c*I - i*s*X) on qubit j
        # For each pair of basis states differing only in bit j:
        # |...0...> -> c|...0...> - i*s|...1...>
        # |...1...> -> c|...1...> - i*s|...0...>
        for mask in range(dim):
            if (mask >> j) & 1 == 0:
                partner = mask | (1 << j)
                old_psi = psi[mask]
                old_partner = psi[partner]
                psi[mask] = c * old_psi - 1j * s * old_partner
                psi[partner] = c * old_partner - 1j * s * old_psi
    
    # Step 3: Apply exp(-i*gamma2*C)
    psi *= np.exp(-1j * gamma2 * cut_values)
    
    # Step 4: Apply exp(-i*beta2 * sum_j X_j)
    c = np.cos(beta2)
    s = np.sin(beta2)
    
    for j in range(n):
        for mask in range(dim):
            if (mask >> j) & 1 == 0:
                partner = mask | (1 << j)
                old_psi = psi[mask]
                old_partner = psi[partner]
                psi[mask] = c * old_psi - 1j * s * old_partner
                psi[partner] = c * old_partner - 1j * s * old_psi
    
    # Now compute the expected cut: E[sum_{(u,v)} (1 - Z_u Z_v)/2]
    # = sum_{(u,v)} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_x |psi_x|^2 * (-1)^{x_u + x_v}
    
    prob = np.abs(psi) ** 2
    
    expected_cut = 0.0
    for u, v in edges:
        zz = 0.0
        for x in range(dim):
            zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
            zz += prob[x] * zu * zv
        expected_cut += (1.0 - zz) / 2.0
    
    return expected_cut


def _expected_cut_approx(gamma1, beta1, gamma2, beta2, n, edges):
    """Approximate expected cut for larger n using a simpler model."""
    # For larger n, use a mean-field approximation
    # This is a rough approximation
    m = len(edges)
    if m == 0:
        return 0.0
    
    # Simple approximation based on the idea that each edge is cut with some probability
    # depending on the angles
    # This is very rough
    p_cut = 0.5 * (1 - np.cos(gamma1) * np.cos(beta1) * np.cos(gamma2) * np.cos(beta2))
    return m * p_cut


def _classical_optimum(n, edges):
    """Compute the exact classical MaxCut optimum by exhaustive search."""
    if n > 15:
        # Fallback: use a heuristic
        return _maxcut_heuristic(n, edges)
    
    best_cut = 0
    dim = 1 << n
    for x in range(dim):
        cut = 0
        for u, v in edges:
            if ((x >> u) & 1) != ((x >> v) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return float(best_cut)


def _maxcut_heuristic(n, edges):
    """Heuristic MaxCut for larger graphs."""
    # Simple greedy
    assignment = np.zeros(n, dtype=int)
    # Random initialization
    rng = np.random.RandomState(42)
    assignment = rng.randint(0, 2, size=n)
    
    def cut_value(assign):
        return sum(1 for u, v in edges if assign[u] != assign[v])
    
    best = cut_value(assignment)
    # Local search
    for _ in range(100):
        improved = False
        for i in range(n):
            new_assign = assignment.copy()
            new_assign[i] = 1 - new_assign[i]
            val = cut_value(new_assign)
            if val > best:
                best = val
                assignment = new_assign
                improved = True
        if not improved:
            break
    return float(best)


def _spectral_init(n, edges):
    """Spectral-based initialization for QAOA angles."""
    A = _build_adjacency(n, edges)
    m = len(edges)
    
    if m == 0:
        return [0.1, 0.1, 0.1, 0.1]
    
    # Compute eigenvalues of the adjacency matrix
    eigvals = np.linalg.eigvalsh(A)
    lambda_max = eigvals[-1]
    
    # Initialize based on spectral gap and density
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # Heuristic initialization
    gamma1 = np.pi / (2.0 * max(1, int(np.sqrt(m))))
    beta1 = np.pi / (4.0 * max(1, n))
    gamma2 = gamma1 * 0.5
    beta2 = beta1 * 0.7
    
    # Clip to reasonable ranges
    gamma1 = np.clip(gamma1, 0.01, np.pi / 2)
    beta1 = np.clip(beta1, 0.01, np.pi / 4)
    gamma2 = np.clip(gamma2, 0.01, np.pi / 2)
    beta2 = np.clip(beta2, 0.01, np.pi / 4)
    
    return [gamma1, beta1, gamma2, beta2]


def _objective(params, n, edges, opt_val):
    """Objective function to minimize (negative ratio)."""
    gamma1, beta1, gamma2, beta2 = params
    try:
        exp_cut = _expected_cut_value(gamma1, beta1, gamma2, beta2, n, edges)
        if opt_val <= 0:
            return 0.0
        ratio = exp_cut / opt_val
        return -ratio
    except Exception:
        return 0.0


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of edges (u, v)
        p: number of layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Ensure p is handled correctly
    if p != 2:
        # Fallback for unexpected p
        return np.tile([0.35, 0.25], (p, 1))
    
    # Compute classical optimum
    opt_val = _classical_optimum(n, edges)
    if opt_val <= 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Get spectral initialization
    init_params = _spectral_init(n, edges)
    
    # Grid search over a coarse parameter space
    # gamma in [0.05, pi/2], beta in [0.05, pi/4]
    gamma_grid = np.linspace(0.05, np.pi / 2, 8)
    beta_grid = np.linspace(0.05, np.pi / 4, 8)
    
    best_score = -np.inf
    best_params = init_params
    
    # Coarse grid search
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    if time.time() > deadline - 30:
                        break
                    params = [g1, b1, g2, b2]
                    score = -_objective(params, n, edges, opt_val)
                    if score > best_score:
                        best_score = score
                        best_params = params[:]
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Also try the spectral init
    score = -_objective(init_params, n, edges, opt_val)
    if score > best_score:
        best_score = score
        best_params = init_params[:]
    
    # Nelder-Mead refinement from best grid point
    if time.time() < deadline - 15:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, edges, opt_val),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < -best_score:
                best_params = result.x.tolist()
                best_score = -result.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, edges, opt_val),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < -best_score:
                best_params = result.x.tolist()
                best_score = -result.fun
        except Exception:
            pass
    
    # Clamp parameters to valid ranges
    gamma1 = np.clip(best_params[0], 0.0, np.pi)
    beta1 = np.clip(best_params[1], 0.0, np.pi)
    gamma2 = np.clip(best_params[2], 0.0, np.pi)
    beta2 = np.clip(best_params[3], 0.0, np.pi)
    
    return np.array([[gamma1, beta1], [gamma2, beta2]])
# EVOLVE-BLOCK-END
