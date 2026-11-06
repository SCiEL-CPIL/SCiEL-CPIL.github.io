"""QAOA parameter design for MaxCut on small graphs."""
import numpy as np
import time
from itertools import combinations

# Deadline guard
_DEADLINE = 120.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edges:
        if 0 <= u < n and 0 <= v < n:
            A[u, v] = 1.0
            A[v, u] = 1.0
    return A


def _exact_maxcut(n, edges):
    """Exhaustive search for exact MaxCut value."""
    best = 0
    for mask in range(1, 1 << n):
        cut = 0
        for (u, v) in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _expected_cut_p2(n, edges, gamma, beta):
    """
    Compute the expected cut value for p=2 QAOA on a given graph.
    
    The circuit: |0>^n -> H^n -> exp(-i gamma C) -> exp(-i beta * sum X) -> 
                 exp(-i gamma C) -> exp(-i beta * sum X)
    
    For p=2, the expected cut can be computed analytically.
    
    The cost Hamiltonian C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    The mixer M = sum_i X_i
    
    After the full circuit, the expected value of C is:
    <C> = sum_{(i,j) in E} (1 - <Z_i Z_j>)/2
    
    We need to compute <Z_i Z_j> for each edge after the p=2 circuit.
    
    For p=2, the state is:
    |psi> = exp(-i b2 M) exp(-i g2 C) exp(-i b1 M) exp(-i g1 C) |+>^n
    
    This is complex to compute directly. Let's use the standard approach:
    simulate the state vector for small n.
    """
    # For small n (<=10), we can simulate the state vector directly
    dim = 1 << n
    
    # Start state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_i and X_i operators as sparse-like operations
    # Z_i flips sign of amplitudes where bit i is 1
    # X_i flips bit i
    
    def apply_Z_psi(psi, i):
        """Apply Z_i to psi (diagonal operation)."""
        mask = 1 << i
        # Where bit i is set, flip sign
        for k in range(dim):
            if k & mask:
                psi[k] = -psi[k]
        return psi
    
    def apply_X_psi(psi, i):
        """Apply X_i to psi (permutation operation)."""
        mask = 1 << i
        new_psi = np.empty_like(psi)
        for k in range(dim):
            if k & mask:
                new_psi[k] = psi[k ^ mask]
            else:
                new_psi[k] = psi[k ^ mask]
        return new_psi
    
    # Cost operator C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    # exp(-i gamma C) = exp(-i gamma/2 * sum_{(i,j)} (I - Z_i Z_j))
    # This is a product of commuting terms since all Z_i Z_j commute.
    # exp(-i gamma/2 * (I - Z_i Z_j)) = exp(-i gamma/2) * exp(i gamma/2 * Z_i Z_j)
    # = exp(-i gamma/2) * cos(gamma/2) * I + i * exp(-i gamma/2) * sin(gamma/2) * Z_i Z_j
    
    # Actually, for the full cost:
    # C = sum_e (1 - Z_i Z_j)/2
    # exp(-i gamma C) = prod_e exp(-i gamma (1 - Z_i Z_j)/2)
    # = prod_e [cos(gamma/2) I + i sin(gamma/2) Z_i Z_j]  (times global phase)
    
    # For efficiency with small n, let's just use matrix exponentiation approach
    # Build the cost matrix and use expm
    
    from scipy.linalg import expm
    
    # Build C as a matrix
    C = np.zeros((dim, dim), dtype=np.complex128)
    for (u, v) in edges:
        # Z_u Z_v operator
        mask_u = 1 << u
        mask_v = 1 << v
        for k in range(dim):
            # Z_u Z_v |k> = (-1)^{bit_u(k) + bit_v(k)} |k>
            sign = 1.0
            if k & mask_u:
                sign *= -1.0
            if k & mask_v:
                sign *= -1.0
            C[k, k] += 0.5 * (1.0 - sign)
    
    # Apply exp(-i gamma C)
    U_cost = expm(-1j * gamma * C)
    psi = U_cost @ psi
    
    # Apply exp(-i beta * sum X)
    # M = sum_i X_i
    M = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            M[k ^ mask, k] = 1.0
    
    U_mixer = expm(-1j * beta * M)
    psi = U_mixer @ psi
    
    # Apply second layer
    U_cost2 = expm(-1j * gamma * C)
    psi = U_cost2 @ psi
    U_mixer2 = expm(-1j * beta * M)
    psi = U_mixer2 @ psi
    
    # Compute expected cut
    # <C> = sum_e (1 - <Z_i Z_j>)/2
    expected_cut = 0.0
    for (u, v) in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        zz = 0.0
        for k in range(dim):
            sign = 1.0
            if k & mask_u:
                sign *= -1.0
            if k & mask_v:
                sign *= -1.0
            zz += sign * np.abs(psi[k])**2
        expected_cut += 0.5 * (1.0 - zz)
    
    return expected_cut


def _expected_cut_fast(n, edges, g1, b1, g2, b2):
    """
    Faster computation using the fact that for p=2, we can use
    the structure more efficiently. For very small n, direct simulation is fine.
    """
    if _remaining() < 5.0:
        return 0.0
    
    dim = 1 << n
    
    # Start state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_i Z_j signs for all edges
    edge_signs = []
    for (u, v) in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        signs = np.ones(dim, dtype=np.float64)
        for k in range(dim):
            s = 1.0
            if k & mask_u:
                s *= -1.0
            if k & mask_v:
                s *= -1.0
            signs[k] = s
        edge_signs.append(signs)
    
    # Build cost diagonal (C is diagonal in computational basis for MaxCut)
    C_diag = np.zeros(dim, dtype=np.float64)
    for signs in edge_signs:
        C_diag += 0.5 * (1.0 - signs)
    
    # exp(-i gamma C) is diagonal since C is diagonal
    # U_cost[k,k] = exp(-i gamma * C_diag[k])
    
    # Build mixer matrix M = sum X_i
    # This is a permutation-like matrix
    M = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            M[k ^ mask, k] = 1.0
    
    # For small dim, use expm
    from scipy.linalg import expm
    
    U_mixer = expm(-1j * b1 * M)
    U_mixer2 = expm(-1j * b2 * M)
    
    # Apply: psi -> U_cost2 @ U_mixer2 @ U_cost1 @ U_mixer1 @ psi
    # But U_cost is diagonal, so we can apply it element-wise
    
    # Step 1: psi = U_mixer1 @ psi
    psi = U_mixer @ psi
    
    # Step 2: psi = U_cost1 @ psi (diagonal)
    psi *= np.exp(-1j * g1 * C_diag)
    
    # Step 3: psi = U_mixer2 @ psi
    psi = U_mixer2 @ psi
    
    # Step 4: psi = U_cost2 @ psi (diagonal)
    psi *= np.exp(-1j * g2 * C_diag)
    
    # Compute expected cut
    probs = np.abs(psi)**2
    expected_cut = 0.0
    for signs in edge_signs:
        zz = np.sum(signs * probs)
        expected_cut += 0.5 * (1.0 - zz)
    
    return expected_cut


def _optimize_angles_for_graph(n, edges, exact_opt):
    """Optimize QAOA angles for a single graph."""
    if _remaining() < 10.0:
        return np.array([[0.35, 0.25], [0.25, 0.15]])
    
    # Use Nelder-Mead with a good initial guess based on graph properties
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # Heuristic initial guesses
    g1_init = 0.4 * (1.0 + 0.5 * density)
    b1_init = 0.3 * (1.0 - 0.3 * density)
    g2_init = 0.3 * (1.0 + 0.3 * density)
    b2_init = 0.2 * (1.0 - 0.2 * density)
    
    def objective(params):
        g1, b1, g2, b2 = params
        try:
            ec = _expected_cut_fast(n, edges, g1, b1, g2, b2)
            return -ec / exact_opt  # minimize negative ratio
        except:
            return 10.0
    
    from scipy.optimize import minimize
    
    x0 = np.array([g1_init, b1_init, g2_init, b2_init])
    
    # Try multiple starting points
    best_result = None
    best_val = np.inf
    
    starts = [
        x0,
        np.array([0.5, 0.2, 0.3, 0.1]),
        np.array([0.3, 0.4, 0.5, 0.2]),
        np.array([0.6, 0.1, 0.4, 0.3]),
        np.array([0.2, 0.5, 0.3, 0.4]),
    ]
    
    for s in starts:
        if _remaining() < 15.0:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            val = -res.fun
            if val < best_val:
                best_val = val
                best_result = res.x
        except:
            pass
    
    if best_result is None:
        return np.array([[g1_init, b1_init], [g2_init, b2_init]])
    
    g1, b1, g2, b2 = best_result
    return np.array([[g1, b1], [g2, b2]])


def _graph_cases(seed):
    """Generate development graphs."""
    rng = np.random.RandomState(seed)
    cases = []
    for n in [6, 8, 10]:
        for density in [0.25, 0.55]:
            max_edges = n * (n - 1) // 2
            target_edges = int(round(density * max_edges))
            all_pairs = list(combinations(range(n), 2))
            edges = list(rng.choice(len(all_pairs), size=target_edges, replace=False))
            edge_list = [all_pairs[i] for i in edges]
            cases.append((n, edge_list))
    return cases


def _public_graphs():
    """Generate the six public evaluation graphs (seed 713)."""
    rng = np.random.RandomState(713)
    cases = []
    for n in [6, 8, 10]:
        for density in [0.25, 0.55]:
            max_edges = n * (n - 1) // 2
            target_edges = int(round(density * max_edges))
            all_pairs = list(combinations(range(n), 2))
            edges = list(rng.choice(len(all_pairs), size=target_edges, replace=False))
            edge_list = [all_pairs[i] for i in edges]
            cases.append((n, edge_list))
    return cases


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of (u, v) edges
        p: number of layers (should be 2)
    
    Returns:
        array of shape (p, 2) with rows (gamma, beta) in radians
    """
    global _start_time
    _start_time = time.time()
    
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if _remaining() < 10.0:
        return np.array([[0.35, 0.25], [0.25, 0.15]])
    
    # Compute exact MaxCut
    exact_opt = _exact_maxcut(n, edges)
    
    if exact_opt == 0:
        return np.array([[0.0, 0.0], [0.0, 0.0]])
    
    # Optimize angles for this specific graph
    result = _optimize_angles_for_graph(n, edges, exact_opt)
    
    # Ensure valid output
    if result.shape != (2, 2):
        result = np.array([[0.35, 0.25], [0.25, 0.15]])
    
    return result
