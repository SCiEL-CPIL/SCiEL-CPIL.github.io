"""
QAOA parameter optimization for MaxCut on small graphs.
Fixed-depth (p=2) QAOA: angles(n, edges, p) -> (p, 2) array of (gamma, beta).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Cache for computed optimal angles
_angle_cache = {}

def _exact_maxcut(n, edges):
    """Brute-force MaxCut for small graphs (n <= 12)."""
    if n == 0:
        return 0
    m = len(edges)
    best = 0
    # For n up to ~14, 2^n is feasible
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute the expected cut value for a QAOA circuit with given parameters.
    Uses exact statevector simulation for small n (n <= 16).
    
    Circuit: |+>^n, then for each layer: exp(-i gamma C) then exp(-i beta * sum X_i)
    where C = sum_{(u,v) in E} (1 - Z_u Z_v)/2  (MaxCut cost)
    
    Returns expected number of cut edges.
    """
    # Build adjacency
    adj = [[0] * n for _ in range(n)]
    edge_set = set()
    for (u, v) in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        edge_set.add((u, v)) if u < v else edge_set.add((v, u))
    
    m = len(edges)
    
    # Statevector of size 2^n
    dim = 1 << n
    # Start in |+>^n: uniform superposition
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost unitary: exp(-i gamma * C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
        # So exp(-i gamma C) = product over edges of exp(-i gamma/2 * (1 - Z_u Z_v))
        # = product over edges of exp(-i gamma/2) * exp(i gamma/2 * Z_u Z_v)
        # The global phase exp(-i gamma/2 * m) can be ignored.
        # So we apply exp(i gamma/2 * Z_u Z_v) for each edge.
        # Z_u Z_v is diagonal: eigenvalue +1 if same bit, -1 if different.
        phase = np.ones(dim, dtype=np.complex128)
        for (u, v) in edges:
            # For each basis state, determine if bits u and v are same
            # Z_u Z_v eigenvalue: +1 if same, -1 if different
            # phase contribution: exp(i gamma/2 * eigenvalue)
            # We can compute this efficiently:
            mask_u = 1 << u
            mask_v = 1 << v
            for basis in range(dim):
                bit_u = (basis >> u) & 1
                bit_v = (basis >> v) & 1
                zz = 1 if bit_u == bit_v else -1
                phase[basis] *= np.exp(1j * gamma / 2 * zz)
        psi = phase * psi
        
        # Apply mixing unitary: exp(-i beta * sum_i X_i)
        # X_i flips bit i. sum_i X_i is not diagonal.
        # For small n, we can construct the full matrix or use the fact that
        # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
        # And these commute for different i, so product over i.
        # This is a product of 2x2 blocks in each bit position.
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            # Apply exp(-i beta X_i) to psi
            # X_i flips bit i: maps |...b_i...> to |...1-b_i...>
            mask_i = 1 << i
            for basis in range(dim):
                if (basis & mask_i) == 0:
                    partner = basis | mask_i
                    psi[basis], psi[partner] = (c * psi[basis] - 1j * s * psi[partner],
                                                  -1j * s * psi[basis] + c * psi[partner])
                # else: already handled by the partner
    
    # Now compute expected cut: sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        zz_exp = 0.0
        for basis in range(dim):
            bit_u = (basis >> u) & 1
            bit_v = (basis >> v) & 1
            zz = 1 if bit_u == bit_v else -1
            zz_exp += probs[basis] * zz
        expected_cut += (1 - zz_exp) / 2
    
    return expected_cut


def _qaoa_expectation_fast(n, edges, gammas, betas):
    """
    Faster QAOA expectation using tensor-product structure for the mixing step
    and diagonal cost step. For n <= 16 this should be fast enough.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Cost step: diagonal phase
        phase = np.ones(dim, dtype=np.complex128)
        for (u, v) in edges:
            mu = 1 << u
            mv = 1 << v
            for b in range(dim):
                same = ((b >> u) & 1) == ((b >> v) & 1)
                zz = 1 if same else -1
                phase[b] *= np.exp(1j * gamma * 0.5 * zz)
        psi *= phase
        
        # Mixing step: product of exp(-i beta X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            mi = 1 << i
            for b in range(dim // 2):
                idx0 = b * 2
                idx1 = idx0 + 1
                # Only if bit i is the least significant varying bit
                # Actually, let's do it properly:
                pass
        
        # Better: iterate over all basis states in pairs
        for i in range(n):
            mi = 1 << i
            for b in range(dim):
                if (b & mi) == 0:
                    p = b | mi
                    a = psi[b]
                    b2 = psi[p]
                    psi[b] = c * a - 1j * s * b2
                    psi[p] = -1j * s * a + c * b2
    
    # Expected cut
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        zz_exp = 0.0
        for b in range(dim):
            same = ((b >> u) & 1) == ((b >> v) & 1)
            zz = 1 if same else -1
            zz_exp += probs[b] * zz
        expected_cut += (1.0 - zz_exp) * 0.5
    
    return expected_cut


def _optimize_angles(n, edges, p, max_time=110.0):
    """
    Optimize QAOA parameters for a given graph.
    Uses Nelder-Mead with multiple random restarts.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    deadline = start_time + max_time
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        return -_qaoa_expectation(n, edges, gammas, betas)
    
    best_val = float('inf')
    best_params = np.array([0.35, 0.25, 0.35, 0.25])  # default fallback
    
    # Multiple restarts
    n_restarts = 10
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initial guess
        x0 = np.concatenate([
            np.random.uniform(0, np.pi, p),
            np.random.uniform(0, np.pi, p)
        ])
        
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x.copy()
        
        if time.time() > deadline - 15:
            break
    
    # Also try a few structured initial guesses
    structured = [
        [0.35, 0.25, 0.35, 0.25],
        [0.5, 0.25, 0.35, 0.15],
        [0.4, 0.3, 0.3, 0.2],
        [np.pi/4, np.pi/6, np.pi/5, np.pi/8],
    ]
    for x0 in structured:
        if time.time() > deadline - 10:
            break
        x0 = np.array(x0, dtype=np.float64)
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x.copy()
    
    # Convert to (p, 2) array: rows are (gamma_i, beta_i)
    gammas = best_params[:p]
    betas = best_params[p:]
    result = np.column_stack([gammas, betas])
    return result


def _optimize_angles_v2(n, edges, p, max_time=110.0):
    """
    Optimized version with better handling of the mixing step.
    Uses the fact that the mixing unitary is a tensor product of single-qubit rotations.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Precompute edge bitmasks
    edge_data = []
    for (u, v) in edges:
        edge_data.append((u, v))
    
    dim = 1 << n
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        
        # Statevector
        psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        for gamma, beta in zip(gammas, betas):
            # Cost: diagonal phase
            phase = np.ones(dim, dtype=np.complex128)
            for (u, v) in edge_data:
                mu = 1 << u
                mv = 1 << v
                # Vectorized: for each basis state, determine zz
                basis_arr = np.arange(dim)
                bits_u = (basis_arr >> u) & 1
                bits_v = (basis_arr >> v) & 1
                zz = np.where(bits_u == bits_v, 1, -1)
                phase *= np.exp(1j * gamma * 0.5 * zz)
            psi *= phase
            
            # Mixing: product of exp(-i beta X_i)
            # This is a tensor product, so we can apply each X_i rotation
            c = np.cos(beta)
            s = np.sin(beta)
            for i in range(n):
                mi = 1 << i
                # Apply 2x2 rotation on qubit i
                # For each pair of states differing only in bit i:
                for b in range(dim):
                    if (b & mi) == 0:
                        p_idx = b | mi
                        a = psi[b]
                        b2 = psi[p_idx]
                        psi[b] = c * a - 1j * s * b2
                        psi[p_idx] = -1j * s * a + c * b2
        
        # Expected cut
        probs = np.abs(psi) ** 2
        expected_cut = 0.0
        for (u, v) in edge_data:
            basis_arr = np.arange(dim)
            bits_u = (basis_arr >> u) & 1
            bits_v = (basis_arr >> v) & 1
            zz = np.where(bits_u == bits_v, 1, -1)
            zz_exp = np.dot(probs, zz)
            expected_cut += (1.0 - zz_exp) * 0.5
        
        return -expected_cut
    
    best_val = float('inf')
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    np.random.seed(123)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        x0 = np.concatenate([
            np.random.uniform(0, np.pi, p),
            np.random.uniform(0, np.pi, p)
        ])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Structured starts
    structured = [
        [0.35, 0.25, 0.35, 0.25],
        [0.5, 0.2, 0.3, 0.15],
        [0.4, 0.3, 0.3, 0.2],
        [0.6, 0.25, 0.4, 0.2],
    ]
    for x0 in structured:
        if time.time() > deadline - 10:
            break
        x0 = np.array(x0, dtype=np.float64)
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Compute optimal QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of (u, v) tuples
        p: number of QAOA layers
    
    Returns:
        numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check cache
    key = (n, tuple(sorted(edges)), p)
    if key in _angle_cache:
        return _angle_cache[key].copy()
    
    # Deadline guard: must finish well within 120s total
    start = time.time()
    remaining = 120.0 - (time.time() - start) - 5.0  # 5s safety margin
    if remaining < 5:
        # Not enough time, return default
        result = np.tile([0.35, 0.25], (p, 1))
        _angle_cache[key] = result
        return result
    
    # For very small graphs, we can do more thorough optimization
    try:
        result = _optimize_angles_v2(n, edges, p, max_time=remaining)
    except Exception:
        # Fallback to default
        result = np.tile([0.35, 0.25], (p, 1))
    
    # Ensure valid output
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (p, 2):
        result = np.tile([0.35, 0.25], (p, 1))
    
    _angle_cache[key] = result
    return result
# EVOLVE-BLOCK-END
