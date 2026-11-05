"""QAOA parameter design with multi-restart Nelder-Mead + gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for precomputed data per graph
_graph_cache = {}


def _build_graph(n, edges):
    """Build adjacency matrix and cut value arrays."""
    key = (n, tuple(sorted(edges)))
    if key in _graph_cache:
        return _graph_cache[key]
    
    # Build adjacency
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Precompute cut values for all 2^n assignments
    # cut(z) = sum_{(u,v) in E} z_u * (1 - z_v) where z in {0,1}^n
    # Equivalently, cut = (n_edges - sum_{(u,v)} (1 - 2*z_u)*(1 - 2*z_v)) / 2
    # Let s_u = 2*z_u - 1 in {-1, 1}
    # cut = (m - sum_{(u,v)} s_u * s_v) / 2
    
    m = len(edges)
    # All possible sign vectors
    signs = np.zeros((2**n, n), dtype=np.int8)
    for i in range(n):
        signs[:, i] = (np.arange(2**n) >> i) & 1
    signs = 2 * signs - 1  # now in {-1, 1}
    
    # Compute sum of s_u * s_v for each edge
    edge_sum = np.zeros(2**n, dtype=np.float64)
    for u, v in edges:
        edge_sum += signs[:, u] * signs[:, v]
    
    cut_vals = (m - edge_sum) / 2.0  # expected cut for each assignment
    optimal_cut = cut_vals.max()
    
    # Also precompute the Hamiltonian matrix elements for fast QAOA evaluation
    # <z|C|z'> = cut(z) if z==z', 0 otherwise (C is diagonal in computational basis)
    # Actually C = sum_{(u,v)} Z_u Z_v / 2 ... let's be careful.
    
    # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    # <z|C|z> = sum_{(u,v)} (1 - s_u s_v)/2 = cut(z)
    # <z|C|z'> = 0 if z != z' since C is diagonal
    
    # For the mixer: H_M = sum_i X_i
    # X|0> = |1>, X|1> = |0>
    # So the mixer flips each bit independently.
    
    result = {
        'adj': adj,
        'cut_vals': cut_vals,
        'optimal_cut': optimal_cut,
        'm': m,
        'signs': signs,
        'edge_sum': edge_sum,
        'n': n
    }
    _graph_cache[key] = result
    return result


def _qaoa_expectation(n, edges, gammas, betas, cache=None):
    """Compute expected cut for given QAOA parameters."""
    if cache is None:
        cache = _build_graph(n, edges)
    
    cut_vals = cache['cut_vals']
    n = cache['n']
    m = cache['m']
    
    # Start state: |+>^n, so |psi> = (1/sqrt(2^n)) * sum_z |z>
    # Amplitudes are all equal: psi[z] = 1/sqrt(2^n)
    
    # Apply layers: exp(-i*gamma_p * C) then exp(-i*beta_p * sum X)
    # C is diagonal, so exp(-i*gamma*C) just multiplies each |z> by exp(-i*gamma*cut(z))
    # sum X flips each bit, so it's a permutation of basis states.
    
    # For n <= 10, 2^n <= 1024, so we can work with the full state vector.
    dim = 2**n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Phase kickback from C (diagonal)
        psi *= np.exp(-1j * gamma * cut_vals)
        
        # Apply exp(-i*beta * sum_i X_i)
        # This is a product of single-qubit rotations: exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        # For each qubit i, it maps |z> to cos(beta)*|z> - i*sin(beta)*|z with bit i flipped>
        # We can apply this efficiently by iterating over qubits.
        
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        for i in range(n):
            # For each state, mix with the state where bit i is flipped
            # We need to be careful: this is a simultaneous operation on all states
            # Let's use the structure: the operator is tensor product of (cos(beta)*I - i*sin(beta)*X) on each qubit
            
            # Actually, let's do it differently. The full operator is:
            # U = tensor_{i=0}^{n-1} (cos(beta)*I - i*sin(beta)*X_i)
            # This can be applied as: for each qubit i, apply the 2x2 rotation.
            
            # Reshape to apply on qubit i
            # Qubit i corresponds to bit i in our encoding
            shape = [2] * n
            psi_reshaped = psi.reshape(shape)
            
            # Move axis i to front
            psi_reshaped = np.moveaxis(psi_reshaped, i, 0)
            
            # Apply the 2x2 matrix [[cos, -i*sin], [-i*sin, cos]]
            new_psi = cb * psi_reshaped[0] - 1j * sb * psi_reshaped[1]
            new_psi2 = -1j * sb * psi_reshaped[0] + cb * psi_reshaped[1]
            
            # Reconstruct
            psi_reshaped[0] = new_psi
            psi_reshaped[1] = new_psi2
            
            # Move axis back
            psi_reshaped = np.moveaxis(psi_reshaped, 0, i)
            psi = psi_reshaped.reshape(dim)
    
    # Expected cut = sum_z |psi[z]|^2 * cut(z)
    probs = np.abs(psi)**2
    expected_cut = np.dot(probs, cut_vals)
    return expected_cut


def _qaoa_expectation_fast(n, edges, params, cache=None):
    """Fast version taking params as (gamma1, beta1, gamma2, beta2)."""
    if cache is None:
        cache = _build_graph(n, edges)
    
    gammas = params[::2]
    betas = params[1::2]
    return _qaoa_expectation(n, edges, gammas, betas, cache)


def _optimize_angles(n, edges, p, time_limit=110.0):
    """Optimize QAOA angles using multi-restart Nelder-Mead with gradient polish."""
    start_time = time.time()
    cache = _build_graph(n, edges)
    optimal_cut = cache['optimal_cut']
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        try:
            ec = _qaoa_expectation_fast(n, edges, params, cache)
            return -ec / optimal_cut
        except:
            return 0.0
    
    best_val = -1.0
    best_params = None
    
    # Multiple restarts with different initial guesses
    n_params = 2 * p
    rng = np.random.RandomState(42)
    
    n_restarts = 30
    max_iters = 500
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            x0 = np.array([0.4, 0.2, 0.3, 0.15])
        elif restart == 1:
            x0 = np.array([0.5, 0.3, 0.4, 0.2])
        elif restart == 2:
            x0 = np.array([0.3, 0.1, 0.5, 0.3])
        elif restart == 3:
            x0 = np.array([0.6, 0.4, 0.3, 0.2])
        elif restart == 4:
            x0 = np.array([0.2, 0.15, 0.4, 0.25])
        else:
            # Random initialization in reasonable range
            x0 = rng.uniform(0.05, 0.8, size=n_params)
            # Ensure gammas and betas are positive
            x0 = np.abs(x0)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
        
        # Also try a local refinement with smaller initial simplex
        if time.time() - start_time > time_limit - 10:
            break
    
    # Gradient-based polish on best solution
    if best_params is not None and time.time() - start_time < time_limit - 5:
        try:
            def obj_grad(params):
                if time.time() - start_time > time_limit - 5:
                    return 0.0, np.zeros_like(params)
                ec = _qaoa_expectation_fast(n, edges, params, cache)
                val = -ec / optimal_cut
                # Numerical gradient
                eps = 1e-7
                grad = np.zeros_like(params)
                for i in range(len(params)):
                    params_plus = params.copy()
                    params_plus[i] += eps
                    params_minus = params.copy()
                    params_minus[i] -= eps
                    f_plus = _qaoa_expectation_fast(n, edges, params_plus, cache) / optimal_cut
                    f_minus = _qaoa_expectation_fast(n, edges, params_minus, cache) / optimal_cut
                    grad[i] = -(f_plus - f_minus) / (2 * eps)
                return val, grad
            
            res2 = minimize(obj_grad, best_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except:
            pass
    
    if best_params is None:
        # Fallback
        best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Convert back to (p, 2) array: rows are (gamma, beta)
    result = np.zeros((p, 2))
    for i in range(p):
        result[i, 0] = best_params[2*i]
        result[i, 1] = best_params[2*i+1]
    
    return result


def angles(n, edges, p):
    """Main entry point: return optimized QAOA angles."""
    start_time = time.time()
    time_limit = 110.0  # Leave safety margin before 120s limit
    
    try:
        result = _optimize_angles(n, edges, p, time_limit)
        return result
    except Exception:
        # Fallback to reasonable defaults
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
