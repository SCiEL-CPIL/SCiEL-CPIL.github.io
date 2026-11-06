"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

# Deadline guard: kill after 140s wall-clock (10s safety margin before 150s limit)
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _exhaustive_maxcut(n, edges):
    """Find the exact maximum cut value for a graph with n nodes."""
    if not edges:
        return 0
    
    # Build adjacency list for faster checking
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_cut = 0
    # For small n (<=12), brute force all 2^(n-1) assignments (fix node 0 to 0)
    for mask in range(1 << (n - 1)):
        # Node 0 is always in partition 0
        assignment = [0] + [int((mask >> i) & 1) for i in range(n - 1)]
        cut = 0
        for u, v in edges:
            if assignment[u] != assignment[v]:
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return best_cut

def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for a QAOA circuit with given angles.
    C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    H_C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    H_B = sum_i X_i
    
    The circuit: |+>^n -> exp(-i gamma_1 H_C) -> exp(-i beta_1 H_B) -> ... -> exp(-i gamma_p H_C) -> exp(-i beta_p H_B)
    
    Expected cut = <psi| H_C |psi>
    
    For each edge (u,v), contribution is (1 - <Z_u Z_v>)/2
    So expected_cut = |E|/2 - (1/2) * sum_{(u,v) in E} <Z_u Z_v>
    """
    if not edges:
        return 0.0
    
    # We need to compute <Z_u Z_v> for each edge (u,v)
    # This requires simulating the QAOA circuit.
    
    # For small n, we can do exact state vector simulation.
    # State is 2^n complex amplitudes.
    
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Pauli operators as sparse-like operations
    # Z operator: flip sign of amplitudes where bit is 1
    # X operator: swap amplitudes where bit is 0 <-> bit is 1
    
    def apply_Z(state, qubit):
        """Apply Z to qubit: multiply by -1 where qubit bit is 1."""
        mask = 1 << qubit
        # For each index, if bit is set, negate
        # Vectorized: 
        idx = np.arange(dim)
        sign = np.where((idx & mask) != 0, -1.0, 1.0)
        return state * sign
    
    def apply_X(state, qubit):
        """Apply X to qubit: swap amplitudes where qubit bit differs."""
        mask = 1 << qubit
        result = state.copy()
        # Swap: for indices where bit is 0, swap with index ^ mask
        # Vectorized approach:
        idx = np.arange(dim)
        # For all i, result[i] = state[i ^ mask] if we want to swap
        # Actually X|0> = |1>, X|1> = |0>
        # So (X|psi>)_i = psi_{i^mask}
        result[:] = state[idx ^ mask]
        return result
    
    # H_C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # exp(-i gamma H_C) = product over edges of exp(-i gamma (1 - Z_u Z_v)/2)
    # But these don't commute in general! Wait, Z_u Z_v for different edges DO commute
    # because they are all diagonal in the computational basis.
    # So H_C is diagonal, and exp(-i gamma H_C) is just phase multiplication.
    
    # For each basis state |x>, the eigenvalue of H_C is:
    # sum_{(u,v) in E} (1 - (-1)^{x_u + x_v})/2
    # = number of edges where x_u != x_v
    # = cut value of partition x
    
    # So exp(-i gamma H_C)|x> = exp(-i gamma * cut(x)) |x>
    
    # Let's precompute cut values for all 2^n states
    cut_values = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        c = 0
        for u, v in edges:
            xu = (x >> u) & 1
            xv = (x >> v) & 1
            if xu != xv:
                c += 1
        cut_values[x] = c
    
    # Apply each layer
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma H_C): phase each basis state
        phases = np.exp(-1j * gamma * cut_values)
        state = state * phases
        
        # Apply exp(-i beta H_B) = product_i exp(-i beta X_i)
        # Since X_i commute, this is product of exp(-i beta X_i)
        # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
        # So for each qubit, apply: state = cos(beta)*state - i*sin(beta)*X(state)
        
        for q in range(n):
            x_state = apply_X(state, q)
            state = np.cos(beta) * state - 1j * np.sin(beta) * x_state
    
    # Now compute <H_C> = sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_x (-1)^{x_u + x_v} |psi_x|^2
    probs = np.abs(state) ** 2
    
    # For each edge, compute <Z_u Z_v>
    zuzv_sum = 0.0
    for u, v in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        idx = np.arange(dim)
        # (-1)^(x_u + x_v) = +1 if x_u == x_v, -1 if different
        xu = (idx & mask_u) != 0
        xv = (idx & mask_v) != 0
        zuzv = np.where(xu == xv, 1.0, -1.0)
        zuzv_sum += np.dot(probs, zuzv)
    
    expected_cut = len(edges) / 2.0 - zuzv_sum / 2.0
    return expected_cut

def _optimize_angles(n, edges, p, max_time=60.0):
    """Optimize QAOA angles using gradient-free optimization."""
    if not edges:
        return np.zeros((p, 2))
    
    # Initial guess
    best_angles = np.tile([0.35, 0.25], (p, 1))
    
    # Compute the optimum
    opt_cut = _exhaustive_maxcut(n, edges)
    if opt_cut == 0:
        return best_angles
    
    # Try a grid search first for small n
    # Then refine with Nelder-Mead or similar
    
    # Use scipy if available for optimization
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        # Constrain to reasonable ranges
        gammas = np.clip(gammas, 0.0, np.pi)
        betas = np.clip(betas, 0.0, np.pi)
        ec = _expected_cut(n, edges, gammas, betas)
        return -ec  # minimize negative expected cut
    
    # Initial parameters
    x0 = np.concatenate([np.full(p, 0.35), np.full(p, 0.25)])
    
    if minimize is not None and _time_left() > 30:
        # Try Nelder-Mead
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
            if result.fun < 0:  # valid result
                best_params = np.concatenate([np.clip(result.x[:p], 0, np.pi), 
                                              np.clip(result.x[p:], 0, np.pi)])
                best_angles = np.column_stack([best_params[:p], best_params[p:]])
        except Exception:
            pass
    
    # Also try a few random restarts if time permits
    if _time_left() > 20:
        rng = np.random.RandomState(42)
        current_best_ec = _expected_cut(n, edges, best_angles[:, 0], best_angles[:, 1])
        
        for restart in range(10):
            if _time_left() < 10:
                break
            x0_r = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)])
            if minimize is not None:
                try:
                    result = minimize(objective, x0_r, method='Nelder-Mead',
                                    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
                    if result.fun < 0:
                        params = np.concatenate([np.clip(result.x[:p], 0, np.pi), 
                                                  np.clip(result.x[p:], 0, np.pi)])
                        ec = -result.fun
                        if ec > current_best_ec:
                            current_best_ec = ec
                            best_angles = np.column_stack([params[:p], params[p:]])
                except Exception:
                    pass
    
    return best_angles

# Cache for optimized angles
_angles_cache = {}

def angles(n, edges, p):
    """
    Return QAOA angles for the given graph.
    
    Parameters:
    n: number of nodes
    edges: list of (u, v) edges
    p: number of QAOA layers
    
    Returns:
    numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check deadline
    if _elapsed() > _DEADLINE - 5:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Create cache key
    edge_key = tuple(sorted(edges))
    cache_key = (n, edge_key, p)
    
    if cache_key in _angles_cache:
        return _angles_cache[cache_key]
    
    # Optimize angles
    result = _optimize_angles(n, edges, p)
    
    _angles_cache[cache_key] = result
    return result

# EVOLVE-BLOCK-END
