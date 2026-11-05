"""QAOA parameter design: graph-dependent angle optimization via simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for computed results to avoid recomputation
_angle_cache = {}

def _build_graph(n, edges):
    """Build adjacency matrix and edge list from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    edge_list = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u < v:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            edge_list.append((u, v))
    return adj, edge_list

def _exact_maxcut(n, edge_list):
    """Compute exact maximum cut by brute force for small n."""
    if not edge_list:
        return 0
    best = 0
    m = len(edge_list)
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edge_list:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _expected_cut(n, edge_list, gammas, betas, adj):
    """
    Compute the expected cut for a 2-layer QAOA circuit.
    
    For each edge (u,v), the probability that u and v are in different
    partitions after the QAOA circuit is applied.
    
    We use the fact that for the initial |+>^n state and the specific
    QAOA structure, we can compute the expected cut using the
    correlation functions.
    
    For p=2 QAOA:
    - Start with |+>^n
    - Apply exp(-i*gamma1 * C) where C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    - Apply exp(-i*beta1 * sum X_i)
    - Apply exp(-i*gamma2 * C)
    - Apply exp(-i*beta2 * sum X_i)
    
    The expected cut = sum_{(u,v) in E} P[u != v]
    where P[u != v] = (1 - <Z_u Z_v>)/2
    """
    # For efficiency with small n, we can use the exact state vector approach
    # or use the known analytical results.
    
    # For n <= 10, we can use the exact state vector approach
    # State is a vector of size 2^n
    dim = 1 << n
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * [1, 1, ..., 1]
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply layers
    for layer in range(2):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i * gamma * C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
        # For each edge, Z_u Z_v is diagonal with entries +/-1
        # So exp(-i*gamma*C) is diagonal with entries exp(-i*gamma * c_s)
        # where c_s is the cut value of spin configuration s
        
        # Compute the cut value for each basis state
        # For state s (bitstring), Z_u = (-1)^{s_u}, so Z_u Z_v = (-1)^{s_u + s_v}
        # (1 - Z_u Z_v)/2 = 1 if s_u != s_v, 0 otherwise
        # C_s = number of edges cut in configuration s
        
        # Build the diagonal of the cost unitary
        cost_diag = np.zeros(dim, dtype=np.complex128)
        for s in range(dim):
            cut_val = 0
            for (u, v) in edge_list:
                if ((s >> u) & 1) != ((s >> v) & 1):
                    cut_val += 1
            cost_diag[s] = np.exp(-1j * gamma * cut_val)
        
        state = cost_diag * state
        
        # Apply exp(-i * beta * sum X_i)
        # sum X_i = sum_i X_i
        # This is a tensor product of single-qubit rotations
        # X = [[0,1],[1,0]], exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        # For each qubit, the rotation matrix is [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        
        # We can apply this efficiently using the fact that it's a tensor product
        # The full operator is the tensor product of n single-qubit operators
        # Each single-qubit operator: R = [[c, -is], [-is, c]] where c=cos(beta), s=sin(beta)
        
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Apply the tensor product rotation
        # For each basis state, the new amplitude is computed by
        # applying the single-qubit rotations
        new_state = np.zeros(dim, dtype=np.complex128)
        for s_idx in range(dim):
            # Get the bitstring
            bits = [(s_idx >> i) & 1 for i in range(n)]
            
            # The rotation on each qubit:
            # If bit is 0: new_amp_0 = c * amp_0 - i*s * amp_1
            # If bit is 1: new_amp_1 = c * amp_1 - i*s * amp_0
            # But since it's a tensor product, we need to apply all simultaneously
            
            # Actually, the operator exp(-i*beta*sum X_i) applied to |s> gives:
            # sum over all t: <t| exp(-i*beta*sum X_i) |s>
            # = product over i of <t_i| exp(-i*beta*X_i) |s_i>
            # = product over i of [c if t_i == s_i, -i*s if t_i != s_i]
            
            # So <t| U |s> = product_i [c if t_i==s_i else -i*s]
            # = c^(n-h) * (-i*s)^h where h = Hamming distance between s and t
            
            # This means: new_state[t] = sum_s state[s] * c^(n-H(s,t)) * (-i*s)^H(s,t)
            
            # For efficiency, let's just iterate over all t
            # But that's O(4^n) which is fine for n<=10 (4^10 = 1M)
            pass
        
        # More efficient approach: use the fact that the operator is a tensor product
        # We can reshape the state into an n-dimensional tensor and apply the
        # single-qubit rotations along each axis
        state_tensor = state.reshape([2] * n)
        
        for qubit in range(n):
            # Apply the rotation along axis qubit
            # R = [[c, -i*s], [-i*s, c]]
            # new_state_tensor[..., 0, ...] = c * old[..., 0, ...] - i*s * old[..., 1, ...]
            # new_state_tensor[..., 1, ...] = -i*s * old[..., 0, ...] + c * old[..., 1, ...]
            
            # Create slices
            sl0 = [slice(None)] * n
            sl1 = [slice(None)] * n
            sl0[qubit] = 0
            sl1[qubit] = 1
            
            val0 = state_tensor[tuple(sl0)].copy()
            val1 = state_tensor[tuple(sl1)].copy()
            
            state_tensor[tuple(sl0)] = c * val0 - 1j * s * val1
            state_tensor[tuple(sl1)] = -1j * s * val0 + c * val1
        
        state = state_tensor.reshape(dim)
    
    # Compute expected cut: sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_s |state[s]|^2 * (-1)^{s_u + s_v}
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for (u, v) in edge_list:
        # Compute <Z_u Z_v>
        # (-1)^{s_u + s_v} = +1 if s_u == s_v, -1 if s_u != s_v
        zz = 0.0
        for s_idx in range(dim):
            su = (s_idx >> u) & 1
            sv = (s_idx >> v) & 1
            if su == sv:
                zz += probs[s_idx]
            else:
                zz -= probs[s_idx]
        expected_cut += (1.0 - zz) / 2.0
    
    return expected_cut

def _objective(params, n, edge_list, adj, exact_opt):
    """Objective function to minimize (negative of expected cut)."""
    gammas = params[:2]
    betas = params[2:4]
    ec = _expected_cut(n, edge_list, gammas, betas, adj)
    return -ec

def _optimize_angles(n, edges, p, time_budget=8.0):
    """
    Optimize QAOA angles for a given graph using multi-start optimization.
    """
    adj, edge_list = _build_graph(n, edges)
    exact_opt = _exact_maxcut(n, edge_list)
    
    if exact_opt == 0:
        return np.zeros((p, 2))
    
    best_params = None
    best_val = -np.inf
    
    start_time = time.time()
    
    # Multiple starting points
    n_starts = 8
    rng = np.random.RandomState(42)
    
    for start_idx in range(n_starts):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Generate initial guess
        if start_idx == 0:
            # Standard initial guess
            params0 = np.array([0.5, 0.5, 0.5, 0.5])
        elif start_idx == 1:
            params0 = np.array([0.3, 0.3, 0.4, 0.4])
        elif start_idx == 2:
            params0 = np.array([0.7, 0.7, 0.3, 0.3])
        else:
            params0 = rng.uniform(0.1, 1.5, size=4)
        
        # Use Nelder-Mead optimization
        try:
            res = minimize(
                _objective,
                params0,
                args=(n, edge_list, adj, exact_opt),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_params = np.array([0.5, 0.5, 0.5, 0.5])
        best_val = _expected_cut(n, edge_list, best_params[:2], best_params[2:], adj)
    
    # Format output: (p, 2) array with rows (gamma, beta)
    result = np.zeros((p, 2))
    for i in range(min(p, 2)):
        result[i, 0] = best_params[i]  # gamma
        result[i, 1] = best_params[2 + i]  # beta
    
    return result

def angles(n, edges, p):
    """
    Compute QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of edges
        p: number of QAOA layers
    
    Returns:
        (p, 2) array with rows (gamma, beta) in radians
    """
    # Create cache key
    edge_tuple = tuple(sorted((min(u, v), max(u, v)) for u, v in edges))
    cache_key = (n, edge_tuple, p)
    
    if cache_key in _angle_cache:
        return _angle_cache[cache_key]
    
    # Check time budget
    start_time = time.time()
    
    result = _optimize_angles(n, edges, p, time_budget=10.0)
    
    _angle_cache[cache_key] = result
    
    return result
# EVOLVE-BLOCK-END
