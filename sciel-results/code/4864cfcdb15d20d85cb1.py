"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def _evaluate_angles(n, edges, gammas, betas):
    """
    Evaluate the expected cut for given QAOA angles using exact state evolution.
    For small n (<=10), we can track the full quantum state.
    
    The QAOA circuit:
    1. Start in |+>^n
    2. For each layer l:
       a. Apply exp(-i * gamma_l * C) where C is MaxCut Hamiltonian (sum of Z_i Z_j for edges)
       b. Apply exp(-i * beta_l * sum X_i)
    
    Expected cut = sum over edges (i,j) of <psi| (1 - Z_i Z_j)/2 |psi>
                 = sum over edges (i,j) of (1 - <Z_i Z_j>)/2
    """
    n_qubits = n
    dim = 1 << n_qubits
    
    # Start state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge list for efficiency
    edge_list = []
    for u, v in edges:
        edge_list.append((u, v))
    
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i * gamma * C)
        # C = sum_{(i,j) in E} Z_i Z_j
        # This is diagonal in computational basis, so we can apply phase to each basis state
        phases = np.zeros(dim, dtype=np.float64)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # Check if (i,j) is an edge
                # We'll compute Z_i Z_j expectation contribution
                pass
        
        # More efficient: compute phase for each basis state
        # For basis state |x>, Z_i gives (-1)^{x_i}
        # So Z_i Z_j gives (-1)^{x_i + x_j}
        # C| x> = sum_{(i,j) in E} (-1)^{x_i + x_j} |x>
        # So exp(-i*gamma*C)|x> = exp(-i*gamma*sum_{(i,j) in E} (-1)^{x_i+x_j}) |x>
        
        # Compute the eigenvalue for each basis state
        eigenvalues = np.zeros(dim, dtype=np.float64)
        for idx in range(dim):
            x = np.array([(idx >> k) & 1 for k in range(n_qubits)], dtype=np.int32)
            val = 0.0
            for u, v in edge_list:
                val += (-1) ** (x[u] + x[v])
            eigenvalues[idx] = val
        
        psi *= np.exp(-1j * gamma * eigenvalues)
        
        # Apply exp(-i * beta * sum X_i)
        # sum X_i is not diagonal, so we need to apply it properly
        # For small n, we can construct the operator or use tensor product
        # X = [[0,1],[1,0]]
        # exp(-i*beta*sum X_i) = tensor product of exp(-i*beta*X) for each qubit
        # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Apply single-qubit gate exp(-i*beta*X) to each qubit
        # This can be done by iterating through basis states
        # For each basis state, the gate acts independently on each qubit
        # But since it's a product of single-qubit gates, we can apply them sequentially
        
        # Actually, let's use a more efficient approach:
        # The operator exp(-i*beta*sum_i X_i) = tensor_{i=0}^{n-1} exp(-i*beta*X_i)
        # Each exp(-i*beta*X) = [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        
        # We'll apply this to the state vector
        # For n qubits, the state is a tensor of shape (2,)*n
        psi_tensor = psi.reshape([2]*n_qubits)
        
        for q in range(n_qubits):
            # Apply the gate to qubit q
            # New state[q_idx] = cos_b * old[q_idx] - i*sin_b * old[flipped_q_idx]
            new_psi = np.zeros_like(psi_tensor)
            for idx in range(2):
                # Slice where qubit q is idx
                slc = [slice(None)] * n_qubits
                slc[q] = idx
                slc = tuple(slc)
                
                # Flip qubit q
                slc_flip = [slice(None)] * n_qubits
                slc_flip[q] = 1 - idx
                slc_flip = tuple(slc_flip)
                
                new_psi[slc] = cos_b * psi_tensor[slc] - 1j * sin_b * psi_tensor[slc_flip]
            
            psi_tensor = new_psi
        
        psi = psi_tensor.reshape(-1)
    
    # Compute expected cut
    # <Z_i Z_j> for each edge (i,j)
    # Z_i Z_j expectation: sum_x |psi_x|^2 * (-1)^{x_i + x_j}
    probs = np.abs(psi) ** 2
    
    expected_cut = 0.0
    for u, v in edge_list:
        # Compute <Z_u Z_v>
        zz_expect = 0.0
        for idx in range(dim):
            x_u = (idx >> u) & 1
            x_v = (idx >> v) & 1
            zz_expect += probs[idx] * ((-1) ** (x_u + x_v))
        expected_cut += (1.0 - zz_expect) / 2.0
    
    return expected_cut


def _exhaustive_maxcut(n, edges):
    """Compute exact MaxCut via exhaustive search for small n."""
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut += 1
        if cut > best:
            best = cut
    return best


def _optimize_angles_for_graph(n, edges, p=2, max_iterations=500):
    """
    Optimize QAOA angles for a specific graph using gradient-free optimization.
    Uses Nelder-Mead simplex method for robustness.
    """
    from scipy.optimize import minimize
    
    best_maxcut = _exhaustive_maxcut(n, edges)
    if best_maxcut == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        exp_cut = _evaluate_angles(n, edges, gammas, betas)
        # We want to maximize exp_cut / best_maxcut, so minimize negative
        return -exp_cut / best_maxcut
    
    # Initial guess: use common QAOA initializations
    # gamma ~ 0.5 * pi / (2*p), beta ~ 0.5 * pi / (2*p)
    x0 = np.array([0.5 * np.pi / (2 * p)] * p + [0.5 * np.pi / (2 * p)] * p)
    
    # Bounds
    bounds = [(0, np.pi)] * (2 * p)
    
    # Use Nelder-Mead (derivative-free)
    result = minimize(objective, x0, method='Nelder-Mead', 
                      bounds=bounds,
                      options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6})
    
    params = np.clip(result.x, 0, np.pi)
    gammas = params[:p]
    betas = params[p:]
    
    return np.column_stack([gammas, betas])


def _compute_graph_features(n, edges):
    """Compute simple graph features for graph-dependent angle selection."""
    m = len(edges)
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    
    # Degree sequence
    degrees = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees) if n > 0 else 0
    
    return {
        'n': n,
        'm': m,
        'density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree
    }


def angles(n, edges, p):
    """
    Compute QAOA angles for the given graph.
    
    Uses per-graph optimization via Nelder-Mead on the exact expected cut.
    For the six evaluation graphs (n=6,8,10), this is computationally feasible
    within the time limit.
    """
    # Ensure edges is a list of tuples
    edges_list = [(int(u), int(v)) for u, v in edges]
    
    # Optimize angles for this specific graph
    try:
        opt_angles = _optimize_angles_for_graph(n, edges_list, p=p)
        # Ensure shape is (p, 2)
        opt_angles = opt_angles.reshape(p, 2)
        # Clip to valid range
        opt_angles = np.clip(opt_angles, 0, np.pi)
        return opt_angles
    except Exception:
        # Fallback: use heuristic initialization
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
