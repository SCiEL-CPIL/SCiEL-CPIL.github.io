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


def _expected_cut_value(n, adj, gammas, betas):
    """
    Compute the expected cut value for a given set of QAOA angles.
    
    The expected cut value is:
    sum_{(i,j) in E} <s_i (1 - s_j)/2 + (1 - s_i) s_j / 2>
    = sum_{(i,j) in E} (1/2) * (1 - <s_i s_j>)
    
    where <s_i s_j> is the expectation of the product of the two spin variables.
    
    For QAOA with p layers:
    U = prod_{l=1}^{p} exp(-i beta_l B) exp(-i gamma_l C)
    where C = sum_{(i,j) in E} (1 - Z_i Z_j)/2 = |E|/2 - 1/2 sum_{(i,j) in E} Z_i Z_j
    and B = sum_i X_i
    
    We compute the state evolution exactly for small n.
    """
    # Initial state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    # We work with the density matrix or state vector.
    # For n up to 10, state vector has 2^10 = 1024 entries, which is manageable.
    
    dim = 2 ** n
    # Initial state: uniform superposition
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute the cost Hamiltonian matrix C and mixing Hamiltonian B
    # C = |E|/2 * I - 1/2 * sum_{(i,j) in E} Z_i Z_j
    # B = sum_i X_i
    
    # Build Z_i Z_j terms
    # Z_i is diagonal with +1 for bit i = 0, -1 for bit i = 1
    # Z_i Z_j is diagonal with (+1) if bits i and j are same, (-1) if different
    
    # For efficiency, let's compute the action of exp(-i gamma C) and exp(-i beta B)
    
    # C is diagonal in the computational basis
    # C|x> = c(x) |x> where c(x) = |E|/2 - 1/2 * sum_{(i,j) in E} z_i(x) z_j(x)
    # where z_i(x) = 1 if bit i is 0, -1 if bit i is 1
    
    # Precompute c(x) for all x
    c_vals = np.zeros(dim, dtype=float)
    for x in range(dim):
        cut_val = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    # Bit i of x: (x >> i) & 1
                    bi = (x >> i) & 1
                    bj = (x >> j) & 1
                    # z_i = 1 if bi=0, -1 if bi=1
                    zi = 1 - 2*bi
                    zj = 1 - 2*bj
                    cut_val += zi * zj
        c_vals[x] = cut_val
    
    # Actually, let me reconsider. The cost Hamiltonian for MaxCut is:
    # C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
    # The eigenvalue for state |x> is:
    # c(x) = sum_{(i,j) in E} (1 - z_i z_j) / 2
    # where z_i = 1 if bit i = 0, -1 if bit i = 1
    # This equals the number of edges cut by the partition defined by x.
    
    # Let me recompute c_vals correctly
    c_vals = np.zeros(dim, dtype=float)
    for x in range(dim):
        cut_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    bi = (x >> i) & 1
                    bj = (x >> j) & 1
                    if bi != bj:
                        cut_count += 1
        c_vals[x] = cut_count
    
    # B = sum_i X_i
    # X_i flips bit i
    # exp(-i beta B) = exp(-i beta sum_i X_i)
    # This is a product of single-qubit rotations: exp(-i beta/2 X_i) for each qubit
    # exp(-i theta X/2) = cos(theta/2) I - i sin(theta/2) X
    
    # So exp(-i beta B) = tensor product over i of [cos(beta/2) I - i sin(beta/2) X_i]
    
    # Let's define the single-qubit rotation matrix
    # R_x(beta) = [[cos(beta/2), -i sin(beta/2)], [-i sin(beta/2), cos(beta/2)]]
    
    def apply_rx(psi, beta, qubit):
        """Apply exp(-i beta X_qubit / 2) to state vector psi."""
        # psi is indexed by computational basis states
        # We need to flip bit qubit and add amplitude
        result = np.zeros_like(psi)
        c = np.cos(beta / 2)
        s = np.sin(beta / 2) * (-1j)
        for x in range(dim):
            # Original amplitude at x
            amp = psi[x]
            # Bit qubit of x
            if (x >> qubit) & 1 == 0:
                # Stays at x with factor c
                result[x] += c * amp
                # Flips to x with bit qubit set, with factor s
                x_flip = x | (1 << qubit)
                result[x_flip] += s * amp
            else:
                # Stays at x with factor c
                result[x] += c * amp
                # Flips to x with bit qubit cleared, with factor s
                x_flip = x & ~(1 << qubit)
                result[x_flip] += s * amp
        return result
    
    # Apply the QAOA circuit: for l in range(p): apply exp(-i gamma_l C), then exp(-i beta_l B)
    # Order: first layer applies exp(-i gamma_1 C) then exp(-i beta_1 B), etc.
    # Wait, the task says: "applies exp(-i gamma C), then exp(-i beta sum X), for each row"
    # So for each layer l: first apply cost, then apply mixing.
    
    for l in range(p):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i gamma C): diagonal operation
        # psi[x] *= exp(-i gamma c_vals[x])
        psi = psi * np.exp(-1j * gamma * c_vals)
        
        # Apply exp(-i beta B) = product of single-qubit X rotations
        for q in range(n):
            psi = apply_rx(psi, beta, q)
        
        # Normalize to prevent numerical drift
        norm = np.sqrt(np.sum(np.abs(psi) ** 2))
        if norm > 0:
            psi = psi / norm
    
    # Compute expected cut value
    # E[cut] = sum_{(i,j) in E} P(bit i != bit j)
    # P(bit i != bit j) = 1 - Re(<psi| Z_i Z_j |psi>) / 2 ... no wait
    # P(bit i = 0, bit j = 0) + P(bit i = 1, bit j = 1) = (1 + <Z_i Z_j>)/2
    # P(bit i != bit j) = (1 - <Z_i Z_j>)/2
    # <Z_i Z_j> = sum_x |psi[x]|^2 * z_i(x) * z_j(x)
    
    probs = np.abs(psi) ** 2
    
    expected_cut = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                zz = 0.0
                for x in range(dim):
                    bi = (x >> i) & 1
                    bj = (x >> j) & 1
                    zi = 1 - 2*bi
                    zj = 1 - 2*bj
                    zz += probs[x] * zi * zj
                expected_cut += (1 - zz) / 2
    
    return expected_cut


def _exact_maxcut(n, adj):
    """Compute the exact maximum cut value by exhaustive search."""
    dim = 2 ** n
    best_cut = 0
    for x in range(dim):
        cut_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    bi = (x >> i) & 1
                    bj = (x >> j) & 1
                    if bi != bj:
                        cut_count += 1
        if cut_count > best_cut:
            best_cut = cut_count
    return best_cut


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of (u, v) tuples
        p: number of layers (should be 2)
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    adj = _build_adjacency(n, edges)
    
    # Compute exact MaxCut for this graph
    opt_cut = _exact_maxcut(n, adj)
    if opt_cut == 0:
        # No edges, any angles work
        return np.zeros((p, 2))
    
    # Search over gamma and beta for p=2 layers
    # We'll do a grid search followed by local refinement
    # gamma in [0, pi], beta in [0, pi]
    
    # Coarse grid search
    best_ratio = -1
    best_params = (0.35, 0.25, 0.35, 0.25)  # gamma1, beta1, gamma2, beta2
    
    # Use a moderate grid for speed
    n_grid = 12
    gamma_vals = np.linspace(0.05, np.pi - 0.05, n_grid)
    beta_vals = np.linspace(0.05, np.pi - 0.05, n_grid)
    
    # For speed, we'll evaluate on a coarser grid first
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    gammas = np.array([g1, g2])
                    betas = np.array([b1, b2])
                    exp_cut = _expected_cut_value(n, adj, gammas, betas)
                    ratio = exp_cut / opt_cut
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_params = (g1, b1, g2, b2)
    
    # Local refinement around the best point found
    g1, b1, g2, b2 = best_params
    
    # Use scipy optimize if available, otherwise do a finer grid
    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        
        def objective(params):
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            exp_cut = _expected_cut_value(n, adj, gammas, betas)
            return -exp_cut / opt_cut
        
        # Use differential evolution for global optimization
        bounds = [(0.01, np.pi - 0.01)] * 4
        result = differential_evolution(objective, bounds, maxiter=50, tol=1e-4, seed=42, polish=True)
        
        if result.success or -result.fun > best_ratio:
            best_params = tuple(result.x)
            best_ratio = -result.fun
        
        # Further local refinement
        result2 = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if result2.success or -result2.fun > best_ratio:
            best_params = tuple(result2.x)
            best_ratio = -result2.fun
            
    except ImportError:
        # Finer grid search around the best point
        step = 0.05
        for _ in range(3):
            improved = False
            for delta_g1 in np.linspace(-step, step, 5):
                for delta_b1 in np.linspace(-step, step, 5):
                    for delta_g2 in np.linspace(-step, step, 5):
                        for delta_b2 in np.linspace(-step, step, 5):
                            ng1 = np.clip(g1 + delta_g1, 0.01, np.pi - 0.01)
                            nb1 = np.clip(b1 + delta_b1, 0.01, np.pi - 0.01)
                            ng2 = np.clip(g2 + delta_g2, 0.01, np.pi - 0.01)
                            nb2 = np.clip(b2 + delta_b2, 0.01, np.pi - 0.01)
                            gammas = np.array([ng1, ng2])
                            betas = np.array([nb1, nb2])
                            exp_cut = _expected_cut_value(n, adj, gammas, betas)
                            ratio = exp_cut / opt_cut
                            if ratio > best_ratio:
                                best_ratio = ratio
                                g1, b1, g2, b2 = ng1, nb1, ng2, nb2
                                improved = True
            step *= 0.5
            if not improved:
                break
    
    g1, b1, g2, b2 = best_params
    
    # Build the output array: shape (p, 2), rows are (gamma, beta)
    result = np.array([[g1, b1], [g2, b2]], dtype=float)
    
    return result
# EVOLVE-BLOCK-END
