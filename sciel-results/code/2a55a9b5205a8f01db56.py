"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _build_qaoa_state(n, edges, gammas, betas):
    """
    Compute the state vector after applying the QAOA circuit.
    The circuit starts in |+>^n and applies:
      for k in range(p):
        exp(-i * gammas[k] * C)
        exp(-i * betas[k] * sum_j X_j)
    where C is the MaxCut cost operator (unweighted).
    
    Since n is small (6, 8, 10), we can work with the full state vector
    of dimension 2^n.
    """
    dim = 1 << n
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute the cost operator action.
    # C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    # exp(-i * gamma * C) = exp(-i * gamma/2 * sum (1 - Z_i Z_j))
    # = exp(-i * gamma/2 * |E|) * exp(-i * gamma/2 * sum (-Z_i Z_j))
    # Actually, let's be more careful.
    # C = sum_{(i,j) in edges} (I - Z_i Z_j) / 2
    # So exp(-i gamma C) = exp(-i gamma/2 * |E|) * prod_{(i,j)} exp(i gamma/2 * Z_i Z_j)
    # Z_i Z_j is diagonal with entries +1 or -1.
    # So exp(i gamma/2 * Z_i Z_j) is diagonal with entries exp(i gamma/2) or exp(-i gamma/2).
    
    # Precompute for each bitstring the value of Z_i Z_j for each edge.
    # For a bitstring b, Z_i = (-1)^{b_i}.
    # So Z_i Z_j = (-1)^{b_i + b_j}.
    
    # We can precompute a matrix M of shape (dim, len(edges)) where
    # M[b, e] = (-1)^{b_{i_e} + b_{j_e}}
    
    if len(edges) == 0:
        # No edges, C = 0, so exp(-i gamma C) = I
        # Just apply the beta rotations
        for k in range(len(gammas)):
            # exp(-i beta * sum X_j)
            # This is a product of single-qubit rotations: (cos(beta) I - i sin(beta) X)^{⊗n}
            # Actually, exp(-i beta sum X_j) = prod_j exp(-i beta X_j)
            # exp(-i beta X) = cos(beta) I - i sin(beta) X
            # = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
            # This is a single-qubit gate applied to each qubit.
            # We can apply it to the state vector.
            pass  # Handled below
    
    # Let's implement the unitary application more carefully.
    # For efficiency with small n, we'll apply the unitaries directly.
    
    # Helper: apply exp(-i gamma * C) to the state
    def apply_cost(state, gamma):
        if gamma == 0 or len(edges) == 0:
            return state.copy()
        
        # Compute the phase for each basis state
        # Phase = -i * gamma * C(b) where C(b) = sum_{(i,j) in edges} (1 - (-1)^{b_i + b_j}) / 2
        # = -i * gamma * sum_{(i,j)} (1 - (-1)^{b_i + b_j}) / 2
        
        # Let's compute C(b) for all b
        # C(b) = sum_{(i,j) in edges} (1 - (-1)^{b_i + b_j}) / 2
        # = |E|/2 - (1/2) * sum_{(i,j)} (-1)^{b_i + b_j}
        
        phases = np.zeros(dim, dtype=np.complex128)
        for b in range(dim):
            c_val = 0.0
            for (i, j) in edges:
                bi = (b >> i) & 1
                bj = (b >> j) & 1
                zprod = 1 if (bi == bj) else -1
                c_val += (1 - zprod) / 2.0
            phases[b] = np.exp(-1j * gamma * c_val)
        
        return state * phases
    
    # Helper: apply exp(-i beta * sum_j X_j) to the state
    # This is prod_j exp(-i beta X_j)
    # Each factor is a 2x2 matrix: [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    # We can apply this by iterating over qubits.
    def apply_x_rotations(state, beta):
        if beta == 0:
            return state.copy()
        
        c = np.cos(beta)
        s = np.sin(beta)
        # The single-qubit gate is [[c, -1j*s], [-1j*s, c]]
        # Apply to each qubit in the tensor product.
        # We reshape state to a tensor of shape (2,)*n
        shape = (2,) * n
        state_tensor = state.reshape(shape)
        
        result = state_tensor.copy()
        for q in range(n):
            # Apply the gate on qubit q
            # We need to swap axes to make qubit q the last axis
            # Move axis q to the end
            new_order = list(range(n))
            new_order.remove(q)
            new_order.append(q)
            rotated = np.transpose(result, new_order)
            # Now the last axis is qubit q
            # Reshape to (-1, 2)
            flat = rotated.reshape(-1, 2)
            # Apply the gate: [[c, -1j*s], [-1j*s, c]]
            new_flat = np.empty_like(flat)
            new_flat[:, 0] = c * flat[:, 0] - 1j * s * flat[:, 1]
            new_flat[:, 1] = -1j * s * flat[:, 0] + c * flat[:, 1]
            # Reshape back
            rotated = new_flat.reshape(rotated.shape)
            # Move axis back
            inv_order = [0] * n
            for idx, axis in enumerate(new_order):
                inv_order[axis] = idx
            result = np.transpose(rotated, inv_order)
        
        return result.reshape(-1)
    
    # Apply the QAOA circuit
    p = len(gammas)
    for k in range(p):
        state = apply_cost(state, gammas[k])
        state = apply_x_rotations(state, betas[k])
    
    return state


def _expected_cut(state, edges):
    """
    Compute the expected cut value: <state| C |state>
    where C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    """
    dim = len(state)
    n = int(np.log2(dim))
    
    expected = 0.0
    for (i, j) in edges:
        # Compute <Z_i Z_j>
        zz = 0.0
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> j) & 1
            zprod = 1 if (bi == bj) else -1
            prob = np.abs(state[b]) ** 2
            zz += zprod * prob
        # C_ij = (1 - Z_i Z_j) / 2, so <C_ij> = (1 - <Z_i Z_j>) / 2
        expected += (1 - zz) / 2.0
    
    return expected


def _exact_maxcut(n, edges):
    """
    Compute the exact maximum cut value by exhaustive search.
    """
    dim = 1 << n
    best = 0.0
    for b in range(dim):
        cut_val = 0.0
        for (i, j) in edges:
            bi = (b >> i) & 1
            bj = (b >> j) & 1
            if bi != bj:
                cut_val += 1.0
        if cut_val > best:
            best = cut_val
    return best


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of edges (i, j)
        p: number of layers (fixed at 2)
    
    Returns:
        numpy array of shape (p, 2) with rows (gamma, beta) in radians
    """
    # Compute the exact optimum
    opt = _exact_maxcut(n, edges)
    if opt == 0:
        # No edges or disconnected; return default angles
        return np.tile([0.35, 0.25], (p, 1))
    
    # Global grid search over a coarsely spaced, graph-scaled parameter space
    # For p=2, we have 4 parameters: gamma1, beta1, gamma2, beta2
    # Typical QAOA angles are in [0, pi] for both gamma and beta
    
    # Coarse grid
    grid_points = 8
    gamma_vals = np.linspace(0, np.pi, grid_points)
    beta_vals = np.linspace(0, np.pi, grid_points)
    
    best_ratio = -1.0
    best_params = None
    
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    gammas = np.array([g1, g2])
                    betas = np.array([b1, b2])
                    
                    state = _build_qaoa_state(n, edges, gammas, betas)
                    exp_cut = _expected_cut(state, edges)
                    ratio = exp_cut / opt
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_params = (g1, b1, g2, b2)
    
    # Single high-precision Nelder-Mead refinement from the best grid point
    def objective(params):
        g1, b1, g2, b2 = params
        gammas = np.array([g1, g2])
        betas = np.array([b1, b2])
        state = _build_qaoa_state(n, edges, gammas, betas)
        exp_cut = _expected_cut(state, edges)
        return -exp_cut / opt  # Minimize negative ratio
    
    x0 = np.array(best_params)
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    g1, b1, g2, b2 = result.x
    
    # Clip to reasonable range
    g1 = np.clip(g1, 0, np.pi)
    b1 = np.clip(b1, 0, np.pi)
    g2 = np.clip(g2, 0, np.pi)
    b2 = np.clip(b2, 0, np.pi)
    
    return np.array([[g1, b1], [g2, b2]])

# EVOLVE-BLOCK-END
