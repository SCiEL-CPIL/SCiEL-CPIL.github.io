"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_objective(params, n, adj):
    """
    Compute the expected cut size for given QAOA parameters.
    Uses exact state-vector simulation for small n (n <= 10).
    """
    gamma1, beta1, gamma2, beta2 = params
    
    # Initialize state vector: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Helper to apply exp(-i * gamma * C) where C is the MaxCut cost function
    # C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
    # exp(-i gamma C) = prod_{(i,j) in E} exp(-i gamma/2 (1 - Z_i Z_j))
    # = prod_{(i,j) in E} [exp(-i gamma/2) * (1/2)(I - Z_i Z_j) + exp(i gamma/2) * (1/2)(I + Z_i Z_j)]
    # Actually, for MaxCut, the standard QAOA cost is C = sum (1 - Z_i Z_j)/2.
    # The unitary is U_C = exp(-i gamma C).
    # For each edge (i,j), the factor is exp(-i gamma/2 (1 - Z_i Z_j)).
    # Let's precompute the edge list.
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
                
    if not edges:
        # No edges, expected cut is 0
        return 0.0

    # Apply C unitary
    # We can do this by iterating over edges and applying the two-qubit gate.
    # For speed with n<=10, we can use the fact that the gate only acts on pairs.
    # However, implementing a generic two-qubit gate application on a state vector
    # of size 2^n requires careful index manipulation.
    
    # A more efficient way for small n is to use the spectral decomposition or
    # simply apply the rotation.
    # Let's stick to a robust implementation.
    
    def apply_edge_gate(state, i, j, angle):
        """
        Apply exp(-i * angle * (1 - Z_i Z_j) / 2) to the state vector.
        Note: The standard QAOA cost term is usually defined such that the
        eigenvalues are 0 or 1. Here we use the phase kickback approach.
        
        The operator is U = exp(-i gamma/2) * P_0 + exp(i gamma/2) * P_1
        where P_0 projects onto the subspace where Z_i Z_j = 1 (same bit)
        and P_1 projects onto the subspace where Z_i Z_j = -1 (different bit).
        Wait, (1 - Z_i Z_j)/2 is 1 if bits differ, 0 if same.
        So exp(-i gamma * C_edge) = exp(-i gamma) if bits differ, 1 if same.
        
        Let's re-verify the definition.
        C = sum (1 - Z_i Z_j)/2.
        If Z_i Z_j = 1 (same), term is 0. exp(-i gamma * 0) = 1.
        If Z_i Z_j = -1 (diff), term is 1. exp(-i gamma * 1) = exp(-i gamma).
        
        So for each edge, we multiply the amplitude by exp(-i gamma) if the bits at i and j are different,
        and by 1 if they are the same.
        """
        # This can be done by iterating through all basis states? No, that's O(2^n * n) per edge.
        # Better to use numpy broadcasting or reshape.
        
        # Reshape state to n-qubit tensor
        tensor = state.reshape([2] * n)
        
        # Create masks for bits
        # This is tricky to do efficiently in pure numpy without loops for arbitrary n.
        # Given n <= 10, a loop over 2^n states might be acceptable if optimized, 
        # but 2^10 = 1024, and we have ~20 edges. 1024 * 20 = 20k operations. Very fast.
        
        # Let's just do it in a loop for clarity and correctness, it's fast enough for n=10.
        new_state = np.zeros_like(state)
        for mask in range(dim):
            bit_i = (mask >> i) & 1
            bit_j = (mask >> j) & 1
            if bit_i != bit_j:
                new_state[mask] = state[mask] * np.exp(-1j * angle)
            else:
                new_state[mask] = state[mask]
        
        return new_state

    def apply_mixer(state, beta):
        """
        Apply exp(-i beta sum X) = prod_k exp(-i beta X_k)
        exp(-i beta X) = cos(beta) I - i sin(beta) X
        So for each qubit k, we apply the rotation.
        """
        for k in range(n):
            # Apply gate to all pairs of states differing only in bit k
            for mask in range(dim):
                if (mask >> k) & 1 == 0:
                    # mask has 0 at k
                    mask_flip = mask | (1 << k)
                    # The gate is [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
                    # acting on |0> and |1>
                    a = state[mask]
                    b = state[mask_flip]
                    state[mask] = a * np.cos(beta) - 1j * b * np.sin(beta)
                    state[mask_flip] = -1j * a * np.sin(beta) + b * np.cos(beta)
        return state

    # Layer 1
    for i, j in edges:
        state = apply_edge_gate(state, i, j, gamma1)
    state = apply_mixer(state, beta1)
    
    # Layer 2
    for i, j in edges:
        state = apply_edge_gate(state, i, j, gamma2)
    state = apply_mixer(state, beta2)
    
    # Measure expected cut
    # Cut size for a mask is number of edges with different bits.
    # Probability of mask is |state[mask]|^2
    expected_cut = 0.0
    for mask in range(dim):
        prob = np.abs(state[mask])**2
        if prob > 1e-15:
            # Count cut edges for this mask
            cut_val = 0
            for i, j in edges:
                bit_i = (mask >> i) & 1
                bit_j = (mask >> j) & 1
                if bit_i != bit_j:
                    cut_val += 1
            expected_cut += prob * cut_val
            
    return expected_cut

def _exact_maxcut(n, adj):
    """Compute exact MaxCut value using brute force for small n."""
    dim = 1 << n
    max_cut = 0
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
                
    if not edges:
        return 0.0

    for mask in range(dim):
        cut_val = 0
        for i, j in edges:
            bit_i = (mask >> i) & 1
            bit_j = (mask >> j) & 1
            if bit_i != bit_j:
                cut_val += 1
        if cut_val > max_cut:
            max_cut = cut_val
    return max_cut

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        
    # Calculate density for heuristic initialization
    max_edges = n * (n - 1) / 2
    density = len(edges) / max_edges if max_edges > 0 else 0.0
    
    # Heuristic initial guess
    if density < 0.3:
        init = np.array([0.35, 0.25, 0.25, 0.15])
    elif density < 0.6:
        init = np.array([0.40, 0.20, 0.30, 0.10])
    else:
        init = np.array([0.45, 0.15, 0.35, 0.05])
        
    # Exact MaxCut
    opt_cut = _exact_maxcut(n, adj)
    if opt_cut == 0:
        return np.tile([0.35, 0.25], (p, 1))
        
    best_params = init
    best_val = -1.0
    
    # Objective function to maximize
    def objective(params):
        if time.time() > deadline:
            return -1e10
        val = _compute_objective(params, n, adj)
        return -val  # Minimize negative
        
    # Try L-BFGS-B optimization
    try:
        res = minimize(objective, init, method='L-BFGS-B', 
                       options={'maxiter': 50, 'disp': False})
        
        if time.time() < deadline:
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
                
    except Exception:
        pass
        
    # If time remains, try a few random restarts
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            rand_init = rng.uniform(0, np.pi/2, size=4)
            try:
                res = minimize(objective, rand_init, method='L-BFGS-B', 
                               options={'maxiter': 20, 'disp': False})
                if res.fun < -best_val:
                    best_val = -res.fun
                    best_params = res.x
            except Exception:
                pass
                
    # Format output
    # p=2, so we need 2 rows of (gamma, beta)
    # best_params is [gamma1, beta1, gamma2, beta2]
    out = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
