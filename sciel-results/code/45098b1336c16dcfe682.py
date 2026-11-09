"""Analytic gradient ascent QAOA angle design for fixed-depth p=2."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def _build_graph(n, edges):
    """Build adjacency matrix and edge list from input."""
    adj = np.zeros((n, n), dtype=float)
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        u, v = int(u), int(v)
        if u < 0 or u >= n or v < 0 or v >= n:
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    return adj, sorted(edge_list)


def _exhaustive_optimum(n, edge_list):
    """Compute exact MaxCut optimum by exhaustive enumeration."""
    best_cut = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edge_list:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return float(best_cut)


def _qaoa_expected_cut(n, edge_list, gamma, beta):
    """
    Compute the expected cut size for a p=2 QAOA circuit.
    
    The circuit is: |+>^n -> exp(-i*gamma*C) -> exp(-i*beta*sum X) -> exp(-i*gamma*C) -> exp(-i*beta*sum X)
    
    For p=2, the expected cut is:
    <psi| C |psi> where |psi> = U_beta U_gamma U_beta U_gamma |+>^n
    
    We use the fact that for each edge (u,v), the contribution is:
    (1 - <Z_u Z_v>) / 2
    
    So expected_cut = m - (1/2) * sum_{(u,v)} <Z_u Z_v>
    
    For the p=2 QAOA, we can compute <Z_u Z_v> efficiently.
    """
    m = len(edge_list)
    if m == 0:
        return 0.0
    
    # For p=2 QAOA, the state after each layer can be tracked.
    # Let's use the standard formula for 2-layer QAOA.
    # 
    # The key insight: for a graph, the 2-layer QAOA expected cut can be computed
    # using the adjacency structure.
    #
    # After first U_gamma: the state has phases based on cut size.
    # After first U_beta: rotations.
    # After second U_gamma: more phases.
    # After second U_beta: final state.
    #
    # A more efficient approach: use the fact that the cost operator C = sum_{(u,v)} (I - Z_u Z_v)/2
    # and compute <psi|C|psi> directly.
    #
    # For p=2, we can use the following approach:
    # The state |psi> = U_beta U_gamma U_beta U_gamma |+>^n
    #
    # Let's compute this using tensor networks or direct state evolution for small n.
    # Since n <= 10, we can use the full state vector (2^10 = 1024 dimensions).
    
    dim = 1 << n
    
    # Start state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    psi = np.ones(dim, dtype=complex)
    psi /= np.sqrt(dim)
    
    # Build the cost operator C as a matrix? No, that's 2^n x 2^n which is 1024x1024.
    # That's manageable. But let's think of a better way.
    
    # Actually, let's just apply the unitaries directly.
    # U_gamma = exp(-i * gamma * C) where C = sum_{(u,v) in E} (I - Z_u Z_v)/2
    # U_beta = exp(-i * beta * sum_i X_i)
    
    # For small n, we can build the matrices.
    
    def apply_u_gamma(psi, gamma):
        """Apply exp(-i*gamma*C) to psi."""
        # C = sum_{(u,v) in E} (I - Z_u Z_v)/2
        # exp(-i*gamma*C) = product over edges of exp(-i*gamma*(I-Z_u Z_v)/2)
        # But these don't commute in general, so we can't just multiply.
        # However, for the purpose of computing <psi|C|psi>, we can use:
        # <psi|C|psi> = sum_{(u,v)} <psi| (I - Z_u Z_v)/2 |psi>
        #              = m - (1/2) * sum_{(u,v)} <psi| Z_u Z_v |psi>
        #
        # But we need the state after applying U_gamma, not just the expectation.
        #
        # Let's build the full matrix for C and exponentiate it.
        
        # Build C matrix
        C = np.zeros((dim, dim), dtype=complex)
        for u, v in edge_list:
            # Z_u Z_v operator
            for x in range(dim):
                zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
                zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
                C[x, x] += (1.0 - zu * zv) / 2.0
        
        # exp(-i * gamma * C)
        # Since C is diagonal in the computational basis (it's a sum of Z_u Z_v terms),
        # exp(-i*gamma*C) is also diagonal!
        # C[x,x] = sum_{(u,v) in E} (1 - z_u(x) z_v(x))/2 = number of edges cut by x
        # So exp(-i*gamma*C)[x,x] = exp(-i*gamma*cut(x))
        
        result = psi.copy()
        for x in range(dim):
            cut_x = 0
            for u, v in edge_list:
                zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
                zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
                if zu != zv:
                    cut_x += 1
            result[x] *= np.exp(-1j * gamma * cut_x)
        return result
    
    def apply_u_beta(psi, beta):
        """Apply exp(-i*beta*sum_i X_i) to psi."""
        # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        # So for each basis state, we mix with its flip.
        result = np.zeros(dim, dtype=complex)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        for x in range(dim):
            # x and x^1 (flip bit 0), x^2 (flip bit 1), etc.
            # Actually, sum_i X_i means we flip one bit at a time.
            # exp(-i*beta*sum_i X_i) is NOT simply a product of single-qubit rotations
            # because the X operators on different qubits commute, so:
            # exp(-i*beta*sum_i X_i) = prod_i exp(-i*beta*X_i)
            # = prod_i (cos(beta)*I_i - i*sin(beta)*X_i)
            # This is a tensor product of single-qubit rotations!
            # So we can apply it qubit by qubit.
            pass
        
        # Apply single-qubit rotation for each qubit
        # R_x(beta) = cos(beta/2)*I - i*sin(beta/2)*X  ... wait, exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        # Let me redo: exp(-i*theta*X) = cos(theta)*I - i*sin(theta)*X
        # So for each qubit, the rotation is:
        # |0> -> cos(beta)|0> - i*sin(beta)|1>
        # |1> -> -i*sin(beta)|0> + cos(beta)|1>
        
        # Since it's a tensor product, we can apply it to each qubit independently.
        for q in range(n):
            # Apply rotation on qubit q
            new_psi = psi.copy()
            for x in range(dim):
                if ((x >> q) & 1) == 0:
                    # |0> component
                    partner = x | (1 << q)
                    new_psi[x] = cos_b * psi[x] - 1j * sin_b * psi[partner]
                else:
                    # |1> component
                    partner = x & ~(1 << q)
                    new_psi[x] = -1j * sin_b * psi[partner] + cos_b * psi[x]
            psi = new_psi
        
        return psi
    
    # Apply the circuit
    psi = apply_u_gamma(psi, gamma)
    psi = apply_u_beta(psi, beta)
    psi = apply_u_gamma(psi, gamma)
    psi = apply_u_beta(psi, beta)
    
    # Compute expected cut
    expected_cut = 0.0
    for u, v in edge_list:
        # <Z_u Z_v>
        zz = 0.0
        for x in range(dim):
            zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
            zz += zu * zv * (np.abs(psi[x]) ** 2)
        expected_cut += (1.0 - zz) / 2.0
    
    return float(np.real(expected_cut))


def _qaoa_expected_cut_full(n, edge_list, gamma1, beta1, gamma2, beta2):
    """
    Compute the expected cut size for a p=2 QAOA circuit with two layers.
    
    The circuit is: |+>^n -> exp(-i*gamma1*C) -> exp(-i*beta1*sum X) -> exp(-i*gamma2*C) -> exp(-i*beta2*sum X)
    """
    m = len(edge_list)
    if m == 0:
        return 0.0
    
    dim = 1 << n
    
    # Start state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    psi = np.ones(dim, dtype=complex)
    psi /= np.sqrt(dim)
    
    def apply_u_gamma(psi, gamma):
        """Apply exp(-i*gamma*C) to psi."""
        result = psi.copy()
        for x in range(dim):
            cut_x = 0
            for u, v in edge_list:
                zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
                zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
                if zu != zv:
                    cut_x += 1
            result[x] *= np.exp(-1j * gamma * cut_x)
        return result
    
    def apply_u_beta(psi, beta):
        """Apply exp(-i*beta*sum_i X_i) to psi."""
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Since it's a tensor product, we can apply it to each qubit independently.
        for q in range(n):
            # Apply rotation on qubit q
            new_psi = psi.copy()
            for x in range(dim):
                if ((x >> q) & 1) == 0:
                    # |0> component
                    partner = x | (1 << q)
                    new_psi[x] = cos_b * psi[x] - 1j * sin_b * psi[partner]
                else:
                    # |1> component
                    partner = x & ~(1 << q)
                    new_psi[x] = -1j * sin_b * psi[partner] + cos_b * psi[x]
            psi = new_psi
        
        return psi
    
    # Apply the circuit
    psi = apply_u_gamma(psi, gamma1)
    psi = apply_u_beta(psi, beta1)
    psi = apply_u_gamma(psi, gamma2)
    psi = apply_u_beta(psi, beta2)
    
    # Compute expected cut
    expected_cut = 0.0
    for u, v in edge_list:
        # <Z_u Z_v>
        zz = 0.0
        for x in range(dim):
            zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
            zz += zu * zv * (np.abs(psi[x]) ** 2)
        expected_cut += (1.0 - zz) / 2.0
    
    return float(np.real(expected_cut))


def _qaoa_gradient(n, edge_list, cuts, gamma1, beta1, gamma2, beta2, eps=1e-4):
    """
    Compute the gradient of the expected cut with respect to the four angles.
    Uses central differences for numerical stability.
    """
    # Compute current value
    curr_val = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1, gamma2, beta2)
    
    # Gradient with respect to gamma1
    val_g1_plus = _qaoa_expected_cut_full(n, edge_list, gamma1 + eps, beta1, gamma2, beta2)
    val_g1_minus = _qaoa_expected_cut_full(n, edge_list, gamma1 - eps, beta1, gamma2, beta2)
    grad_g1 = (val_g1_plus - val_g1_minus) / (2 * eps)
    
    # Gradient with respect to beta1
    val_b1_plus = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1 + eps, gamma2, beta2)
    val_b1_minus = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1 - eps, gamma2, beta2)
    grad_b1 = (val_b1_plus - val_b1_minus) / (2 * eps)
    
    # Gradient with respect to gamma2
    val_g2_plus = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1, gamma2 + eps, beta2)
    val_g2_minus = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1, gamma2 - eps, beta2)
    grad_g2 = (val_g2_plus - val_g2_minus) / (2 * eps)
    
    # Gradient with respect to beta2
    val_b2_plus = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1, gamma2, beta2 + eps)
    val_b2_minus = _qaoa_expected_cut_full(n, edge_list, gamma1, beta1, gamma2, beta2 - eps)
    grad_b2 = (val_b2_plus - val_b2_minus) / (2 * eps)
    
    return np.array([grad_g1, grad_b1, grad_g2, grad_b2])


def angles(n, edges, p):
    """
    Design p=2 QAOA angles for the given graph using analytic gradient ascent.
    """
    adj, edge_list = _build_graph(n, edges)
    m = len(edge_list)
    
    if m == 0:
        return np.zeros((p, 2), dtype=float)
    
    opt = _exhaustive_optimum(n, edge_list)
    if opt == 0:
        return np.zeros((p, 2), dtype=float)
    
    # Precompute cuts for all basis states
    dim = 1 << n
    cuts = np.zeros(dim, dtype=int)
    for x in range(dim):
        cut_x = 0
        for u, v in edge_list:
            zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
            if zu != zv:
                cut_x += 1
        cuts[x] = cut_x
    
    # Multiple initializations to avoid local optima
    inits = [
        (0.35, 0.25, 0.25, 0.15),
        (0.5, 0.3, 0.4, 0.2),
        (0.4, 0.4, 0.3, 0.3),
        (0.6, 0.2, 0.5, 0.25),
        (0.3, 0.5, 0.2, 0.4),
        (0.55, 0.35, 0.45, 0.3),
        (0.45, 0.45, 0.35, 0.35),
    ]
    
    best_val = -np.inf
    best_angles = (0.35, 0.25, 0.25, 0.15)
    
    for g1_0, b1_0, g2_0, b2_0 in inits:
        g1, b1, g2, b2 = g1_0, b1_0, g2_0, b2_0
        
        # Adaptive learning rate gradient ascent
        lr = 0.1
        num_steps = 500
        
        for step in range(num_steps):
            # Compute gradient
            grad = _qaoa_gradient(n, edge_list, cuts, g1, b1, g2, b2)
            
            # Gradient ascent step
            g1 += lr * grad[0]
            b1 += lr * grad[1]
            g2 += lr * grad[2]
            b2 += lr * grad[3]
            
            # Keep angles in reasonable range
            g1 = np.clip(g1, -np.pi, np.pi)
            b1 = np.clip(b1, -np.pi, np.pi)
            g2 = np.clip(g2, -np.pi, np.pi)
            b2 = np.clip(b2, -np.pi, np.pi)
            
            # Decay learning rate
            if step % 100 == 0 and step > 0:
                lr *= 0.7
        
        # Evaluate final
        val = _qaoa_expected_cut_full(n, edge_list, g1, b1, g2, b2)
        
        if val > best_val:
            best_val = val
            best_angles = (g1, b1, g2, b2)
    
    g1, b1, g2, b2 = best_angles
    return np.array([[g1, b1], [g2, b2]], dtype=float)
# EVOLVE-BLOCK-END
