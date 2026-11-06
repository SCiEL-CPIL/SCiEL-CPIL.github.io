"""
Strategy: Physics-informed 1D search for p=2 QAOA.

Key insight: For p=2 QAOA, the optimal parameters often follow a pattern where
the second layer "corrects" the first layer based on graph structure. We exploit
this by:
1. Computing the graph Laplacian eigenvalues to characterize the graph.
2. Deriving a 1D parameterization where (gamma2, beta2) are functions of (gamma1, beta1)
   and graph features.
3. Exhaustively searching over the 1D parameter space.

This reduces the 4D optimization to 1D, enabling exhaustive search without local minima issues.
"""
import numpy as np
import time
import sys

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """Build C (MaxCut) and B (sum X) Hamiltonians with eigendecomposition."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
    
    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
    
    wC, VC = np.linalg.eigh(C)
    wB, VB = np.linalg.eigh(B)
    
    return wC, VC, wB, VB

def _expected_cut_fast(n, edges, gamma1, beta1, gamma2, beta2, wC, VC, wB, VB):
    """Fast expected cut using precomputed eigendecompositions."""
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Apply exp(-i gamma1 C)
    vH_psi = VC.conj().T @ psi
    vH_psi *= np.exp(-1j * gamma1 * wC)
    psi = VC @ vH_psi
    
    # Apply exp(-i beta1 B)
    vH_psi = VB.conj().T @ psi
    vH_psi *= np.exp(-1j * beta1 * wB)
    psi = VB @ vH_psi
    
    # Apply exp(-i gamma2 C)
    vH_psi = VC.conj().T @ psi
    vH_psi *= np.exp(-1j * gamma2 * wC)
    psi = VC @ vH_psi
    
    # Apply exp(-i beta2 B)
    vH_psi = VB.conj().T @ psi
    vH_psi *= np.exp(-1j * beta2 * wB)
    psi = VB @ vH_psi
    
    # Calculate <psi| C |psi>
    vC_psi = VC.conj().T @ psi
    val = np.sum(np.abs(vC_psi)**2 * wC).real
    return val

def _graph_features(n, edges):
    """Compute graph features for physics-informed parameterization."""
    if len(edges) == 0:
        return {'density': 0.0, 'max_degree': 0, 'avg_degree': 0.0, 'spectral_gap': 0.0}
    
    # Build adjacency matrix
    adj = np.zeros((n, n))
    for (u, v) in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Degree sequence
    degrees = adj.sum(axis=1)
    max_degree = int(np.max(degrees))
    avg_degree = float(np.mean(degrees))
    
    # Density
    max_edges = n * (n - 1) // 2
    density = len(edges) / max_edges if max_edges > 0 else 0.0
    
    # Spectral gap of Laplacian
    L = np.diag(degrees) - adj
    eigvals = np.linalg.eigvalsh(L)
    # Spectral gap is the second smallest eigenvalue (Fiedler value)
    spectral_gap = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    
    return {
        'density': density,
        'max_degree': max_degree,
        'avg_degree': avg_degree,
        'spectral_gap': spectral_gap
    }

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 5.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    if len(edges) == 0:
        return np.zeros((p, 2))
    
    try:
        wC, VC, wB, VB = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Compute graph features
    features = _graph_features(n, edges)
    
    def eval_angles(g1, b1, g2, b2):
        return _expected_cut_fast(n, edges, g1, b1, g2, b2, wC, VC, wB, VB)
    
    best_val = -1e10
    best_angles = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Physics-informed 1D search:
    # Hypothesis: Optimal parameters follow a pattern where:
    #   gamma2 = alpha * gamma1 + offset_gamma
    #   beta2 = alpha * beta1 + offset_beta
    # where alpha, offset_gamma, offset_beta depend on graph features.
    
    # Try multiple parameterization families
    
    # Family 1: Linear relationship with graph-dependent scaling
    # Based on empirical observations, for sparse graphs, gamma2 ~ gamma1/2, beta2 ~ beta1
    # For dense graphs, gamma2 ~ gamma1, beta2 ~ beta1/2
    
    density = features['density']
    spectral_gap = features['spectral_gap']
    
    # Scaling factor based on density
    # Sparse: gamma2 = 0.5*gamma1, Dense: gamma2 = gamma1
    alpha_g = 0.5 + 0.5 * density
    # Beta scaling is more complex
    alpha_b = 1.0 - 0.3 * density
    
    # Offset based on spectral gap
    offset_g = 0.1 * spectral_gap
    offset_b = 0.05 * spectral_gap
    
    # 1D search: vary gamma1 over [0, pi], beta1 = 0.25 (typical optimal)
    # Then compute gamma2, beta2 from the parameterization
    n_search = 200
    gamma1_vals = np.linspace(0.0, np.pi, n_search)
    
    for g1 in gamma1_vals:
        if _time_remaining() < 10.0:
            break
        
        # Try several beta1 values
        for b1 in [0.2, 0.25, 0.3, 0.35, 0.4]:
            g2 = np.clip(alpha_g * g1 + offset_g, 0, np.pi)
            b2 = np.clip(alpha_b * b1 + offset_b, 0, np.pi/2)
            
            val = eval_angles(g1, b1, g2, b2)
            if val > best_val:
                best_val = val
                best_angles = np.array([g1, b1, g2, b2])
    
    # Family 2: Inverse relationship
    # gamma2 = pi - gamma1, beta2 = pi/2 - beta1 (complementary)
    for g1 in gamma1_vals:
        if _time_remaining() < 10.0:
            break
        
        for b1 in [0.2, 0.25, 0.3]:
            g2 = np.clip(np.pi - g1, 0, np.pi)
            b2 = np.clip(np.pi/2 - b1, 0, np.pi/2)
            
            val = eval_angles(g1, b1, g2, b2)
            if val > best_val:
                best_val = val
                best_angles = np.array([g1, b1, g2, b2])
    
    # Family 3: Quadratic relationship
    # gamma2 = c * gamma1^2, beta2 = d * beta1^2
    c_vals = [0.5, 1.0, 1.5]
    d_vals = [0.5, 1.0, 1.5]
    
    for c in c_vals:
        for d in d_vals:
            for g1 in gamma1_vals[::2]:  # Coarser grid for this family
                if _time_remaining() < 10.0:
                    break
                
                for b1 in [0.25, 0.3]:
                    g2 = np.clip(c * g1**2 / np.pi, 0, np.pi)
                    b2 = np.clip(d * b1**2 / (np.pi/2), 0, np.pi/2)
                    
                    val = eval_angles(g1, b1, g2, b2)
                    if val > best_val:
                        best_val = val
                        best_angles = np.array([g1, b1, g2, b2])
    
    # Final polish: local search around the best found point
    if _time_remaining() > 15.0:
        # Simple local search with small perturbations
        step_sizes = [0.1, 0.05, 0.01]
        current_angles = best_angles.copy()
        
        for step in step_sizes:
            improved = True
            while improved and _time_remaining() > 10.0:
                improved = False
                for i in range(4):
                    for direction in [-1, 1]:
                        new_angles = current_angles.copy()
                        new_angles[i] += direction * step
                        # Bounds
                        if i in [0, 2]:  # gamma
                            new_angles[i] = np.clip(new_angles[i], 0, np.pi)
                        else:  # beta
                            new_angles[i] = np.clip(new_angles[i], 0, np.pi/2)
                        
                        val = eval_angles(*new_angles)
                        if val > best_val + 1e-10:
                            best_val = val
                            best_angles = new_angles
                            current_angles = new_angles
                            improved = True
    
    # Reshape to (p, 2) format
    result = best_angles.reshape(p, 2)
    
    # Ensure values are within reasonable bounds
    result[:, 0] = np.clip(result[:, 0], 0, np.pi)
    result[:, 1] = np.clip(result[:, 1], 0, np.pi/2)
    
    return result
