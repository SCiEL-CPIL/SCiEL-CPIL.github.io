"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force for small n."""
    if n > 20:
        raise ValueError("Exact MaxCut only for small n")
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for given QAOA angles.
    State vector simulation: |psi> = prod_{k=1..p} exp(-i beta_k X_total) exp(-i gamma_k C) |+>^n
    """
    # Initialize |+>^n
    # State vector size is 2^n. For n=10, this is 1024. Manageable.
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge bitmasks for cost operator C
    # C = sum_{(u,v) in E} Z_u Z_v
    # Z_u Z_v has eigenvalue 1 if bits u,v are same, -1 if different.
    # We apply exp(-i gamma C).
    # Since C is diagonal in computational basis, exp(-i gamma C) is diagonal.
    # D_j = exp(-i gamma * c_j) where c_j is eigenvalue of C on basis state j.
    
    # Precompute eigenvalues of C for all basis states
    c_vals = np.zeros(dim, dtype=np.float64)
    for j in range(dim):
        c = 0
        for u, v in edges:
            if ((j >> u) & 1) != ((j >> v) & 1):
                c -= 1  # Z_u Z_v = -1
            else:
                c += 1  # Z_u Z_v = 1
        c_vals[j] = c

    # Precompute mixer operator X_total = sum_i X_i
    # exp(-i beta X_total) is not diagonal, but can be applied efficiently.
    # However, for small n, we can precompute the matrix or use iterative application.
    # Actually, X_total is the sum of Pauli X on each qubit.
    # exp(-i beta X_total) = (cos(beta) I - i sin(beta) X_total)
    # This is a linear operator. We can apply it to the state vector.
    # X_total |j> = sum_i |j with bit i flipped>
    
    # Let's apply layers in reverse order of definition? No, definition is:
    # Start |+>^n
    # For k in 1..p:
    #   Apply exp(-i gamma_k C)
    #   Apply exp(-i beta_k X_total)
    # Wait, standard QAOA is usually:
    # U = exp(-i beta_p H_M) exp(-i gamma_p H_C) ... exp(-i beta_1 H_M) exp(-i gamma_1 H_C) |+>^n
    # The task says: "applies exp(-i gamma C), then exp(-i beta sum X), for each row"
    # So for row k (gamma_k, beta_k):
    #   psi = exp(-i beta_k X_total) @ exp(-i gamma_k C) @ psi
    
    for k in range(len(gammas)):
        # 1. Apply Cost: exp(-i gamma_k C)
        # This is diagonal multiplication
        phase = np.exp(-1j * gammas[k] * c_vals)
        psi = psi * phase
        
        # 2. Apply Mixer: exp(-i beta_k X_total)
        # = cos(beta_k) * psi - i sin(beta_k) * X_total psi
        # X_total psi: for each basis state j, (X_total psi)_j = sum_i psi_{j xor (1<<i)}
        # We can compute X_total psi efficiently.
        
        # Let's compute X_total psi
        x_psi = np.zeros_like(psi)
        # For each qubit i, flip bit i and add
        # This is O(n * 2^n)
        for i in range(n):
            mask = 1 << i
            # Flip bit i
            # We can use bit manipulation on indices
            # j -> j ^ mask
            # x_psi[j] += psi[j ^ mask]
            # Vectorized:
            # Create index array
            # Actually, simple loop is fine for n=10 (1024 states * 10 qubits = 10k ops)
            for j in range(dim):
                x_psi[j] += psi[j ^ mask]
        
        psi = np.cos(betas[k]) * psi - 1j * np.sin(betas[k]) * x_psi
        
        # Normalize to prevent numerical drift (optional but good)
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm

    # Calculate expected cut
    # E[Cut] = sum_{(u,v) in E} E[Z_u Z_v]
    # E[Z_u Z_v] = sum_j |psi_j|^2 * c_uv(j)
    # where c_uv(j) is 1 if j_u == j_v, -1 otherwise.
    
    probs = np.abs(psi) ** 2
    exp_cut = 0.0
    for u, v in edges:
        # Sum over all j
        # We can precompute the expectation of Z_u Z_v for all pairs?
        # Or just loop.
        exp_zu_zv = 0.0
        for j in range(dim):
            if ((j >> u) & 1) == ((j >> v) & 1):
                exp_zu_zv += probs[j]
            else:
                exp_zu_zv -= probs[j]
        exp_cut += exp_zu_zv
        
    return exp_cut

def _objective_and_grad(n, edges, gammas, betas):
    """
    Compute objective and gradient using finite differences or analytic?
    For p=2, we have 4 parameters.
    Finite differences are easy and robust for small p.
    """
    eps = 1e-5
    obj = _expected_cut(n, edges, gammas, betas)
    grad = np.zeros(4)
    
    # gamma_0
    g0 = gammas.copy(); g0[0] += eps
    grad[0] = (_expected_cut(n, edges, g0, betas) - obj) / eps
    g0[0] -= 2 * eps
    grad[0] = (obj - _expected_cut(n, edges, g0, betas)) / eps
    g0[0] += eps
    
    # gamma_1
    g1 = gammas.copy(); g1[1] += eps
    grad[1] = (_expected_cut(n, edges, g1, betas) - obj) / eps
    g1[1] -= 2 * eps
    grad[1] = (obj - _expected_cut(n, edges, g1, betas)) / eps
    g1[1] += eps
    
    # beta_0
    b0 = betas.copy(); b0[0] += eps
    grad[2] = (_expected_cut(n, edges, gammas, b0) - obj) / eps
    b0[0] -= 2 * eps
    grad[2] = (obj - _expected_cut(n, edges, gammas, b0)) / eps
    b0[0] += eps
    
    # beta_1
    b1 = betas.copy(); b1[1] += eps
    grad[3] = (_expected_cut(n, edges, gammas, b1) - obj) / eps
    b1[1] -= 2 * eps
    grad[3] = (obj - _expected_cut(n, edges, gammas, b1)) / eps
    b1[1] += eps
    
    return obj, grad

def _optimize_angles(n, edges, p):
    """
    Optimize angles using gradient descent with multiple restarts.
    """
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    best_params = None
    best_obj = -np.inf
    
    # Define a few diverse initial points
    # Known good heuristics for QAOA MaxCut:
    # gamma ~ 0.1-0.5, beta ~ 0.1-0.3
    # Random restarts in [0, pi]
    
    np.random.seed(42) # Reproducibility for development, but maybe randomize per call?
    # The task says "graph-dependent", so we should optimize for this specific graph.
    # But we have a time limit.
    
    # Let's try a few specific initializations
    inits = [
        [0.35, 0.35, 0.25, 0.25], # Seed program
        [0.1, 0.1, 0.1, 0.1],
        [0.5, 0.5, 0.2, 0.2],
        [0.2, 0.4, 0.1, 0.3],
        [np.pi/4, np.pi/4, np.pi/8, np.pi/8],
        [np.random.uniform(0, np.pi, 4)],
        [np.random.uniform(0, np.pi, 4)],
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        gammas = np.array([init[0], init[1]])
        betas = np.array([init[2], init[3]])
        
        # Gradient Descent
        lr = 0.1
        for step in range(100):
            if time.time() > deadline:
                break
                
            obj, grad = _objective_and_grad(n, edges, gammas, betas)
            
            if obj > best_obj:
                best_obj = obj
                best_params = (gammas.copy(), betas.copy())
                
            # Update
            gammas[0] += lr * grad[0]
            gammas[1] += lr * grad[1]
            betas[0] += lr * grad[2]
            betas[1] += lr * grad[3]
            
            # Clip to [0, 2*pi] or reasonable range
            gammas = np.clip(gammas, 0, 2*np.pi)
            betas = np.clip(betas, 0, 2*np.pi)
            
            # Adaptive learning rate
            if step % 10 == 9:
                # Check if progress is stalled
                pass # Simple fixed lr for now
        
    if best_params is None:
        # Fallback
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    g, b = best_params
    return np.array([[g[0], b[0]], [g[1], b[1]]])

def angles(n, edges, p):
    """
    Entry point.
    """
    start_time = time.time()
    
    # Check if we have time to optimize
    # If n is small, optimization is fast.
    # If n is large, it's slow.
    # For n=6,8,10, state vector is 64, 256, 1024.
    # _expected_cut is O(n * 2^n) per call.
    # Gradient requires 8 calls.
    # So one step is ~80 * 2^n operations.
    # For n=10, 80 * 1024 = 80k ops. Very fast.
    # 100 steps * 8 restarts = 800 steps.
    # 800 * 80k = 64M ops. Should be well under 1 second in Python/NumPy if vectorized.
    # My current _expected_cut is not fully vectorized (inner loop over j).
    # Let's optimize _expected_cut to be faster.
    
    # For now, let's just call the optimizer.
    # If it takes too long, we return the seed.
    
    try:
        res = _optimize_angles(n, edges, p)
    except Exception:
        res = np.tile([0.35, 0.25], (p, 1))
        
    return res

# EVOLVE-BLOCK-END
