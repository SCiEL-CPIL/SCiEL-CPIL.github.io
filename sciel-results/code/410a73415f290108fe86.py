"""
Optimized product formula for Heisenberg-chain time evolution.
Uses precomputed eigen-decompositions and Nelder-Mead optimization
to minimize mean spectral-norm error over three evolution times.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half uniform field) and B (even bonds + half uniform field)
    for a 6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op, N):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz, N)
        B += 0.5 * field * op_on_site(j, sz, N)
    
    return A, B


def _precompute_eig(H):
    """Precompute eigen-decomposition for Hermitian H."""
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def _expm_fast(evals, evecs, t):
    """Compute exp(-i*t*H) using precomputed eigen-decomposition."""
    phase = np.exp(-1j * t * evals)
    return evecs @ (phase[:, None] * evecs.conj().T)


def _objective(coeffs_vec, A_eig, B_eig, H_eig, dim):
    """
    Objective function: mean spectral norm error over t = 0.25, 0.5, 1.0.
    coeffs_vec: 24-element vector = [a_1..a_12, b_1..b_12]
    A_eig, B_eig, H_eig: (evals, evecs) tuples
    """
    stages = 12
    a = coeffs_vec[:stages]
    b = coeffs_vec[stages:]
    
    # Check consistency
    if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
        return 1e10
    
    # Check bounds
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact: exp(-i t H)
        exact = _expm_fast(H_eig[0], H_eig[1], t)
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        
        for k in range(stages):
            # exp(-i a_k t A)
            exp_a = _expm_fast(A_eig[0], A_eig[1], a[k] * t)
            
            # exp(-i b_k t B)
            exp_b = _expm_fast(B_eig[0], B_eig[1], b[k] * t)
            
            U = exp_b @ exp_a @ U
        
        err = np.linalg.svd(U - exact, compute_uv=False)[0]
        errors.append(err)
    
    return np.mean(errors)


def _nelder_mead(objective, x0, max_iter=500, timeout=None, start_time=None):
    """
    Nelder-Mead optimization with adaptive parameters.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        v = np.array(x0, dtype=float).copy()
        step = 0.1 * (abs(v[i]) + 0.5)
        v[i] += step
        simplex.append(v)
    
    fvals = [objective(v) for v in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(fvals)
    stagnant = 0
    
    for i in range(max_iter):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[idx] for idx in order]
        fvals = [fvals[idx] for idx in order]
        
        if fvals[0] < best_f:
            best_f = fvals[0]
            stagnant = 0
        else:
            stagnant += 1
        
        if stagnant > 30:
            break
        
        if abs(fvals[0] - fvals[-1]) < 1e-12:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    fvals[j] = objective(simplex[j])
    
    order = np.argsort(fvals)
    best_idx = order[0]
    return simplex[best_idx], fvals[best_idx]


def coefficients(stages):
    """
    Return 12-stage product formula coefficients (a, b) that minimize
    the mean spectral-norm error of the time evolution approximation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = 2 ** 6
    
    # Precompute eigen-decompositions
    A_eig = _precompute_eig(A)
    B_eig = _precompute_eig(B)
    H_eig = _precompute_eig(H)
    
    # Yoshida 4th-order coefficients for initial guess
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    
    # Build a symmetric 12-stage initial guess based on Yoshida pattern
    # The 5-step Yoshida pattern: A(x3), B(x2), A(x1), B(x2), A(x3)
    # For 12 stages with interleaved A,B: use a symmetric distribution
    # a: 6 A-stages, b: 6 B-stages interleaved
    # Use Yoshida weights distributed across the stages
    
    # Symmetric a pattern (12 values):
    # Group into pairs for symmetry: [w1, w2, w3, w4, w5, w6, w6, w5, w4, w3, w2, w1]
    # Sum must be 1
    # Based on Yoshida: x3 appears at ends, x1 in middle
    a_weights = np.array([x3/3, x1/3, x1/3, x1/3, x1/3, x3/3, x3/3, x1/3, x1/3, x1/3, x1/3, x3/3])
    a_weights = a_weights / np.sum(a_weights)
    
    # b: 12 values, all from Yoshida x2 distributed
    # Symmetric: [x2/6, x2/6, x2/6, x2/6, x2/6, x2/6, x2/6, x2/6, x2/6, x2/6, x2/6, x2/6]
    b_weights = np.array([x2/6] * 12)
    b_weights = b_weights / np.sum(b_weights)
    
    a_init = a_weights
    b_init = b_weights
    
    x0 = np.concatenate([a_init, b_init])
    
    # Define objective with cached eigen-decompositions
    def obj(vec):
        return _objective(vec, A_eig, B_eig, H_eig, dim)
    
    # Run Nelder-Mead with time limit
    remaining = deadline - time.time() - 5.0  # 5s safety margin
    if remaining < 5.0:
        # Not enough time for optimization, return initial guess
        return a_init.tolist(), b_init.tolist()
    
    x_best, f_best = _nelder_mead(
        obj, x0, max_iter=300,
        timeout=remaining, start_time=start_time
    )
    
    a_opt = x_best[:stages].tolist()
    b_opt = x_best[stages:].tolist()
    
    # Clamp to [-50, 50]
    a_opt = [max(-50.0, min(50.0, v)) for v in a_opt]
    b_opt = [max(-50.0, min(50.0, v)) for v in b_opt]
    
    # Normalize to ensure sum = 1
    sum_a = sum(a_opt)
    sum_b = sum(b_opt)
    if abs(sum_a) > 1e-10:
        a_opt = [v / sum_a for v in a_opt]
    if abs(sum_b) > 1e-10:
        b_opt = [v / sum_b for v in b_opt]
    
    # Final validation
    if len(a_opt) != stages or len(b_opt) != stages:
        return a_init.tolist(), b_init.tolist()
    
    return a_opt, b_opt

# EVOLVE-BLOCK-END
