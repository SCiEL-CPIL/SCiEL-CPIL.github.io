"""Optimized 12-stage product formula for Heisenberg chain time evolution."""
import numpy as np
import time
from scipy.linalg import expm
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    
    A = sum over odd bonds (1-2, 3-4, 5-6) of XXZ interaction + half uniform field
    B = sum over even bonds (2-3, 4-5) of XXZ interaction + half uniform field
    
    XXZ interaction on bond (i, i+1): Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + lambda Sz_i Sz_{i+1}
    Uniform field: h * sum_i Sz_i, split as h/2 in A and h/2 in B
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Compute Kronecker product of list of operators."""
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op, N):
        """Apply operator op at given site, identity elsewhere."""
        ops = [ident] * N
        ops[site] = op
        return kron_all(ops)
    
    def add_bond(A, i, j, lam):
        """Add XXZ interaction term between sites i and j (0-indexed)."""
        # Sx_i Sx_j + Sy_i Sy_j + lam * Sz_i Sz_j
        ops_sxsx = [ident] * N
        ops_sxsx[i] = sx
        ops_sxsx[j] = sx
        A += kron_all(ops_sxsx)
        
        ops_sysy = [ident] * N
        ops_sysy[i] = sy
        ops_sysy[j] = sy
        A += kron_all(ops_sysy)
        
        ops_szsz = [ident] * N
        ops_szsz[i] = sz
        ops_szsz[j] = sz
        A += lam * kron_all(ops_szsz)
    
    dim = 2 ** N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> sites 1-2, 3-4, 5-6 in 1-indexed
    for i in range(0, N, 2):
        add_bond(A, i, i + 1, lam)
    
    # Even bonds: (1,2), (3,4) -> sites 2-3, 4-5 in 1-indexed
    for i in range(1, N - 1, 2):
        add_bond(B, i, i + 1, lam)
    
    # Add half field to each
    for site in range(N):
        A += (h / 2) * op_on_site(site, sz, N)
        B += (h / 2) * op_on_site(site, sz, N)
    
    return A, B

def _compute_error(a, b, A, B, exact_factors, t_values):
    """Compute mean spectral norm error over given time values."""
    dim = A.shape[0]
    errors = []
    for t in t_values:
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            if b[k] != 0.0:
                U = np.linalg.multi_dot([U, expm(-1j * b[k] * t * B)])
            if a[k] != 0.0:
                U = np.linalg.multi_dot([U, expm(-1j * a[k] * t * A)])
        
        # Exact propagator
        U_exact = exact_factors[t]
        
        # Spectral norm of difference
        diff = U - U_exact
        # Use singular values for spectral norm
        sv = np.linalg.svd(diff, compute_uv=False)
        if len(sv) > 0:
            errors.append(sv[0])
        else:
            errors.append(0.0)
    
    return np.mean(errors)

def _nelder_mead(objective, x0, bounds, max_iter=2000, tol=1e-10, time_limit=100):
    """Nelder-Mead optimization with bounds and time limit."""
    n = len(x0)
    start_time = time.time()
    
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.01 if x0[i] == 0 else abs(x0[i]) * 0.1
        point[i] += step
        simplex.append(point)
    
    # Evaluate initial simplex
    fvals = []
    for i, point in enumerate(simplex):
        fvals.append(objective(point))
    
    # Sort by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    alpha, gamma, rho = 1.0, 2.0, 0.5
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < fvals[0]:
            # Expansion
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
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        
        # Re-sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Update best
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()
        
        # Check convergence
        if max(fvals) - min(fvals) < tol:
            break
    
    return best_x, best_f

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients.
    Returns (a, b) where each is a list of 12 finite numbers in [-50, 50]
    with sum(a) = sum(b) = 1.
    """
    if stages != 12:
        # Fallback for non-12 stage requests
        a = [1.0 / 12] * 12
        b = [1.0 / 12] * 12
        return a[:stages] + [0.0] * (12 - stages) if stages < 12 else a[:stages]
    
    start_time = time.time()
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Precompute exact propagators at evaluation times
    t_values = [0.25, 0.5, 1.0]
    H = A + B
    exact_factors = {}
    for t in t_values:
        exact_factors[t] = expm(-1j * t * H)
    
    # Initial guess: equal weights
    x0 = [1.0/12] * 24  # 12 a's and 12 b's
    
    def objective(x):
        """Objective function: mean spectral norm error."""
        a = list(x[:12])
        b = list(x[12:])
        return _compute_error(a, b, A, B, exact_factors, t_values)
    
    # Run Nelder-Mead optimization
    x_opt, f_opt = _nelder_mead(objective, x0, bounds=[(-50, 50)] * 24, 
                                  max_iter=5000, tol=1e-12, time_limit=100)
    
    a = list(x_opt[:12])
    b = list(x_opt[12:])
    
    # Ensure sums are exactly 1
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [ai / sum_a for ai in a]
    if abs(sum_b) > 1e-15:
        b = [bi / sum_b for bi in b]
    
    # Clamp to valid range
    a = [max(-50.0, min(50.0, ai)) for ai in a]
    b = [max(-50.0, min(50.0, bi)) for bi in b]
    
    # Re-normalize after clamping
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [ai / sum_a for ai in a]
    if abs(sum_b) > 1e-15:
        b = [bi / sum_b for bi in b]
    
    # Final check on time budget
    if time.time() - start_time > 110:
        # Return a safe fallback
        a = [1.0/12] * 12
        b = [1.0/12] * 12
    
    return a, b
