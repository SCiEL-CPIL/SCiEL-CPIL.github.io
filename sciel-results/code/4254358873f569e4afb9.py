import numpy as np
import time

def _build_matrices():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain with anisotropy 0.8, field 0.3"""
    N = 6
    dim = 2 ** N
    delta = 0.8  # anisotropy
    field = 0.3
    
    # Identity
    I = np.eye(dim, dtype=complex)
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build local operators
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for site in range(N):
        # Build Sx, Sy, Sz for this site
        ops_x = [I] * N
        ops_y = [I] * N
        ops_z = [I] * N
        ops_x[site] = sx
        ops_y[site] = sy
        ops_z[site] = sz
        
        Sx_site = kron_all(ops_x)
        Sy_site = kron_all(ops_y)
        Sz_site = kron_all(ops_z)
        
        # Field term: half goes to A, half to B
        field_term = 0.5 * field * Sz_site
        A += field_term
        B += field_term
    
    # Exchange terms: Sx_i Sx_j + Sy_i Sy_j + delta * Sz_i Sz_j
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    # Wait, for 6 sites: bonds are (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds (odd index): (1,2), (3,4) -> B
    # Even bonds (even index): (0,1), (2,3), (4,5) -> A
    
    # Actually, let me reconsider. The task says "odd- and even-bond halves"
    # For a chain of 6 sites, bonds are between sites (0,1), (1,2), (2,3), (3,4), (4,5)
    # "Odd bonds" could mean bonds with odd indices: (1,2), (3,4)
    # "Even bonds" could mean bonds with even indices: (0,1), (2,3), (4,5)
    
    # Let me assign:
    # A = even bonds: (0,1), (2,3), (4,5) + half field
    # B = odd bonds: (1,2), (3,4) + half field
    
    # Clear and rebuild
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field: split evenly
    for site in range(N):
        ops_z = [I] * N
        ops_z[site] = sz
        Sz_site = kron_all(ops_z)
        field_term = 0.5 * field * Sz_site
        A += field_term
        B += field_term
    
    # Exchange terms
    def exchange_term(i, j):
        ops_xi = [I] * N
        ops_xj = [I] * N
        ops_yi = [I] * N
        ops_yj = [I] * N
        ops_zi = [I] * N
        ops_zj = [I] * N
        
        ops_xi[i] = sx
        ops_xj[j] = sx
        ops_yi[i] = sy
        ops_yj[j] = sy
        ops_zi[i] = sz
        ops_zj[j] = sz
        
        Sx_i_Sx_j = kron_all(ops_xi) @ kron_all(ops_xj)
        Sy_i_Sy_j = kron_all(ops_yi) @ kron_all(ops_yj)
        Sz_i_Sz_j = kron_all(ops_zi) @ kron_all(ops_zj)
        
        return Sx_i_Sx_j + Sy_i_Sy_j + delta * Sz_i_Sz_j
    
    # Even bonds: (0,1), (2,3), (4,5) -> A
    A += exchange_term(0, 1)
    A += exchange_term(2, 3)
    A += exchange_term(4, 5)
    
    # Odd bonds: (1,2), (3,4) -> B
    B += exchange_term(1, 2)
    B += exchange_term(3, 4)
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error over given times"""
    dim = A.shape[0]
    errors = []
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B)
            eB = np.linalg.matrix_exp(-1j * b[k] * t * B)
            U = eB @ U
            # exp(-i a_k t A)
            eA = np.linalg.matrix_exp(-1j * a[k] * t * A)
            U = eA @ U
        
        # Exact propagator
        H = A + B
        U_exact = np.linalg.matrix_exp(-1j * t * H)
        
        # Spectral norm of difference
        diff = U - U_exact
        err = np.linalg.norm(diff, 2)
        errors.append(err)
    
    return np.mean(errors)

def _nelder_mead(f, x0, bounds=None, max_iter=200, tol=1e-8, time_limit=None, start_time=None):
    """Simple Nelder-Mead optimizer with time limit"""
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        perturbation = x0.copy()
        step = 0.01 if abs(x0[i]) < 1e-6 else 0.01 * abs(x0[i])
        perturbation[i] += step
        simplex.append(perturbation)
    
    # Evaluate initial points
    values = [f(s) for s in simplex]
    
    best_x = simplex[np.argmin(values)]
    best_f = min(values)
    
    for iteration in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i].copy() for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        elif fr < values[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        else:
            # Contraction
            rho = 0.5
            if fr < values[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        values[i] = f(simplex[i])
            else:
                # Inside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        values[i] = f(simplex[i])
        
        # Track best
        idx = np.argmin(values)
        if values[idx] < best_f:
            best_f = values[idx]
            best_x = simplex[idx].copy()
    
    return best_x, best_f

def coefficients(stages):
    start_time = time.time()
    time_limit = 120.0  # Leave margin from 150s limit
    
    if stages != 12:
        # Fallback for non-12 stages
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build matrices
    A, B = _build_matrices()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: use 4th-order Yoshida coefficients pattern
    # For 12 stages, use a symmetric pattern
    c = 2 ** (1.0/3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    
    # Symmetric 12-stage initialization
    # Pattern: a, b, a, b, a, b, a, b, a, b, a, b
    # Use 4th-order-like weights
    a0 = [x3/4, x1/4, x2/4, x3/4, x1/4, x2/4, x2/4, x1/4, x3/4, x2/4, x1/4, x3/4]
    b0 = [x2/4, x1/4, x3/4, x2/4, x1/4, x3/4, x3/4, x1/4, x2/4, x3/4, x1/4, x2/4]
    
    # Normalize to ensure sums are 1
    a0 = [x / sum(a0) for x in a0]
    b0 = [x / sum(b0) for x in b0]
    
    # Combine into single vector for optimization
    x0 = a0 + b0
    
    def objective(x):
        a = x[:12]
        b = x[12:]
        return _compute_error(a, b, A, B, times)
    
    # Run Nelder-Mead
    best_x, best_f = _nelder_mead(
        objective, x0, max_iter=150, tol=1e-8,
        time_limit=time_limit, start_time=start_time
    )
    
    # Extract results
    a = best_x[:12]
    b = best_x[12:]
    
    # Normalize to ensure sums are exactly 1
    a = [x / sum(a) for x in a]
    b = [x / sum(b) for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Re-normalize after clamping
    a = [x / sum(a) for x in a]
    b = [x / sum(b) for x in b]
    
    return a, b
