"""Product-formula design for Heisenberg-chain time evolution."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """
    Build the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain.
    A: odd bonds + half field
    B: even bonds + half field
    Anisotropy lambda = 0.8, field h = 0.3
    """
    N = 6
    lambda_aniso = 0.8
    h_field = 0.3
    
    # Spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build operators for N sites
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    # Build A (odd bonds: 0-1, 2-3, 4-5) and B (even bonds: 1-2, 3-4)
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: each gets half the field
    field_term = 0.5 * h_field
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
    
    # Bond terms
    def add_bond(H, i, j):
        sx_i = op_on_site(sx, i, N)
        sx_j = op_on_site(sx, j, N)
        sy_i = op_on_site(sy, i, N)
        sy_j = op_on_site(sy, j, N)
        sz_i = op_on_site(sz, i, N)
        sz_j = op_on_site(sz, j, N)
        
        H += sx_i @ sx_j
        H += sy_i @ sy_j
        H += lambda_aniso * (sz_i @ sz_j)
    
    # Odd bonds for A: (0,1), (2,3), (4,5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B: (1,2), (3,4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

def _get_eig(H):
    """Diagonalize H (Hermitian) and return eigenvalues and eigenvectors."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals, eigvecs

def _compute_error_fast(a, b, t_values, U_exacts, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B):
    """
    Compute mean spectral-norm error for the product formula.
    U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    """
    dim = len(eigvals_A)
    error = 0.0
    for i, t in enumerate(t_values):
        U_curr = np.eye(dim, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = np.exp(-1j * b[k] * t * eigvals_B)
            U_B = (eigvecs_B * exp_B) @ eigvecs_B.conj().T
            U_curr = U_B @ U_curr
            
            # exp(-i a_k t A)
            exp_A = np.exp(-1j * a[k] * t * eigvals_A)
            U_A = (eigvecs_A * exp_A) @ eigvecs_A.conj().T
            U_curr = U_A @ U_curr
        
        diff = U_curr - U_exacts[i]
        err = np.linalg.norm(diff, 2)
        error += err
    return error / len(t_values)

def _nelder_mead(f, x0, bounds=None, max_iter=500, xatol=1e-8, fatol=1e-10, time_limit=50.0):
    """
    Simple Nelder-Mead implementation with time guard.
    """
    start_time = time.time()
    n = len(x0)
    
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += 0.05 if x0[i] >= 0 else -0.05
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    best_idx = np.argmin(fvals)
    best_x = simplex[best_idx].copy()
    best_f = fvals[best_idx]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for _ in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if fvals[0] - fvals[-1] < fatol:
            break
        if np.max(np.abs(np.array(simplex) - simplex[0])) < xatol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = f(x_e)
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
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
        
        # Track best
        cur_best_idx = np.argmin(fvals)
        if fvals[cur_best_idx] < best_f:
            best_f = fvals[cur_best_idx]
            best_x = simplex[cur_best_idx].copy()
    
    return best_x, best_f

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Evaluator kills at 150s, leave margin
    
    # Pre-compute Hamiltonians
    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Pre-compute exact propagators for the 3 time points
    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
        U_exact = (eigvecs_H * np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
        U_exacts.append(U_exact)
    
    # Diagonalize A and B
    eigvals_A, eigvecs_A = _get_eig(A)
    eigvals_B, eigvecs_B = _get_eig(B)
    
    # 4th order Suzuki-Trotter coefficients
    # c1 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3)), c3 = 1 - c1 - c2
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = 1.0 - c1 - c2
    
    # 12-stage symmetric 4th-order template
    # Pattern: c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1
    st4_template = np.array([c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1])
    
    # Uniform template
    uniform_template = np.ones(12) / 12.0
    
    # Strategy: 1D search over alpha interpolating uniform -> st4
    # a = (1-alpha)*uniform + alpha*st4
    # This preserves sum=1 automatically
    
    best_alpha = 0.0
    best_error = float('inf')
    
    # Coarse grid search
    alphas_coarse = np.linspace(0.0, 1.0, 50)
    for alpha in alphas_coarse:
        if time.time() > deadline:
            break
        
        a = (1.0 - alpha) * uniform_template + alpha * st4_template
        b = a.copy()
        
        error = _compute_error_fast(a, b, t_values, U_exacts, eigvals_A, eigvecs_A, eigvals_B, eigvals_B)
        
        if error < best_error:
            best_error = error
            best_alpha = alpha
    
    # Fine grid search around best
    if best_alpha > 0 and best_alpha < 1:
        alpha_min = max(0.0, best_alpha - 0.05)
        alpha_max = min(1.0, best_alpha + 0.05)
        alphas_fine = np.linspace(alpha_min, alpha_max, 200)
        
        for alpha in alphas_fine:
            if time.time() > deadline:
                break
            
            a = (1.0 - alpha) * uniform_template + alpha * st4_template
            b = a.copy()
            
            error = _compute_error_fast(a, b, t_values, U_exacts, eigvals_A, eigvecs_A, eigvals_B, eigvals_B)
            
            if error < best_error:
                best_error = error
                best_alpha = alpha
    
    # Now try a 2D search: scale the st4 template and add a correction
    # a = alpha * st4 + beta * correction
    # where correction is a 12-dim vector with sum=0
    # Let's try correction = [1,0,0,0,0,0,0,0,0,0,0,-1] (shift weight from last to first)
    # and other patterns
    
    if time.time() < deadline - 5.0:
        # Define a few correction patterns (each sums to 0)
        corrections = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]),
            np.array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0]),
            np.array([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1]),
            np.array([1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0]),
        ]
        
        for corr in corrections:
            if time.time() > deadline:
                break
            
            # 2D Nelder-Mead over (alpha, beta)
            def obj_2d(x):
                alpha_val, beta_val = x
                if alpha_val < 0 or alpha_val > 1.5 or beta_val < -0.5 or beta_val > 0.5:
                    return 1e10
                a = alpha_val * st4_template + beta_val * corr
                # Ensure sum = 1: st4 sums to 1, corr sums to 0, so sum(a) = alpha_val
                # Normalize if needed
                s = np.sum(a)
                if abs(s) < 1e-10:
                    return 1e10
                a = a / s
                b = a.copy()
                return _compute_error_fast(a, b, t_values, U_exacts, eigvals_A, eigvecs_A, eigvals_B, eigvals_B)
            
            x0_2d = np.array([best_alpha, 0.0])
            x_opt, f_opt = _nelder_mead(obj_2d, x0_2d, max_iter=200, xatol=1e-6, fatol=1e-8, 
                                        time_limit=max(0, deadline - time.time() - 2.0))
            
            if f_opt < best_error:
                best_error = f_opt
                best_alpha = x_opt[0]
                # Store the best a,b
                a_best = x_opt[0] * st4_template + x_opt[1] * corr
                s = np.sum(a_best)
                if abs(s) > 1e-10:
                    a_best = a_best / s
                b_best = a_best.copy()
    
    # Use the best alpha result if 2D didn't improve
    if time.time() < deadline:
        a = (1.0 - best_alpha) * uniform_template + best_alpha * st4_template
        b = a.copy()
        
        # Normalize to ensure sum = 1
        s = np.sum(a)
        if abs(s) > 1e-10:
            a = a / s
            b = b / s
    
    # Ensure we have exactly `stages` coefficients
    if stages != 12:
        # Fallback: use uniform
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
    else:
        a = list(a)
        b = list(b)
    
    # Final check for finite numbers in [-50, 50]
    a = [max(-50.0, min(50.0, float(ai))) for ai in a]
    b = [max(-50.0, min(50.0, float(bi))) for bi in b]
    
    return a, b
# EVOLVE-BLOCK-END
