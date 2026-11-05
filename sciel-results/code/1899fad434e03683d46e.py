import numpy as np
import time

def _expm(A):
    """Compute matrix exponential using Taylor series."""
    # Scaling and squaring is better, but for small matrices and small t, Taylor is fine.
    # However, t can be up to 1.0, and eigenvalues might be large.
    # Let's use a robust Taylor series with scaling.
    norm = np.linalg.norm(A, 2)
    if norm == 0:
        return np.eye(A.shape[0])
    
    # Determine scaling factor
    # We want ||A/scale|| to be small enough for Taylor convergence.
    # ||A/scale||^k / k! should drop quickly.
    # Let's try scaling such that norm/scale < 1.
    scale = max(1, int(np.ceil(norm / 1.0)))
    
    A_scaled = A / scale
    
    # Taylor series
    term = np.eye(A.shape[0], dtype=complex)
    result = term.copy()
    for k in range(1, 30):
        term = term @ A_scaled / k
        result = result + term
        if np.linalg.norm(term, 2) < 1e-15:
            break
    
    # Undo scaling: exp(A) = (exp(A/scale))^scale
    for _ in range(scale - 1):
        result = result @ result
        
    return result

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    d = 0.8  # anisotropy
    h = 0.3  # field
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on given site, identity elsewhere."""
        ops = [ident] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    def bond_op(i, j):
        """XXZ bond term between sites i and j."""
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return sx_i @ sx_j + sy_i @ sy_j + d * sz_i @ sz_j
    
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)
    
    # Half field on each
    half_field = 0.5 * h
    for site in range(n):
        A += half_field * op_on_site(sz, site)
        B += half_field * op_on_site(sz, site)
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    dim = A.shape[0]
    H = A + B
    errors = []
    for t in times:
        # Exact
        U_exact = _expm(-1j * t * H)
        # Product formula
        U_approx = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # Precomputing expm for each step is expensive.
            # But 12 stages * 3 times * 2 expms = 72 expms.
            # Each expm is 64x64. This is fast.
            U_approx = _expm(-1j * b[k] * t * B) @ _expm(-1j * a[k] * t * A) @ U_approx
        err = np.linalg.norm(U_approx - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _nelder_mead(objective, x0, max_iter=100, tol=1e-6):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.1 if x_new[i] == 0 else 0.1 * x_new[i]
        simplex.append(x_new)
    
    fvals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    return simplex[0], fvals[0]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    n = stages
    # Palindromic symmetry: 6 free a's, 6 free b's
    # Total 12 free parameters
    
    # Initial guess: uniform
    a_init = [1.0 / n] * n
    b_init = [1.0 / n] * n
    
    # Free params: a[0..5], b[0..5]
    x0 = np.array(a_init[:n//2] + b_init[:n//2])
    
    def objective(params):
        a_local = np.zeros(n)
        b_local = np.zeros(n)
        for k in range(n // 2):
            a_local[k] = params[k]
            a_local[n - 1 - k] = params[k]
            b_local[k] = params[n // 2 + k]
            b_local[n - 1 - k] = params[n // 2 + k]
        
        # Check constraints
        if abs(np.sum(a_local) - 1.0) > 1e-8 or abs(np.sum(b_local) - 1.0) > 1e-8:
            return 1e10
        if np.any(np.abs(a_local) > 50) or np.any(np.abs(b_local) > 50):
            return 1e10
        
        return _compute_error(a_local, b_local, A, B, times)
    
    best_a = a_init.copy()
    best_b = b_init.copy()
    best_err = objective(x0)
    
    # Run a few restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x0_current = x0.copy()
        else:
            # Perturb
            x0_current = x0 + np.random.randn(len(x0)) * 0.1
        
        try:
            x_opt, f_opt = _nelder_mead(objective, x0_current, max_iter=50, tol=1e-5)
            
            if f_opt < best_err:
                best_err = f_opt
                a_opt = np.zeros(n)
                b_opt = np.zeros(n)
                for k in range(n // 2):
                    a_opt[k] = x_opt[k]
                    a_opt[n - 1 - k] = x_opt[k]
                    b_opt[k] = x_opt[n // 2 + k]
                    b_opt[n - 1 - k] = x_opt[n // 2 + k]
                best_a = a_opt.tolist()
                best_b = b_opt.tolist()
        except Exception:
            pass
    
    # Ensure constraints are satisfied
    sum_a = sum(best_a)
    sum_b = sum(best_b)
    if abs(sum_a) > 1e-12:
        best_a = [x / sum_a for x in best_a]
    if abs(sum_b) > 1e-12:
        best_b = [x / sum_b for x in best_b]
    
    # Clamp to [-50, 50]
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Re-normalize after clamping
    sum_a = sum(best_a)
    sum_b = sum(best_b)
    if abs(sum_a) > 1e-12:
        best_a = [x / sum_a for x in best_a]
    if abs(sum_b) > 1e-12:
        best_b = [x / sum_b for x in best_b]
    
    return best_a, best_b
