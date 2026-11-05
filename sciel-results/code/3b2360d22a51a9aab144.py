import numpy as np
import time

def _build_heisenberg_matrices():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ Heisenberg chain."""
    # 6-site open chain, sites 0..5
    # XXZ: Jz * Sz_i Sz_j + Jxy/2 * (Sx_i Sx_j + Sy_i Sy_j)
    # Anisotropy 0.8: Jz/Jxy = 0.8, so Jz = 0.8, Jxy = 1.0 (or scale)
    # Field 0.3: uniform field, split half into A, half into B
    
    N = 6
    dim = 2 ** N
    Jz = 0.8
    Jxy = 1.0
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(mat, site):
        """Apply mat to site, I on others."""
        ops = [I2] * N
        ops[site] = mat
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_hamiltonian(site1, site2, Jz, Jxy):
        """XXZ bond Hamiltonian between site1 and site2."""
        sz1 = op_on_site(sz, site1)
        sz2 = op_on_site(sz, site2)
        sx1 = op_on_site(sx, site1)
        sx2 = op_on_site(sx, site2)
        sy1 = op_on_site(sy, site1)
        sy2 = op_on_site(sy, site2)
        return Jz * sz1 @ sz2 + (Jxy / 2) * (sx1 @ sx2 + sy1 @ sy2)
    
    # Field term: h * sum_i Sz_i, split half into A, half into B
    field = h * sum(op_on_site(sz, i) for i in range(N))
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    A = bond_hamiltonian(0, 1, Jz, Jxy) + bond_hamiltonian(2, 3, Jz, Jxy) + bond_hamiltonian(4, 5, Jz, Jxy) + field / 2
    B = bond_hamiltonian(1, 2, Jz, Jxy) + bond_hamiltonian(3, 4, Jz, Jxy) + field / 2
    
    return A, B

def _spectral_norm_error(a, b, A, B, t_values):
    """Compute mean spectral norm error."""
    errors = []
    for t in t_values:
        # Build product formula
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            eigvals_b, eigvecs_b = np.linalg.eigh(B)
            expB = eigvecs_b @ np.diag(np.exp(-1j * b[k] * t * eigvals_b)) @ eigvecs_b.conj().T
            # exp(-i a_k t A)
            eigvals_a, eigvecs_a = np.linalg.eigh(A)
            expA = eigvecs_a @ np.diag(np.exp(-1j * a[k] * t * eigvals_a)) @ eigvecs_a.conj().T
            U = expB @ U @ expA
        # Exact
        eigvals_ab, eigvecs_ab = np.linalg.eigh(A + B)
        U_exact = eigvecs_ab @ np.diag(np.exp(-1j * t * eigvals_ab)) @ eigvecs_ab.conj().T
        # Error
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        x[i] += 0.1 if x[i] + 0.1 <= bounds[1][i] else -0.1
        simplex.append(x)
    
    fvals = [f(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + (centroid - simplex[-1])
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + 2 * (centroid - simplex[-1])
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
            x_c = centroid + 0.5 * (simplex[-1] - centroid)
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    return min(fvals), simplex[np.argmin(fvals)]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    deadline = time.time() + 50  # 50s safety margin
    
    A, B = _build_heisenberg_matrices()
    t_values = [0.25, 0.5, 1.0]
    
    # Parameterize: 22 free params (12 a's + 12 b's, minus 2 constraints)
    # Use a symmetric parameterization to reduce search space
    # a_k = a_{13-k}, b_k = b_{13-k} -> 6+6=12 free params
    
    def make_coeffs(params):
        a_half = params[:6]
        b_half = params[6:12]
        a = list(a_half) + list(a_half[::-1])
        b = list(b_half) + list(b_half[::-1])
        # Normalize
        a = [x / sum(a) for x in a]
        b = [x / sum(b) for x in b]
        # Clamp
        a = [max(-50, min(50, x)) for x in a]
        b = [max(-50, min(50, x)) for x in b]
        a = [x / sum(a) for x in a]
        b = [x / sum(b) for x in b]
        return a, b
    
    def objective(params):
        a, b = make_coeffs(params)
        return _spectral_norm_error(a, b, A, B, t_values)
    
    # Initial guess: uniform
    x0 = [1.0/12] * 12
    
    best_error = objective(x0)
    best_params = x0.copy()
    
    # Nelder-Mead optimization
    try:
        err, params = _nelder_mead(objective, x0, ([-50]*12, [50]*12), max_iter=300)
        if err < best_error:
            best_error = err
            best_params = params
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    a, b = make_coeffs(best_params)
    return a, b
