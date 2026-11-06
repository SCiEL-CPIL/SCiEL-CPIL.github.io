"""Product-formula coefficient optimization for Heisenberg-chain time evolution."""
import time
import numpy as np
from scipy.linalg import expm, sqrtm

# Deadline guard: the evaluator kills after 150s wall-clock.
_DEADLINE = 140.0
_START = time.time()

def _remaining():
    return _DEADLINE - time.time()

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site open XXZ chain.
    
    Anisotropy gamma = 0.8, field h = 0.3.
    Hamiltonian H = sum_{j=1}^{5} [gamma/2 (S^z_j S^z_{j+1} + S^x_j S^x_{j+1} + S^y_j S^y_{j+1})] + h * sum S^z_j
    Wait, standard XXZ: H = sum [gamma S^z_j S^z_{j+1} + (1/2)(S^x_j S^x_{j+1} + S^y_j S^y_{j+1})] + h sum S^z_j
    
    Let's use the standard form:
    H = sum_{j=1}^{N-1} [ (1/2)(S^x_j S^x_{j+1} + S^y_j S^y_{j+1}) + gamma S^z_j S^z_{j+1} ] + h sum_{j=1}^N S^z_j
    
    With gamma = 0.8, h = 0.3, N = 6.
    Split into even bonds (j=2,4) and odd bonds (j=1,3,5).
    A = odd bonds + h/2 * sum S^z_j
    B = even bonds + h/2 * sum S^z_j
    """
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Spin-1/2 matrices
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def site_op(site, op):
        """Create operator that acts as `op` on `site` (0-indexed) and identity elsewhere."""
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        """Create operator op1 on site j, op2 on site j+1, identity elsewhere."""
        ops = [I] * N
        ops[j] = op1
        ops[j + 1] = op2
        return kron_all(ops)
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: h/2 * sum S^z_j for both A and B
    field_half = (h / 2.0) * sum(site_op(j, Sz) for j in range(N))
    A += field_half
    B += field_half
    
    # Bonds: j=0..4 (0-indexed)
    # Odd bonds in 1-indexed: j=1,3,5 -> 0-indexed: j=0,2,4
    # Even bonds in 1-indexed: j=2,4 -> 0-indexed: j=1,3
    
    for j in range(N - 1):
        # XX + YY term
        xx = bond_op(j, Sx, Sx)
        yy = bond_op(j, Sy, Sy)
        zz = bond_op(j, Sz, Sz)
        
        bond_term = 0.5 * (xx + yy) + gamma * zz
        
        if (j + 1) % 2 == 1:  # 1-indexed odd bond
            A += bond_term
        else:  # 1-indexed even bond
            B += bond_term
    
    return A, B

def _mat_exp(M):
    """Matrix exponential."""
    return expm(M)

def _spectral_norm(M):
    """Spectral norm (largest singular value)."""
    try:
        return np.linalg.norm(M, 2)
    except:
        # Fallback to Frobenius norm if SVD fails
        return np.linalg.norm(M, 'fro')

def _objective(coeffs, A, B, times):
    """Compute mean spectral norm error over the given times."""
    a, b = coeffs
    n = len(a)
    if len(b) != n:
        return 1e10
    
    # Check consistency
    if abs(sum(a) - 1.0) > 1e-8 or abs(sum(b) - 1.0) > 1e-8:
        return 1e10
    
    # Check bounds
    for x in a + b:
        if not np.isfinite(x) or x < -50 or x > 50:
            return 1e10
    
    errors = []
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(n):
            if abs(b[k]) > 1e-15:
                U = _mat_exp(-1j * b[k] * t * B) @ U
            if abs(a[k]) > 1e-15:
                U = _mat_exp(-1j * a[k] * t * A) @ U
        
        U_exact = _mat_exp(-1j * t * (A + B))
        err = _spectral_norm(U - U_exact)
        errors.append(err)
    
    return np.mean(errors)

def _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-10):
    """Pure NumPy Nelder-Mead simplex algorithm with bounds."""
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        x[i] += step
        simplex.append(x)
    
    # Evaluate initial simplex
    fvals = [objective(x) for x in simplex]
    
    # Sort
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    best_x = simplex[0]
    best_f = fvals[0]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for it in range(max_iter):
        if _remaining() < 10.0:
            break
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
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
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        
        # Re-sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0]
    
    return best_x, best_f

def coefficients(stages):
    """Optimize coefficients for the product formula."""
    stages = int(stages)
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: equal weights
    n = stages
    a0 = [1.0 / n] * n
    b0 = [1.0 / n] * n
    
    x0 = a0 + b0
    
    # Run Nelder-Mead optimization
    if _remaining() > 30.0:
        try:
            x_best, f_best = _nelder_mead(
                lambda x: _objective((list(x[:n]), list(x[n:])), A, B, times),
                x0,
                None,
                max_iter=300,
                tol=1e-12
            )
        except Exception:
            x_best = x0
    
    # Post-process: ensure constraints
    a = list(x_best[:n])
    b = list(x_best[n:])
    
    # Clip to bounds
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Normalize to sum = 1
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    else:
        a = [1.0 / n] * n
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    else:
        b = [1.0 / n] * n
    
    # Final clip and re-normalize
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    return a, b
