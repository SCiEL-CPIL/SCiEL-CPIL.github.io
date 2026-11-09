"""
High-order product formula coefficients for Heisenberg chain time evolution.
Uses 24 free parameters with fast error evaluation and multi-start optimization.
"""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_interaction(i, j):
        ops = [eye] * N
        ops[i] = sx
        ops[j] = sx
        term1 = kron_all(ops)
        
        ops = [eye] * N
        ops[i] = sy
        ops[j] = sy
        term2 = kron_all(ops)
        
        ops = [eye] * N
        ops[i] = sz
        ops[j] = sz
        term3 = kron_all(ops)
        
        return term1 + term2 + lam * term3
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_interaction(i, j)
    for i in range(N):
        ops = [eye] * N
        ops[i] = sz
        A += (h/2) * kron_all(ops)
    
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in [(1,2), (3,4)]:
        B += bond_interaction(i, j)
    for i in range(N):
        ops = [eye] * N
        ops[i] = sz
        B += (h/2) * kron_all(ops)
    
    return A, B

def _precompute_exps(A, B, times, max_coeff=50.0):
    """
    Precompute exp(-i * c * A) and exp(-i * c * B) for a grid of coefficients.
    This allows fast lookup during optimization.
    """
    # We'll use a dictionary-based approach with caching
    # For efficiency, precompute for the specific times and coefficient ranges
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    
    return A_eigvals, A_eigvecs, B_eigvals, B_eigvecs

def _nelder_mead(func, x0, max_iter=200, tol=1e-10, deadline=None, initial_step=0.05):
    """Nelder-Mead optimizer with deadline guard."""
    n = len(x0)
    simplex = [x0.copy() for _ in range(n + 1)]
    simplex[0] = x0
    for i in range(1, n + 1):
        simplex[i] = x0.copy()
        simplex[i][i - 1] += initial_step

    fvals = [func(s) for s in simplex]

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        indices = np.argsort(fvals)
        simplex = [simplex[i] for i in indices]
        fvals = [fvals[i] for i in indices]

        if abs(fvals[0] - fvals[-1]) < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)

        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = func(xr)

        if fr < fvals[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = func(xe)
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
            if fr < fvals[-1]:
                xc = centroid + 0.5 * (centroid - simplex[-1])
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])

    return simplex[0], fvals[0]

def coefficients(stages):
    """
    Return (a, b) where each is a list of 12 floats summing to 1.
    Uses 24 free parameters with multi-start Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = 2**6
    
    # Precompute eigen-decompositions
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    
    # Precompute exact propagators at target times
    times = [0.25, 0.5, 1.0]
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    exact_props = []
    for t in times:
        exact = H_eigvecs @ np.diag(np.exp(-1j * t * H_eigvals)) @ H_eigvecs.conj().T
        exact_props.append(exact)
    
    identity = np.eye(dim, dtype=complex)
    
    # Cache for exponential matrices
    expA_cache = {}
    expB_cache = {}
    
    def expm_A(t_val):
        """exp(-i * t_val * A) with caching"""
        key = round(t_val, 10)
        if key not in expA_cache:
            expA_cache[key] = A_eigvecs @ np.diag(np.exp(-1j * t_val * A_eigvals)) @ A_eigvecs.conj().T
        return expA_cache[key]
    
    def expm_B(t_val):
        """exp(-i * t_val * B) with caching"""
        key = round(t_val, 10)
        if key not in expB_cache:
            expB_cache[key] = B_eigvecs @ np.diag(np.exp(-1j * t_val * B_eigvals)) @ B_eigvecs.conj().T
        return expB_cache[key]
    
    def compute_error(a, b):
        """Compute mean spectral-norm error."""
        total_err = 0.0
        for idx, t in enumerate(times):
            U = identity.copy()
            for k in range(12):
                bk_t = b[k] * t
                if abs(bk_t) > 1e-15:
                    expB = expm_B(bk_t)
                else:
                    expB = identity
                U = expB @ U
                
                ak_t = a[k] * t
                if abs(ak_t) > 1e-15:
                    expA = expm_A(ak_t)
                else:
                    expA = identity
                U = expA @ U
            
            diff = U - exact_props[idx]
            # Use frobenius norm as a faster proxy, then refine with spectral norm
            # Actually, we need spectral norm for accuracy
            err = np.linalg.norm(diff, 2)
            total_err += err
        return total_err / len(times)
    
    def objective(params):
        """
        params: 24-vector [a_1,...,a_12, b_1,...,b_12]
        Enforce sum(a)=1, sum(b)=1 by normalizing.
        """
        if time.time() > deadline:
            return 1e10
        
        a_raw = params[:12]
        b_raw = params[12:]
        
        # Normalize to sum=1
        a_sum = np.sum(a_raw)
        b_sum = np.sum(b_raw)
        
        if abs(a_sum) < 1e-15:
            a = np.ones(12) / 12
        else:
            a = a_raw / a_sum
        
        if abs(b_sum) < 1e-15:
            b = np.ones(12) / 12
        else:
            b = b_raw / b_sum
        
        # Clip to [-50, 50] and re-normalize
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        if abs(a_sum) > 1e-12:
            a = a / a_sum
        if abs(b_sum) > 1e-12:
            b = b / b_sum
        
        return compute_error(a, b)
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(24) / 12)
    
    # 2. BCH-inspired 4th order symmetric
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    c2 = (c1 - 1.0) / 2.0
    a_init = np.array([c1, c2, c2, c1, c1, c2, c2, c1, c1, c2, c2, c1])
    b_init = np.array([c2, c1, c1, c2, c2, c1, c1, c2, c2, c1, c1, c2])
    a_init = a_init / np.sum(a_init)
    b_init = b_init / np.sum(b_init)
    initializations.append(np.concatenate([a_init, b_init]))
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(5):
        a_rand = np.random.randn(12)
        b_rand = np.random.randn(12)
        a_rand = a_rand / np.sum(a_rand)
        b_rand = b_rand / np.sum(b_rand)
        initializations.append(np.concatenate([a_rand, b_rand]))
    
    # Evaluate all initializations
    best_x = None
    best_f = float('inf')
    for x_init in initializations:
        if time.time() > deadline - 5:
            break
        try:
            f = objective(x_init)
            if f < best_f:
                best_f = f
                best_x = x_init.copy()
        except Exception:
            pass
    
    # Run Nelder-Mead from best initialization with more iterations
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_x, res_f = _nelder_mead(objective, best_x, max_iter=300, tol=1e-12, 
                                         deadline=deadline - 5, initial_step=0.03)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x
        except Exception:
            pass
    
    # Additional random restarts if time permits
    np.random.seed(123)
    for _ in range(10):
        if time.time() > deadline - 8:
            break
        x_rand = best_x + 0.03 * np.random.randn(24)
        try:
            res_x, res_f = _nelder_mead(objective, x_rand, max_iter=80, tol=1e-11,
                                         deadline=deadline - 5, initial_step=0.02)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x
        except Exception:
            pass
    
    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]
    
    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum
    
    return [float(x) for x in a_full], [float(x) for x in b_full]
