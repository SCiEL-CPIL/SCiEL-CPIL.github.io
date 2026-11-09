"""Improved program using numerical optimization for product formula coefficients."""
import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain with anisotropy 0.8, field 0.3."""
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    anisotropy = 0.8
    field = 0.3
    
    # Build odd bonds (0,1), (2,3), (4,5) and even bonds (1,2), (3,4)
    # Each bond term: J*(sx⊗sx + sy⊗sy + anisotropy*sz⊗sz)/2 + field*sz/2
    # Split field half to A, half to B
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> go into A
    odd_bonds = [(0,1), (2,3), (4,5)]
    for i, j in odd_bonds:
        for pauli_x, pauli_y in [(sx, sx), (sy, sy), (sz, sz)]:
            coeff = anisotropy if pauli_x is sz else 1.0
            term = np.zeros((dim, dim), dtype=complex)
            ops = [I2]*N
            ops[i] = pauli_x
            ops[j] = pauli_y
            term = kron_all(ops)
            A += (coeff * term) / 2.0
        # Add half field on each site
        for site in [i, j]:
            ops = [I2]*N
            ops[site] = sz
            term = kron_all(ops)
            A += (field * term) / 4.0
    
    # Even bonds: (1,2), (3,4) -> go into B
    even_bonds = [(1,2), (3,4)]
    for i, j in even_bonds:
        for pauli_x, pauli_y in [(sx, sx), (sy, sy), (sz, sz)]:
            coeff = anisotropy if pauli_x is sz else 1.0
            term = np.zeros((dim, dim), dtype=complex)
            ops = [I2]*N
            ops[i] = pauli_x
            ops[j] = pauli_y
            term = kron_all(ops)
            B += (coeff * term) / 2.0
        # Add half field on each site
        for site in [i, j]:
            ops = [I2]*N
            ops[site] = sz
            term = kron_all(ops)
            B += (field * term) / 4.0
    
    return A, B

def _spectral_norm_error(a, b, A, B, times):
    """Compute mean spectral norm error over given times."""
    n = len(a)
    total_err = 0.0
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(n):
            # exp(-i * b_k * t * B)
            if abs(b[k]) > 1e-15:
                U = np.linalg.multi_dot([U, _expm_hermitian(B, -1j * b[k] * t)])
            # exp(-i * a_k * t * A)
            if abs(a[k]) > 1e-15:
                U = np.linalg.multi_dot([U, _expm_hermitian(A, -1j * a[k] * t)])
        
        # Exact propagator
        H = A + B
        U_exact = _expm_hermitian(H, -1j * t)
        
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(times)

def _expm_hermitian(H, coeff):
    """Compute exp(coeff * H) where H is Hermitian."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(coeff * eigvals)) @ eigvecs.conj().T

def _nelder_mead_simplex(objective, x0, max_iter=500, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
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
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    # Return best
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _normalize_and_clip(a, b, stages):
    """Ensure sums are 1 and values in [-50, 50]."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    # Clip to bounds
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Normalize sums to 1
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    
    if abs(a_sum) > 1e-15:
        a = a / a_sum
    else:
        a = np.ones(stages) / stages
    
    if abs(b_sum) > 1e-15:
        b = b / b_sum
    else:
        b = np.ones(stages) / stages
    
    # Re-clip after normalization
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    return a.tolist(), b.tolist()

def _objective_from_params(params, A, B, times):
    """Convert parameter vector to (a,b) and compute error."""
    stages = 12
    n = stages
    a = params[:n]
    b = params[n:2*n]
    
    a_list, b_list = _normalize_and_clip(a, b, stages)
    
    return _spectral_norm_error(a_list, b_list, A, B, times)

def coefficients(stages):
    """Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    if stages != 12:
        # Fallback for non-12 stages
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: Strang-like
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    x0 = a0 + b0
    
    best_a, best_b = a0, b0
    best_err = _objective_from_params(x0, A, B, times)
    
    # Nelder-Mead optimization
    try:
        x_best, f_best = _nelder_mead_simplex(
            lambda x: _objective_from_params(x, A, B, times),
            x0,
            max_iter=200,
            tol=1e-8
        )
        
        if f_best < best_err:
            best_err = f_best
            a_best, b_best = _normalize_and_clip(x_best[:12], x_best[12:24], 12)
            best_a, best_b = a_best, b_best
    except Exception:
        pass
    
    # Check time, do polish if time remains
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            x_polish, f_polish = _nelder_mead_simplex(
                lambda x: _objective_from_params(x, A, B, times),
                x_best,
                max_iter=100,
                tol=1e-10
            )
            if f_polish < best_err:
                best_err = f_polish
                a_polish, b_polish = _normalize_and_clip(x_polish[:12], x_polish[12:24], 12)
                best_a, best_b = a_polish, b_polish
        except Exception:
            pass
    
    return best_a, best_b
# EVOLVE-BLOCK-END
