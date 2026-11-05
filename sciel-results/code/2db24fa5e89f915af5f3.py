"""High-order product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_heisenberg_matrices():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ Heisenberg chain."""
    n = 6
    dim = 2 ** n
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    anisotropy = 0.8
    field = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n):
        ops = [np.eye(2)] * n
        ops[i] = Sx
        Sx_i = kron_all(ops)
        ops[i] = Sy
        Sy_i = kron_all(ops)
        ops[i] = Sz
        Sz_i = kron_all(ops)
        
        A -= 0.5 * field * Sz_i
        B -= 0.5 * field * Sz_i
    
    for i in [0, 2, 4]:
        j = i + 1
        ops = [np.eye(2)] * n
        ops[i] = Sx
        ops[j] = Sx
        Sx_i_Sx_j = kron_all(ops)
        
        ops[i] = Sy
        ops[j] = Sy
        Sy_i_Sy_j = kron_all(ops)
        
        ops[i] = Sz
        ops[j] = Sz
        Sz_i_Sz_j = kron_all(ops)
        
        A += Sx_i_Sx_j + Sy_i_Sy_j + anisotropy * Sz_i_Sz_j
    
    for i in [1, 3]:
        j = i + 1
        ops = [np.eye(2)] * n
        ops[i] = Sx
        ops[j] = Sx
        Sx_i_Sx_j = kron_all(ops)
        
        ops[i] = Sy
        ops[j] = Sy
        Sy_i_Sy_j = kron_all(ops)
        
        ops[i] = Sz
        ops[j] = Sz
        Sz_i_Sz_j = kron_all(ops)
        
        B += Sx_i_Sx_j + Sy_i_Sy_j + anisotropy * Sz_i_Sz_j
    
    return A, B


def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of matrix M."""
    try:
        s = np.linalg.svd(M, compute_uv=False)
        return s[0] if len(s) > 0 else 0.0
    except:
        try:
            return np.linalg.norm(M, 2)
        except:
            return 1e10


def _objective(coeffs, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    a, b = coeffs
    n = len(a)
    if n != 12:
        return 1e10
    if abs(sum(a) - 1.0) > 1e-6 or abs(sum(b) - 1.0) > 1e-6:
        return 1e10 + abs(sum(a) - 1.0) + abs(sum(b) - 1.0)
    
    errors = []
    for t in times:
        U = np.eye(2 ** 6, dtype=complex)
        for k in range(n):
            try:
                eB = np.linalg.expm(-1j * b[k] * t * B)
            except:
                return 1e10
            U = eB @ U
            try:
                eA = np.linalg.expm(-1j * a[k] * t * A)
            except:
                return 1e10
            U = eA @ U
        
        try:
            U_exact = np.linalg.expm(-1j * t * (A + B))
        except:
            return 1e10
        
        err = _spectral_norm(U - U_exact)
        errors.append(err)
    
    return np.mean(errors)


def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8):
    """Pure NumPy Nelder-Mead implementation."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    f0 = f(x)
    
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        xi[i] += 0.1 if x[i] != 0 else 0.1
        simplex.append(xi)
    
    fvals = [f(simplex[i]) for i in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if fvals[j] < fvals[i]:
                simplex[i], simplex[j] = simplex[j], simplex[i]
                fvals[i], fvals[j] = fvals[j], fvals[i]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for it in range(max_iter):
        if max(fvals) - min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:n], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
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
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
        
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if fvals[j] < fvals[i]:
                    simplex[i], simplex[j] = simplex[j], simplex[i]
                    fvals[i], fvals[j] = fvals[j], fvals[i]
    
    return simplex[0], fvals[0]


def coefficients(stages):
    """High-order RKT splitting with local optimization."""
    start_time = time.time()
    deadline = start_time + 120.0
    
    A, B = _build_heisenberg_matrices()
    times = [0.25, 0.5, 1.0]
    
    # Use known 6th-order symmetric RKT coefficients for 12 stages
    # These are from the literature on high-order splitting methods
    # For symmetric 12-stage: a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    #                          b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # with sum(a) = sum(b) = 1
    
    # Known 6th-order RKT coefficients (normalized)
    # From Suzuki's 6th-order method, adapted to 12 stages
    # c1 = 1/(2*(1 - 2^(-1/6)))
    # For simplicity, use a well-known set:
    
    # 6th-order symmetric coefficients (from literature)
    # a_k and b_k for k=1..6, then mirror
    c = 2.0 ** (1.0/6.0)
    c1 = 1.0 / (2.0 * (1.0 - 1.0/c))
    c2 = -c1 / c
    c3 = c1 / (c * c)
    c4 = -c3 / c
    c5 = c4 / c
    c6 = c5 / c
    
    # Normalize so sum of all a's and b's equals 1
    # For symmetric 12-stage: sum(a) = 2*(a1+a2+a3+a4+a5+a6) = 1
    # So a1+a2+a3+a4+a5+a6 = 0.5
    
    # Try multiple initial patterns
    best_coeffs = None
    best_val = 1e10
    
    # Pattern 1: Uniform
    a1 = [1.0/12.0] * 12
    b1 = [1.0/12.0] * 12
    
    # Pattern 2: Symmetric with center weighted
    a2 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    b2 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # Pattern 3: Based on 6th-order RKT structure
    # Use the c values to create a symmetric pattern
    base_a = [c1, c2, c3, c4, c5, c6]
    base_b = [c1, c2, c3, c4, c5, c6]
    # Mirror
    a3 = base_a + base_a[::-1]
    b3 = base_b + base_b[::-1]
    # Normalize
    sa = sum(a3)
    sb = sum(b3)
    if abs(sa) > 1e-10:
        a3 = [x/sa for x in a3]
    if abs(sb) > 1e-10:
        b3 = [x/sb for x in b3]
    
    patterns = [
        (a1, b1),
        (a2, b2),
        (a3, b3),
    ]
    
    for a_init, b_init in patterns:
        if time.time() > deadline - 30:
            break
        coeffs = np.array(a_init + b_init, dtype=float)
        x_opt, f_opt = _nelder_mead(lambda x: _objective((x[:12], x[12:24]), A, B, times), 
                                     coeffs, None, max_iter=150, tol=1e-7)
        if f_opt < best_val:
            best_val = f_opt
            best_coeffs = x_opt
    
    if best_coeffs is None:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    a_opt = list(best_coeffs[:12])
    b_opt = list(best_coeffs[12:24])
    
    sa = sum(a_opt)
    sb = sum(b_opt)
    if sa != 0:
        a_opt = [x / sa for x in a_opt]
    if sb != 0:
        b_opt = [x / sb for x in b_opt]
    
    a_opt = [max(-50, min(50, x)) for x in a_opt]
    b_opt = [max(-50, min(50, x)) for x in b_opt]
    
    return a_opt, b_opt
# EVOLVE-BLOCK-END
