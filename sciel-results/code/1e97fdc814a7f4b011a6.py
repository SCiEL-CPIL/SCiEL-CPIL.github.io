"""Symmetric product formula optimization for Heisenberg chain time evolution."""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    
    The full Hamiltonian H = A + B where:
    - A = sum over odd bonds (1-2, 3-4, 5-6) of XXZ interaction + half uniform field
    - B = sum over even bonds (2-3, 4-5) of XXZ interaction + half uniform field
    
    XXZ interaction on bond (i, i+1): Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + lambda Sz_i Sz_{i+1}
    Uniform field: h * sum_i Sz_i, split as h/2 in A and h/2 in B
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    # Pauli matrices / spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Identity for 2^N dimensional space
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply operator op on given site (0-indexed) to full 2^N space."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    # Odd bonds: (0,1), (2,3), (4,5)
    for i in range(0, N, 2):
        j = i + 1
        A += op_on_site(sx, i) @ op_on_site(sx, j)
        A += op_on_site(sy, i) @ op_on_site(sy, j)
        A += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    # Half field
    for i in range(N):
        A += (h/2) * op_on_site(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    # Even bonds: (1,2), (3,4)
    for i in range(1, N-1, 2):
        j = i + 1
        B += op_on_site(sx, i) @ op_on_site(sx, j)
        B += op_on_site(sy, i) @ op_on_site(sy, j)
        B += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    # Half field
    for i in range(N):
        B += (h/2) * op_on_site(sz, i)
    
    return A, B

# Cache for eigen-decompositions
_eig_cache = {}

def _get_eig(A, B):
    """Get eigen-decompositions of A, B, and A+B, cached."""
    key = (id(A), id(B))
    if key not in _eig_cache:
        wA, vA = np.linalg.eigh(A)
        wB, vB = np.linalg.eigh(B)
        H = A + B
        wH, vH = np.linalg.eigh(H)
        _eig_cache[key] = (wA, vA, wB, vB, wH, vH)
    return _eig_cache[key]

def _compute_error(a, b, A, B):
    """Compute mean spectral-norm error at t = 0.25, 0.5, 1.0."""
    wA, vA, wB, vB, wH, vH = _get_eig(A, B)
    
    errors = []
    for t in [0.25, 0.5, 1.0]:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Use eigen-decompositions for efficiency
        U = np.eye(2**6, dtype=complex)
        
        for k in range(len(a)):
            # exp(-i a_k t A)
            expA = vA @ np.diag(np.exp(-1j * a[k] * t * wA)) @ vA.conj().T
            U = U @ expA
            # exp(-i b_k t B)
            expB = vB @ np.diag(np.exp(-1j * b[k] * t * wB)) @ vB.conj().T
            U = U @ expB
        
        # Exact: exp(-i t (A+B))
        U_exact = vH @ np.diag(np.exp(-1j * t * wH)) @ vH.conj().T
        
        diff = U - U_exact
        # Spectral norm = largest singular value
        err = np.linalg.norm(diff, 2)
        errors.append(err)
    
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter=200, tol=1e-10):
    """Simple Nelder-Mead optimizer."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += 0.1 if x0[i] == 0 else 0.1 * x0[i]
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    
    for it in range(max_iter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
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
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                    continue
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                    continue
    
    # Return best point
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def coefficients(stages):
    """Optimize 12-stage product formula using symmetric local search."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    
    # For time-symmetric product formulas, we expect:
    # a_k = a_{13-k} and b_k = b_{13-k}
    # This reduces free parameters from 22 to 10 (5 for a, 5 for b)
    # Plus we need to enforce sum(a)=1 and sum(b)=1
    
    # Symmetric parameterization:
    # a = [x1, x2, x3, x4, x5, x6, x5, x4, x3, x2, x1, x0] where x0 is determined by sum
    # Actually, for 12 stages with symmetry a_k = a_{13-k}:
    # a[0]=a[11], a[1]=a[10], a[2]=a[9], a[3]=a[8], a[4]=a[7], a[5]=a[6]
    # So 6 free parameters for a, but sum constraint reduces to 5 free
    
    def _eval_symmetric(pa, pb):
        """
        pa: 5 values for a[0..4], a[5] determined by sum
        pb: 5 values for b[0..4], b[5] determined by sum
        
        Symmetric: a[5]=a[6], a[4]=a[7], a[3]=a[8], a[2]=a[9], a[1]=a[10], a[0]=a[11]
        Sum constraint: 2*(pa[0]+pa[1]+pa[2]+pa[3]+pa[4]) + 2*a5 = 1
        So a5 = (1 - 2*sum(pa[:5])) / 2
        """
        a = [0.0] * 12
        b = [0.0] * 12
        
        # a[5] = a[6]
        a5 = (1.0 - 2.0 * sum(pa[:5])) / 2.0
        
        a[0] = a[11] = pa[0]
        a[1] = a[10] = pa[1]
        a[2] = a[9] = pa[2]
        a[3] = a[8] = pa[3]
        a[4] = a[7] = pa[4]
        a[5] = a[6] = a5
        
        # b[5] = b[6]
        b5 = (1.0 - 2.0 * sum(pb[:5])) / 2.0
        
        b[0] = b[11] = pb[0]
        b[1] = b[10] = pb[1]
        b[2] = b[9] = pb[2]
        b[3] = b[8] = pb[3]
        b[4] = b[7] = pb[4]
        b[5] = b[6] = b5
        
        # Clamp
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        
        return _compute_error(a, b, A, B)
    
    # Initial guess: uniform
    pa0 = [1.0/12] * 5
    pb0 = [1.0/12] * 5
    
    best_pa = list(pa0)
    best_pb = list(pb0)
    best_err = _eval_symmetric(pa0, pb0)
    
    # Local search: coordinate descent with adaptive step size
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try perturbing each parameter in pa
            for i in range(5):
                for sign in [1, -1]:
                    trial_pa = list(best_pa)
                    trial_pa[i] += sign * step
                    trial_pb = list(best_pb)
                    
                    err = _eval_symmetric(trial_pa, trial_pb)
                    if err < best_err - 1e-12:
                        best_pa = trial_pa
                        best_pb = trial_pb
                        best_err = err
                        improved = True
            
            # Try perturbing each parameter in pb
            for i in range(5):
                for sign in [1, -1]:
                    trial_pa = list(best_pa)
                    trial_pb = list(best_pb)
                    trial_pb[i] += sign * step
                    
                    err = _eval_symmetric(trial_pa, trial_pb)
                    if err < best_err - 1e-12:
                        best_pa = trial_pa
                        best_pb = trial_pb
                        best_err = err
                        improved = True
    
    # Build final a and b from best parameters
    a = [0.0] * 12
    b = [0.0] * 12
    
    a5 = (1.0 - 2.0 * sum(best_pa[:5])) / 2.0
    a[0] = a[11] = best_pa[0]
    a[1] = a[10] = best_pa[1]
    a[2] = a[9] = best_pa[2]
    a[3] = a[8] = best_pa[3]
    a[4] = a[7] = best_pa[4]
    a[5] = a[6] = a5
    
    b5 = (1.0 - 2.0 * sum(best_pb[:5])) / 2.0
    b[0] = b[11] = best_pb[0]
    b[1] = b[10] = best_pb[1]
    b[2] = b[9] = best_pb[2]
    b[3] = b[8] = best_pb[3]
    b[4] = b[7] = best_pb[4]
    b[5] = b[6] = b5
    
    # Final clamp
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Ensure sum is exactly 1 (fix any floating point issues)
    a_sum = sum(a)
    if abs(a_sum - 1.0) > 1e-10:
        # Distribute correction evenly
        corr = (1.0 - a_sum) / 12.0
        a = [x + corr for x in a]
    
    b_sum = sum(b)
    if abs(b_sum - 1.0) > 1e-10:
        corr = (1.0 - b_sum) / 12.0
        b = [x + corr for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
