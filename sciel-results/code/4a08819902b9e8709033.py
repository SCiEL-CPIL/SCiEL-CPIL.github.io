import numpy as np
import time
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Sites 0..5, bonds: (0,1) odd, (1,2) even, (2,3) odd, (3,4) even, (4,5) odd.
    A = sum of odd-bond terms + half field
    B = sum of even-bond terms + half field
    """
    N = 6
    gamma = 0.8
    h = 0.3
    dim = 2 ** N
    
    def site_op(op, site):
        """Apply single-site operator `op` (2x2) at `site`."""
        result = np.zeros((dim, dim))
        for i in range(dim):
            # Get bit representation
            bits = [(i >> j) & 1 for j in range(N)]
            # Apply op at site
            out_val = 0
            for col in range(2):
                if op[bits[site], col] != 0:
                    new_bits = bits[:]
                    new_bits[site] = col
                    new_i = sum(new_bits[j] << j for j in range(N))
                    out_val += op[bits[site], col] * (1 << 0)  # placeholder
            # Better approach: use kron
        # Use kron approach
        ops = [np.eye(2)] * N
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sx2 = 0.5 * sx
    sy2 = 0.5 * sy
    sz2 = 0.5 * sz
    
    def bond_op(i, j):
        """XXZ bond term between sites i and j: sx_i sx_j + sy_i sy_j + gamma sz_i sz_j"""
        term = np.zeros((dim, dim), dtype=complex)
        # sx_i sx_j
        op_i = [np.eye(2)] * N
        op_i[i] = sx2
        op_j = [np.eye(2)] * N
        op_j[j] = sx2
        # We need to apply both simultaneously - use kron
        ops = [np.eye(2)] * N
        ops[i] = sx2
        ops[j] = sx2
        t = ops[0]
        for op in ops[1:]:
            t = np.kron(t, op)
        term += t
        
        # sy_i sy_j
        ops = [np.eye(2)] * N
        ops[i] = sy2
        ops[j] = sy2
        t = ops[0]
        for op in ops[1:]:
            t = np.kron(t, op)
        term += t
        
        # gamma sz_i sz_j
        ops = [np.eye(2)] * N
        ops[i] = sz2
        ops[j] = sz2
        t = ops[0]
        for op in ops[1:]:
            t = np.kron(t, op)
        term += gamma * t
        
        return term
    
    # Odd bonds: (0,1), (2,3), (4,5)
    A = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_op(i, j)
    
    # Even bonds: (1,2), (3,4)
    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(1, 2), (3, 4)]:
        B += bond_op(i, j)
    
    # Uniform field: h * sum sz_i, split half to A, half to B
    field = np.zeros((dim, dim), dtype=complex)
    for site in range(N):
        ops = [np.eye(2)] * N
        ops[site] = sz2
        t = ops[0]
        for op in ops[1:]:
            t = np.kron(t, op)
        field += t
    
    A += 0.5 * h * field
    B += 0.5 * h * field
    
    return A, B

def _compute_error(a, b, A, B, exact_propagators):
    """Compute mean spectral norm error over t = 0.25, 0.5, 1.0."""
    errors = []
    for k, t in enumerate([0.25, 0.5, 1.0]):
        U = np.eye(2**6, dtype=complex)
        for idx in range(12):
            if abs(b[idx]) > 1e-15:
                U = expm(-1j * b[idx] * t * B) @ U
            if abs(a[idx]) > 1e-15:
                U = expm(-1j * a[idx] * t * A) @ U
        err = np.linalg.norm(U - exact_propagators[k], 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    """Optimized 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = 2**6
    
    # Precompute exact propagators
    H = A + B
    exact_propagators = [expm(-1j * t * H) for t in [0.25, 0.5, 1.0]]
    
    # Initial guess: 6th-order symmetric splitting coefficients
    # For a 12-stage formula with palindromic symmetry:
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # with sum(a)=1, sum(b)=1
    
    # 6th-order Yoshida coefficients (for 2s+1 stages, s=3 gives 7 stages)
    # For 12 stages, we can use a 6th-order 12-stage scheme
    # Let's use known optimal coefficients from literature
    
    # Start with a reasonable symmetric initialization
    # Using 4th-order coefficients as starting point, then optimize
    # 4th-order 4-stage: a = [1/4, 1/2, 1/4], b = [0, 1, 0] per substep
    # For 12 stages with 3 substeps of 4 stages each:
    # But let's just start with equal weights and optimize
    
    # Palindromic parameterization: 6 free params for a, 6 for b
    # a_free = [a1, a2, a3, a4, a5, a6], a7=a6, ..., a12=a1
    # sum = 2*(a1+a2+a3+a4+a5+a6) = 1 => a6 = 0.5 - (a1+a2+a3+a4+a5)
    # Similarly for b
    
    def params_to_ab(pa, pb):
        """Convert 6+6 free params to full 12+12 arrays with palindromic symmetry and sum=1."""
        # pa = [a1, a2, a3, a4, a5, a6]
        # a6 is determined: 2*(a1+...+a6) = 1
        a1, a2, a3, a4, a5, a6 = pa
        a_full = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        # Adjust to ensure sum = 1
        s = sum(a_full)
        if abs(s) > 1e-15:
            a_full = [x / s for x in a_full]
        
        b1, b2, b3, b4, b5, b6 = pb
        b_full = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        s = sum(b_full)
        if abs(s) > 1e-15:
            b_full = [x / s for x in b_full]
        
        return a_full, b_full
    
    def objective(params):
        pa = params[:6]
        pb = params[6:12]
        a, b = params_to_ab(pa, pb)
        return _compute_error(a, b, A, B, exact_propagators)
    
    # Initial guess: start with roughly equal weights
    # For 12-stage palindromic: a_i ~ 1/12 each
    x0 = np.array([1.0/12]*6 + [1.0/12]*6)
    
    best_x = x0.copy()
    best_err = objective(x0)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        a, b = params_to_ab(x0[:6], x0[6:])
        return a, b
    
    # Use Nelder-Mead with reasonable settings
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result.fun < best_err:
        best_err = result.fun
        best_x = result.x
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a second round with tighter tolerances from the best point
        result2 = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result2.fun < best_err:
            best_err = result2.fun
            best_x = result2.x
    
    # Clip to [-50, 50]
    a, b = params_to_ab(best_x[:6], best_x[6:])
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Ensure sums are exactly 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    
    return a, b
