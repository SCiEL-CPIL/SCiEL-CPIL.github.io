import numpy as np
import time
import sys

def coefficients(stages):
    if stages != 12:
        return [1.0/stages]*stages, [1.0/stages]*stages
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    try:
        import scipy.optimize as opt
        use_scipy = True
    except ImportError:
        use_scipy = False

    # Construct A and B
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    # Spin operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_chain(ops):
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out
        
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field part: half in A, half in B
    # H_field = field * sum(Sz_i)
    # We add field/2 * sum(Sz_i) to both A and B?
    # "each including half of the uniform field"
    # So A has odd bonds + h/2 * sum Sz, B has even bonds + h/2 * sum Sz.
    
    field_term = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        ops = [identity]*N
        ops[i] = sz
        field_term += kron_chain(ops)
        
    half_field = 0.5 * field * field_term
    
    # Bonds
    # Odd bonds: (0,1), (2,3), (4,5) -> indices 0, 2, 4
    # Even bonds: (1,2), (3,4) -> indices 1, 3
    # Wait, 6 sites: 0-1, 1-2, 2-3, 3-4, 4-5.
    # Odd bonds (if we start counting 1): 1-2, 3-4, 5-6?
    # Standard split:
    # A: bonds (0,1), (2,3), (4,5)
    # B: bonds (1,2), (3,4)
    
    def add_bond(mat, i, j):
        # Sx_i Sx_j + Sy_i Sy_j + anisotropy Sz_i Sz_j
        ops_xx = [identity]*N
        ops_xx[i] = sx
        ops_xx[j] = sx
        mat += kron_chain(ops_xx)
        
        ops_yy = [identity]*N
        ops_yy[i] = sy
        ops_yy[j] = sy
        mat += kron_chain(ops_yy)
        
        ops_zz = [identity]*N
        ops_zz[i] = sz
        ops_zz[j] = sz
        mat += anisotropy * kron_chain(ops_zz)

    # A: Odd bonds (0,1), (2,3), (4,5)
    A += half_field
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # B: Even bonds (1,2), (3,4)
    B += half_field
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    # Exact propagator
    H = A + B
    
    def calc_error(a, b):
        # a, b are lists of 12
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B) exp(-i a_k t A)
            # Note: order is B then A?
            # "prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
            # So U = (B_1 A_1) (B_2 A_2) ... (B_12 A_12)
            # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
            pass # Placeholder, will be implemented in cost func
        
        # We need to compute for t = 0.25, 0.5, 1.0
        err = 0.0
        for t in [0.25, 0.5, 1.0]:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                U = U @ np.linalg.matrix_exp(-1j * b[k] * t * B) @ np.linalg.matrix_exp(-1j * a[k] * t * A)
            U_exact = np.linalg.matrix_exp(-1j * t * H)
            # Spectral norm
            # SVD
            s = np.linalg.svd(U - U_exact, compute_uv=False)
            err += s[0]
        return err / 3.0

    if use_scipy:
        # Initial guess: uniform
        x0 = [1.0/12] * 22
        # x = [a1..a11, b1..b11]
        # a12 = 1 - sum(a1..a11)
        # b12 = 1 - sum(b1..b11)
        
        def cost_func(x):
            a = list(x[:11])
            b = list(x[11:])
            a.append(1.0 - sum(a))
            b.append(1.0 - sum(b))
            return calc_error(a, b)
            
        # Optimize
        # Nelder-Mead is robust for unconstrained (we parameterized constraints)
        # Bounds? Coeffs should be in [-50, 50].
        # We can use L-BFGS-B with bounds on the 22 variables.
        bounds = [(-50.0, 50.0)] * 22
        
        # Run optimization
        res = opt.minimize(cost_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
        
        if time.time() > deadline:
             # Fallback to best found so far or initial
             pass
             
        x_opt = res.x
        a = list(x_opt[:11])
        b = list(x_opt[11:])
        a.append(1.0 - sum(a))
        b.append(1.0 - sum(b))
        
        # Check validity
        if all(np.isfinite(a)) and all(np.isfinite(b)) and all(np.abs(x) <= 50 for x in a+b):
            return a, b
        else:
            # Fallback
            return [1.0/12]*12, [1.0/12]*12
    else:
        # Fallback: 4th order Yoshida (approximate)
        # c1 = 1/(2*(1 - 2**(-1/3)))
        # c2 = -2**(-1/3)/(2*(1 - 2**(-1/3)))
        # This is for 3 stages.
        # We just return uniform for now if scipy fails, or a better hardcoded set.
        return [1.0/12]*12, [1.0/12]*12
