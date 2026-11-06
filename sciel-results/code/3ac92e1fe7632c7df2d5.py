import numpy as np
import time

def _construct_hamiltonians():
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    # Initialize A and B as zero matrices
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Basis states
    # We need to create the spin operators
    # Sx, Sy, Sz for each site
    
    # It is more efficient to construct the bond operators directly in the computational basis
    # Bond operator for site i, i+1:
    # H_bond = Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + anisotropy * Sz_i Sz_{i+1}
    # Field operator:
    # H_field = field * sum(Sz_i)
    
    # Precompute single site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_chain(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    # Build full operators for each site
    # This is slow for N=6 if done naively, but 64x64 is small.
    # Let's use a more direct bit-manipulation approach for speed if needed, but kron is fine for 64x64.
    
    # Actually, for N=6, 64x64 matrices.
    # We can build the full Sx, Sy, Sz for the whole chain.
    # Sx_total = sum_i Sx_i
    # But we need individual site operators to build bonds.
    
    # Let's build the operators for each site in the full space
    ops_x = []
    ops_y = []
    ops_z = []
    
    for i in range(N):
        # Operator for site i is identity on other sites
        # To be efficient, we can just build the full matrices once.
        pass
        
    # Efficient construction for N=6
    # Use bit operations to build the matrix elements?
    # Or just use kron. 6 kronecker products of 2x2 matrices.
    
    # Let's build the full Sx, Sy, Sz for each site
    # Sx_i = I x ... x Sx x ... x I
    # This requires 6 * 6 * 2x2 kron operations? No, just 6 full matrix constructions.
    
    Sx_sites = []
    Sy_sites = []
    Sz_sites = []
    
    for i in range(N):
        # Build Sx at site i
        ops = [identity] * N
        ops[i] = sx
        Sx_sites.append(kron_chain(ops))
        
        ops[i] = sy
        Sy_sites.append(kron_chain(ops))
        
        ops[i] = sz
        Sz_sites.append(kron_chain(ops))

    # Build A and B
    # Bonds: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds (A): (0,1), (2,3), (4,5)
    # Even bonds (B): (1,2), (3,4)
    
    # Field: sum(Sz_i) * field
    H_field = field * sum(Sz_sites)
    
    H_A_field_part = 0.5 * H_field
    H_B_field_part = 0.5 * H_field
    
    for i in [0, 2, 4]:
        # Bond i, i+1
        # Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + anisotropy * Sz_i Sz_{i+1}
        bond_op = (np.dot(Sx_sites[i], Sx_sites[i+1]) + 
                   np.dot(Sy_sites[i], Sy_sites[i+1]) + 
                   anisotropy * np.dot(Sz_sites[i], Sz_sites[i+1]))
        A += bond_op
        
    for i in [1, 3]:
        bond_op = (np.dot(Sx_sites[i], Sx_sites[i+1]) + 
                   np.dot(Sy_sites[i], Sy_sites[i+1]) + 
                   anisotropy * np.dot(Sz_sites[i], Sz_sites[i+1]))
        B += bond_op
        
    A += H_A_field_part
    B += H_B_field_part
    
    return A, B

def _compute_error(a, b, A, B, times):
    # a, b are lists of 12
    # U = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # Exact = exp(-i t (A+B))
    
    H = A + B
    errors = []
    for t in times:
        # Exact
        # Use scipy.linalg.expm if available, else use eigendecomp
        # Since SciPy is risky, use eigendecomp
        vals, vecs = np.linalg.eigh(H)
        # H is Hermitian
        # exp(-i t H) = vecs @ diag(exp(-i t vals)) @ vecs^H
        exp_vals = np.exp(-1j * t * vals)
        U_exact = vecs @ np.diag(exp_vals) @ vecs.conj().T
        
        # Product formula
        U_prod = np.eye(2**6, dtype=complex)
        
        # Precompute exponentials?
        # b_k * t * B and a_k * t * A
        # We can precompute the eigendecomp of A and B once outside the loop?
        # But t changes.
        # exp(-i c t H) = V diag(exp(-i c t lambda)) V^H
        
        # Let's precompute eigendecomp of A and B
        # This should be done outside _compute_error ideally, but for simplicity inside or passed.
        # For now, compute inside. 64x64 eig is fast.
        
        # Optimization: Precompute eigendecomp of A and B
        # But _compute_error is called many times.
        # We should pass A_eig, B_eig.
        
        # Let's refactor to pass eigendecoms.
        
        pass

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    if stages != 12:
        return [1.0/stages]*stages, [1.0/stages]*stages
        
    A, B = _construct_hamiltonians()
    
    # Precompute eigendecomp
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_H, vecs_H = np.linalg.eigh(A+B)
    
    times = [0.25, 0.5, 1.0]
    
    def cost_func(params):
        # params: 10 values [a1..a5, b1..b5]
        # Symmetric: a6 = 1 - sum(a1..a5)
        # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        
        a_uniq = params[0:5]
        b_uniq = params[5:10]
        
        a6 = 1.0 - sum(a_uniq)
        b6 = 1.0 - sum(b_uniq)
        
        a = [a_uniq[0], a_uniq[1], a_uniq[2], a_uniq[3], a_uniq[4], a6, a6, a_uniq[4], a_uniq[3], a_uniq[2], a_uniq[1], a_uniq[0]]
        b = [b_uniq[0], b_uniq[1], b_uniq[2], b_uniq[3], b_uniq[4], b6, b6, b_uniq[4], b_uniq[3], b_uniq[2], b_uniq[1], b_uniq[0]]
        
        # Check bounds? [-50, 50]
        # If any out of bounds, return large penalty
        if any(abs(x) > 50 for x in a+b):
            return 1e6
            
        total_err = 0.0
        for t in times:
            # U_exact
            exp_vals_H = np.exp(-1j * t * vals_H)
            U_exact = vecs_H @ np.diag(exp_vals_H) @ vecs_H.conj().T
            
            U_prod = np.eye(64, dtype=complex)
            
            for k in range(12):
                # exp(-i b_k t B)
                bk = b[k]
                if abs(bk) > 1e-10:
                    exp_vals_B = np.exp(-1j * bk * t * vals_B)
                    U_B = vecs_B @ np.diag(exp_vals_B) @ vecs_B.conj().T
                    U_prod = U_prod @ U_B
                
                # exp(-i a_k t A)
                ak = a[k]
                if abs(ak) > 1e-10:
                    exp_vals_A = np.exp(-1j * ak * t * vals_A)
                    U_A = vecs_A @ np.diag(exp_vals_A) @ vecs_A.conj().T
                    U_prod = U_prod @ U_A
            
            err = np.linalg.norm(U_prod - U_exact, 2)
            total_err += err
            
        return total_err / 3.0

    # Initial guess: 4th order Yoshida
    # Yoshida 4th order coefficients for 3-step scheme?
    # Let's use the coefficients from the current program as initial guess, they sum to 1.
    # a_init = [0.0555, 0.1508, 0.2204, 0.1931, 0.1234]
    # b_init = [0.0555, 0.1508, 0.2204, 0.1931, 0.1234]
    
    x0 = np.array([
        0.055501558671812,
        0.150779964897872,
        0.220446795412794,
        0.193080063904462,
        0.123413233389540,
        0.055501558671812,
        0.150779964897872,
        0.220446795412794,
        0.193080063904462,
        0.123413233389540
    ])
    
    # Simple Nelder-Mead implementation
    def nelder_mead(func, x0, max_iter=200, tol=1e-8):
        n = len(x0)
        # Simplex
        simplex = [x0.copy()]
        for i in range(n):
            point = x0.copy()
            point[i] = point[i] + 0.01
            simplex.append(point)
            
        fvals = [func(p) for p in simplex]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                gamma = 2.0
                x_e = centroid + gamma * (centroid - simplex[-1])
                f_e = func(x_e)
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
                rho = 0.5
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = func(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        
        return simplex[0], fvals[0]

    x_opt, f_opt = nelder_mead(cost_func, x0, max_iter=100)
    
    # Convert back to a, b
    a_uniq = x_opt[0:5]
    b_uniq = x_opt[5:10]
    a6 = 1.0 - sum(a_uniq)
    b6 = 1.0 - sum(b_uniq)
    
    a = [a_uniq[0], a_uniq[1], a_uniq[2], a_uniq[3], a_uniq[4], a6, a6, a_uniq[4], a_uniq[3], a_uniq[2], a_uniq[1], a_uniq[0]]
    b = [b_uniq[0], b_uniq[1], b_uniq[2], b_uniq[3], b_uniq[4], b6, b6, b_uniq[4], b_uniq[3], b_uniq[2], b_uniq[1], b_uniq[0]]
    
    # Final check and normalization
    a_sum = sum(a)
    b_sum = sum(b)
    a = [x/a_sum for x in a]
    b = [x/b_sum for x in b]
    
    return a, b
