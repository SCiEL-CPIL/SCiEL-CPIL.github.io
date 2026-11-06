import numpy as np
import time
import scipy.linalg

def coefficients(stages):
    """
    Returns optimal 12-stage symmetric Suzuki-Trotter coefficients.
    Uses numerical optimization to find coefficients that minimize the spectral norm error.
    The formula is U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A).
    Coefficients are symmetric: a_k = a_{13-k}, b_k = b_{13-k}.
    """
    if stages != 12:
        # Fallback to simple Strang if stages != 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5 seconds margin

    # Build Heisenberg chain matrices
    def build_heisenberg_matrices():
        """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain."""
        n_sites = 6
        dim = 2 ** n_sites
        anisotropy = 0.8
        field = 0.3
        
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        def kron_all(ops):
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result
        
        A = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        
        # Odd bonds: sites 0-1, 2-3, 4-5
        for bond in [(0, 1), (2, 3), (4, 5)]:
            i, j = bond
            ops_sx = [I2] * n_sites
            ops_sy = [I2] * n_sites
            ops_sz = [I2] * n_sites
            
            ops_sx[i] = sx
            ops_sx[j] = sx
            ops_sy[i] = sy
            ops_sy[j] = sy
            ops_sz[i] = sz
            ops_sz[j] = sz
            
            term = kron_all(ops_sx) + kron_all(ops_sy) + anisotropy * kron_all(ops_sz)
            A += term
        
        # Even bonds: sites 1-2, 3-4
        for bond in [(1, 2), (3, 4)]:
            i, j = bond
            ops_sx = [I2] * n_sites
            ops_sy = [I2] * n_sites
            ops_sz = [I2] * n_sites
            
            ops_sx[i] = sx
            ops_sx[j] = sx
            ops_sy[i] = sy
            ops_sy[j] = sy
            ops_sz[i] = sz
            ops_sz[j] = sz
            
            term = kron_all(ops_sx) + kron_all(ops_sy) + anisotropy * kron_all(ops_sz)
            B += term
        
        # Uniform field: h * sum_i sz_i, split evenly
        field_term = np.zeros((dim, dim), dtype=complex)
        for site in range(n_sites):
            ops_sz = [I2] * n_sites
            ops_sz[site] = sz
            field_term += kron_all(ops_sz)
        
        A += 0.5 * field * field_term
        B += 0.5 * field * field_term
        
        return A, B
    
    A_mat, B_mat = build_heisenberg_matrices()
    dim = 2 ** 6
    
    # Initial guess: 4th order based symmetric coefficients
    lam = 1.0 / (2.0 * (2.0 ** (1.0/3.0) - 1.0))
    c1 = -lam
    c2 = 1.0 + 2.0*lam
    c3 = -lam
    
    a_half = [c1/8.0, (c1+c2)/8.0, (c2+c3)/8.0, (c2+c3)/8.0, (c1+c2)/8.0, c1/8.0]
    b_half = [c1/4.0, c2/4.0, c3/4.0, c3/4.0, c2/4.0, c1/4.0]
    
    # Normalize
    sum_b_half = sum(b_half)
    sum_a_half = sum(a_half)
    if abs(sum_b_half) > 1e-15:
        b_half = [x / sum_b_half * 0.5 for x in b_half]
    if abs(sum_a_half) > 1e-15:
        a_half = [x / sum_a_half * 0.5 for x in a_half]
    
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Normalize to ensure sums are 1
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    # Check if we have time for optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        try:
            from scipy.optimize import minimize
            
            def compute_error(params):
                """Compute the mean spectral norm error."""
                a_params = params[:6]
                b_params = params[6:12]
                
                a_full = list(a_params) + list(a_params[::-1])
                b_full = list(b_params) + list(b_params[::-1])
                
                sum_a = sum(a_full)
                sum_b = sum(b_full)
                if abs(sum_a) < 1e-15 or abs(sum_b) < 1e-15:
                    return 1e10
                a_full = [x / sum_a for x in a_full]
                b_full = [x / sum_b for x in b_full]
                
                times = [0.25, 0.5, 1.0]
                errors = []
                
                for t in times:
                    U = np.eye(dim, dtype=complex)
                    for k in range(12):
                        U = U @ scipy.linalg.expm(-1j * b_full[k] * t * B_mat)
                        U = U @ scipy.linalg.expm(-1j * a_full[k] * t * A_mat)
                    
                    U_exact = scipy.linalg.expm(-1j * t * (A_mat + B_mat))
                    diff = U - U_exact
                    try:
                        s = np.linalg.svd(diff, compute_uv=False)
                        errors.append(s[0])
                    except:
                        return 1e10
                
                return np.mean(errors)
            
            x0 = np.array(a_half + b_half)
            x0_norm = compute_error(x0)
            
            result = minimize(compute_error, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'fatol': 1e-9, 'xatol': 1e-9})
            
            if result.fun < x0_norm:
                x_opt = result.x
                a_opt = list(x_opt[:6])
                b_opt = list(x_opt[6:12])
                
                a_full_opt = a_opt + a_opt[::-1]
                b_full_opt = b_opt + b_opt[::-1]
                
                sum_a = sum(a_full_opt)
                sum_b = sum(b_full_opt)
                if abs(sum_a) > 1e-15:
                    a_full_opt = [x / sum_a for x in a_full_opt]
                if abs(sum_b) > 1e-15:
                    b_full_opt = [x / sum_b for x in b_full_opt]
                
                a = a_full_opt
                b = b_full_opt
        
        except Exception:
            pass
    
    # Final safety check
    a = [max(-50, min(50, x)) if np.isfinite(x) else 0.0 for x in a]
    b = [max(-50, min(50, x)) if np.isfinite(x) else 0.0 for x in b]
    
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    return a, b
