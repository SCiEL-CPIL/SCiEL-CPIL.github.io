import numpy as np
import time
import scipy.optimize as opt

def _build_XXZ_6site():
    """Build the 6-site open XXZ Heisenberg chain Hamiltonian split into A and B."""
    anisotropy = 0.8
    field = 0.3
    J = 1.0
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    n = 6
    dim = 2 ** n
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n - 1):
        ops_sxx = [ident] * n
        ops_syy = [ident] * n
        ops_szz = [ident] * n
        ops_szz[i] = sz
        ops_szz[i+1] = sz
        ops_sxx[i] = sx
        ops_sxx[i+1] = sx
        ops_syy[i] = sy
        ops_syy[i+1] = sy
        
        bond = J * (kron_all(ops_sxx) + kron_all(ops_syy) + anisotropy * kron_all(ops_szz))
        
        if i % 2 == 0:
            A += bond
        else:
            B += bond
    
    for i in range(n):
        ops_sz = [ident] * n
        ops_sz[i] = sz
        field_term = field * kron_all(ops_sz)
        A += 0.5 * field_term
        B += 0.5 * field_term
        
    return A, B

def coefficients(stages):
    if stages != 12:
        return [1.0/stages]*stages, [1.0/stages]*stages

    t0 = time.time()
    deadline = t0 + 145
    
    A, B = _build_XXZ_6site()
    dim = A.shape[0]
    H = A + B
    
    # Precompute eigendecompositions
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    
    t_evals = [0.25, 0.5, 1.0]
    
    # Precompute U_exact for each t
    U_exacts = []
    for t in t_evals:
        U_exact = eigvecs_H @ np.diag(np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
        U_exacts.append(U_exact)
    
    def error_func(params):
        """
        params: 12 values for a1..a6 (symmetric) and 12 values for b1..b6 (symmetric)
        We parameterize 11 free a's and 11 free b's to satisfy sum=1.
        Actually, let's just take 12 a's and 12 b's and normalize?
        Normalization might break symmetry if not careful.
        Better: parameterize a1..a6, b1..b6.
        a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        sum(a) = 2 * sum(a1..a6) = 1 => sum(a1..a6) = 0.5
        Similarly for b.
        
        So we optimize 12 variables: a1..a6, b1..b6.
        """
        if time.time() > deadline:
            return 1e10
            
        a_half = params[:6]
        b_half = params[6:12]
        
        # Enforce sum constraint by scaling?
        # No, let's use a parameterization that naturally satisfies sum=1?
        # Or just penalize violation?
        # Simplest: a_full = 2 * a_half / sum(a_half) if sum!=0
        sum_a_half = sum(a_half)
        sum_b_half = sum(b_half)
        
        if abs(sum_a_half) < 1e-10 or abs(sum_b_half) < 1e-10:
            return 1e10
            
        a_full = 2.0 * np.array(a_half) / sum_a_half
        b_full = 2.0 * np.array(b_half) / sum_b_half
        
        a = list(a_full[:6]) + list(a_full[5::-1])
        b = list(b_full[:6]) + list(b_full[5::-1])
        
        # Check bounds
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10

        total_err = 0.0
        for i_t, t in enumerate(t_evals):
            U_exact = U_exacts[i_t]
            U_prod = np.eye(dim, dtype=complex)
            
            for k in range(12):
                # exp(-i b_k t B)
                expB = eigvecs_B @ np.diag(np.exp(-1j * b[k] * t * eigvals_B)) @ eigvecs_B.conj().T
                # exp(-i a_k t A)
                expA = eigvecs_A @ np.diag(np.exp(-1j * a[k] * t * eigvals_A)) @ eigvecs_A.conj().T
                U_prod = expB @ expA @ U_prod
                
            diff = U_prod - U_exact
            # Spectral norm
            s = np.linalg.svd(diff, compute_uv=False)
            total_err += s[0]
            
        return total_err / len(t_evals)

    # Initial guess: Uniform
    # a1..a6 sum to 0.5 => each 0.5/6
    init_a = [0.5/6.0] * 6
    init_b = [0.5/6.0] * 6
    init_params = init_a + init_b
    
    # Try inspiration-like: Strang
    # Strang: B A B ...
    # If we use symmetric a and b, it's harder to mimic exactly the "zero" pattern of inspiration
    # But we can try a few starts.
    
    starts = [
        init_params,
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.05, 0.05, 0.05, 0.05, 0.05, 0.25], # Random-ish
        [0.2, 0.1, 0.1, 0.1, 0.1, 0.0] + [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ]
    
    best_params = init_params
    best_err = error_func(init_params)
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = opt.minimize(error_func, start, method='Nelder-Mead', 
                               options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract best a and b
    a_half = best_params[:6]
    b_half = best_params[6:12]
    
    sum_a_half = sum(a_half)
    sum_b_half = sum(b_half)
    
    if abs(sum_a_half) < 1e-10: sum_a_half = 1.0
    if abs(sum_b_half) < 1e-10: sum_b_half = 1.0
    
    a_full = 2.0 * np.array(a_half) / sum_a_half
    b_full = 2.0 * np.array(b_half) / sum_b_half
    
    a = list(a_full[:6]) + list(a_full[5::-1])
    b = list(b_full[:6]) + list(b_full[5::-1])
    
    # Final normalization check
    sa = sum(a)
    sb = sum(b)
    if sa != 0: a = [x/sa for x in a]
    if sb != 0: b = [x/sb for x in b]
    
    # Clamp
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    return a, b
