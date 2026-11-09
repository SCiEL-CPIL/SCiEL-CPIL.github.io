"""Product-formula coefficient optimizer using Suzuki-Fröhlich high-order splitting."""
import numpy as np
import time
import sys

_DEADLINE = 135.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
    s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
    
    def bond_operator(i, j):
        ops_splus_sminus = [identity] * N
        ops_splus_sminus[i] = s_plus
        ops_splus_sminus[j] = s_minus
        term1 = 0.5 * kron_all(ops_splus_sminus)
        
        ops_sminus_splus = [identity] * N
        ops_sminus_splus[i] = s_minus
        ops_sminus_splus[j] = s_plus
        term2 = 0.5 * kron_all(ops_sminus_splus)
        
        ops_sz_sz = [identity] * N
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        term3 = delta * kron_all(ops_sz_sz)
        
        return term1 + term2 + term3
    
    # Odd bonds: (1,2) and (3,4) -> A
    A += bond_operator(1, 2)
    A += bond_operator(3, 4)
    
    # Even bonds: (0,1), (2,3), (4,5) -> B
    B += bond_operator(0, 1)
    B += bond_operator(2, 3)
    B += bond_operator(4, 5)
    
    # Uniform field split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B


def _precompute_eigs(A, B, H_full):
    """Precompute eigen-decompositions for fast matrix exponential."""
    eigA_vals, eigA_vecs = np.linalg.eigh(A)
    eigB_vals, eigB_vecs = np.linalg.eigh(B)
    eigH_vals, eigH_vecs = np.linalg.eigh(H_full)
    return eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, eigH_vals, eigH_vecs


def _expm_from_eig(eigvals, eigvecs, coeff):
    """Compute exp(coeff * H) given eigen-decomposition."""
    if abs(coeff) < 1e-15:
        return np.eye(len(eigvals), dtype=complex)
    return eigvecs @ np.diag(np.exp(coeff * eigvals)) @ eigvecs.conj().T


def _compute_error_fast(a, b, eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                         eigH_vals, eigH_vecs, t_values, dim):
    """Compute mean spectral norm error using precomputed eigen-decompositions."""
    I = np.eye(dim, dtype=complex)
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = I.copy()
        
        for k in range(len(a)):
            # U = exp(-i b_k t B) @ U @ exp(-i a_k t A)
            # Actually the formula is: prod_k exp(-i b_k t B) exp(-i a_k t A)
            # So U = U @ exp(-i a_k t A) @ exp(-i b_k t B)? No...
            # U = prod_{k=1}^{12} [exp(-i b_k t B) exp(-i a_k t A)]
            # This means: U = E_b1 E_a1 E_b2 E_a2 ... E_b12 E_a12
            # Where E_bk = exp(-i b_k t B), E_ak = exp(-i a_k t A)
            
            eA = _expm_from_eig(eigA_vals, eigA_vecs, -1j * a[k] * t)
            eB = _expm_from_eig(eigB_vals, eigB_vecs, -1j * b[k] * t)
            U = eB @ (eA @ U)
        
        U_exact = _expm_from_eig(eigH_vals, eigH_vecs, -1j * t)
        
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def _suzuki_froehlich_init():
    """
    Generate 12-stage symmetric Suzuki-Fröhlich-like coefficients.
    
    The idea: use a high-order splitting pattern. For 12 stages with symmetry,
    we use 6 free parameters for the first half, mirrored for the second.
    
    We start from a 4th-order Suzuki pattern adapted to 12 stages.
    
    Standard 4th-order Suzuki (3 stages per substep, 2 substeps = 6 stages):
    c1, c2, c3 with specific values.
    
    For 12 stages, we can use a nested 4th-order scheme or a direct 6th-order.
    
    Let's use a symmetric pattern with 6 free a-parameters and 6 free b-parameters.
    
    Initialize with values that approximate a 4th-order method:
    For a 4th-order 3-stage method: a = [c1, c2, c1], b = [0, 1-2c1, 0] where c1 = 1/(2-2^(1/3))
    
    For 12 stages, we can think of it as 4 substeps of a 3-stage method, or 
    a single 12-stage method. Let's use a symmetric initialization based on 
    known high-order coefficients.
    
    A 4th-order 5-stage method has coefficients:
    a = [c1, c2, c3, c2, c1], b = [0, d1, d2, d1, 0]
    
    For 12 stages, let's use a pattern that emphasizes the middle stages.
    
    Actually, let's just use a good symmetric initialization and let the optimizer refine.
    """
    # 6 free parameters for first half of a and b
    # Symmetric: a[0..5] mirrored to a[11..6], same for b
    
    # Start with a pattern inspired by 4th-order Suzuki
    # c1 = 1/(2 - 2^(1/3)) ≈ 0.6505
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    
    # For 12 stages, distribute the weight
    # a_half: 6 values that sum to 0.5 (so full sum = 1)
    # b_half: 6 values that sum to 0.5
    
    # Use a pattern that has zeros at the ends (like Strang) and weight in the middle
    a_half = [0.0, c1/4.0, 0.5 - 2*c1/4.0, 0.5 - 2*c1/4.0, c1/4.0, 0.0]
    # Scale to sum to 0.5
    a_sum = sum(a_half)
    if abs(a_sum) > 1e-15:
        a_half = [x / a_sum * 0.5 for x in a_half]
    
    b_half = [0.0, 0.0, 0.25, 0.25, 0.0, 0.0]
    b_sum = sum(b_half)
    if abs(b_sum) > 1e-15:
        b_half = [x / b_sum * 0.5 for x in b_half]
    
    a_full = a_half + a_half[::-1]
    b_full = b_half + b_half[::-1]
    
    # Normalize
    a_sum = sum(a_full)
    b_sum = sum(b_full)
    if abs(a_sum) > 1e-15:
        a_full = [x / a_sum for x in a_full]
    if abs(b_sum) > 1e-15:
        b_full = [x / b_sum for x in b_full]
    
    return a_full, b_full


def coefficients(stages):
    """Optimized product formula coefficients for 6-site Heisenberg chain.
    
    Uses Suzuki-Fröhlich-inspired initialization with local refinement.
    """
    stages = 12
    
    A, B = _build_hamiltonians()
    H_full = A + B
    dim = 2**6
    
    t_values = [0.25, 0.5, 1.0]
    
    # Precompute eigen-decompositions
    eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, eigH_vals, eigH_vecs = _precompute_eigs(A, B, H_full)
    
    # Initialize with Suzuki-Fröhlich pattern
    a_init, b_init = _suzuki_froehlich_init()
    
    best_a = a_init[:]
    best_b = b_init[:]
    best_error = _compute_error_fast(a_init, b_init, eigA_vals, eigA_vecs, 
                                       eigB_vals, eigB_vecs, eigH_vals, eigH_vecs, 
                                       t_values, dim)
    
    # Local refinement using Nelder-Mead on 12 free parameters (6 a + 6 b for first half)
    if _remaining() > 40:
        def objective(params):
            """Params: 6 a-values and 6 b-values for the first half."""
            a_params = params[:6]
            b_params = params[6:12]
            
            # Construct full symmetric vectors
            a_full = list(a_params) + list(a_params[::-1])
            b_full = list(b_params) + list(b_params[::-1])
            
            # Enforce sum constraint by scaling
            a_sum = sum(a_full)
            b_sum = sum(b_full)
            if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
                return 1e10
            
            a_scaled = [x / a_sum for x in a_full]
            b_scaled = [x / b_sum for x in b_full]
            
            # Clamp
            a_scaled = [max(-50.0, min(50.0, x)) for x in a_scaled]
            b_scaled = [max(-50.0, min(50.0, x)) for x in b_scaled]
            
            return _compute_error_fast(a_scaled, b_scaled, eigA_vals, eigA_vecs,
                                        eigB_vals, eigB_vecs, eigH_vals, eigH_vecs,
                                        t_values, dim)
        
        x0 = np.array(a_init[:6] + b_init[:6])
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12})
            
            if result.fun < best_error:
                best_error = result.fun
                a_params = result.x[:6]
                b_params = result.x[6:12]
                
                a_full = list(a_params) + list(a_params[::-1])
                b_full = list(b_params) + list(b_params[::-1])
                
                a_sum = sum(a_full)
                b_sum = sum(b_full)
                if abs(a_sum) > 1e-15:
                    best_a = [x / a_sum for x in a_full]
                if abs(b_sum) > 1e-15:
                    best_b = [x / b_sum for x in b_full]
                
                best_a = [max(-50.0, min(50.0, x)) for x in best_a]
                best_b = [max(-50.0, min(50.0, x)) for x in best_b]
                
        except Exception:
            pass
    
    # Second round of refinement with different initial perturbation if time permits
    if _remaining() > 20:
        # Perturb the best solution and try again
        np.random.seed(42)
        a_perturbed = [x + 0.05 * np.random.randn() for x in best_a]
        b_perturbed = [x + 0.05 * np.random.randn() for x in best_b]
        
        # Normalize
        a_sum = sum(a_perturbed)
        b_sum = sum(b_perturbed)
        if abs(a_sum) > 1e-15:
            a_perturbed = [x / a_sum for x in a_perturbed]
        if abs(b_sum) > 1e-15:
            b_perturbed = [x / b_sum for x in b_perturbed]
        
        err_perturbed = _compute_error_fast(a_perturbed, b_perturbed, eigA_vals, eigA_vecs,
                                              eigB_vals, eigB_vecs, eigH_vals, eigH_vecs,
                                              t_values, dim)
        
        if err_perturbed < best_error:
            best_a = a_perturbed[:]
            best_b = b_perturbed[:]
            best_error = err_perturbed
            
            if _remaining() > 15:
                def objective2(params):
                    a_params = params[:6]
                    b_params = params[6:12]
                    
                    a_full = list(a_params) + list(a_params[::-1])
                    b_full = list(b_params) + list(b_params[::-1])
                    
                    a_sum = sum(a_full)
                    b_sum = sum(b_full)
                    if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
                        return 1e10
                    
                    a_scaled = [x / a_sum for x in a_full]
                    b_scaled = [x / b_sum for x in b_full]
                    
                    a_scaled = [max(-50.0, min(50.0, x)) for x in a_scaled]
                    b_scaled = [max(-50.0, min(50.0, x)) for x in b_scaled]
                    
                    return _compute_error_fast(a_scaled, b_scaled, eigA_vals, eigA_vecs,
                                                eigB_vals, eigB_vecs, eigH_vals, eigH_vecs,
                                                t_values, dim)
                
                x0_2 = np.array(best_a[:6] + best_b[:6])
                
                try:
                    from scipy.optimize import minimize
                    result2 = minimize(objective2, x0_2, method='Nelder-Mead', 
                                    options={'maxiter': 300, 'fatol': 1e-12, 'xatol': 1e-12})
                    
                    if result2.fun < best_error:
                        best_error = result2.fun
                        a_params = result2.x[:6]
                        b_params = result2.x[6:12]
                        
                        a_full = list(a_params) + list(a_params[::-1])
                        b_full = list(b_params) + list(b_params[::-1])
                        
                        a_sum = sum(a_full)
                        b_sum = sum(b_full)
                        if abs(a_sum) > 1e-15:
                            best_a = [x / a_sum for x in a_full]
                        if abs(b_sum) > 1e-15:
                            best_b = [x / b_sum for x in b_full]
                        
                        best_a = [max(-50.0, min(50.0, x)) for x in best_a]
                        best_b = [max(-50.0, min(50.0, x)) for x in best_b]
                        
                except Exception:
                    pass
    
    # Final normalization and clamping
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-15:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-15:
        best_b = [x / b_sum for x in best_b]
    
    best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
    best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
    
    return best_a, best_b
