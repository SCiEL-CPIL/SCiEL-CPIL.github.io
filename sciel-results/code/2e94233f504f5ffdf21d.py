"""Product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time

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

def _suzuki_order3_coefficients():
    """Return 3rd-order Suzuki-Trotter coefficients for 2-splitting.
    
    The 3rd-order formula is: exp(-i c1 A) exp(-i c2 B) exp(-i c3 A) exp(-i c2 B) exp(-i c1 A)
    where:
    c1 = 1/(2*(1 - 2^(-1/3)))
    c2 = 1/(2*(2^(1/3) - 1))  
    c3 = 1 - 2*c1
    
    For our 12-stage structure with pattern exp(-i b_k B) exp(-i a_k A),
    we can compose this. Let's use a symmetric 12-stage formula.
    
    Actually, let's think differently. We have 12 stages of the form:
    exp(-i b_1 B) exp(-i a_1 A) exp(-i b_2 B) exp(-i a_2 A) ... exp(-i b_12 B) exp(-i a_12 A)
    
    This is equivalent to 12 pairs. For a symmetric formula, we want
    a_k = a_{13-k} and b_k = b_{13-k}.
    
    Let's try to embed the 3rd-order coefficients. One approach:
    Use 4 substeps of a 3rd-order-like pattern.
    
    Alternatively, let's just use the known 4th-order coefficients which are well-studied:
    4th order: exp(-i c1 A) exp(-i c2 B) exp(-i c1 A) exp(-i c3 B) exp(-i c1 A) exp(-i c2 B) exp(-i c1 A)
    c1 = 1/(2*(1 - 2^(-1/4)))
    c2 = 1 - 2*c1
    c3 = 1 - 2*c2 = 2*c1 - 1
    
    For 12 stages, we can use this pattern scaled appropriately.
    """
    # 4th order coefficients
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(-0.25)))
    c2 = 1.0 - 2.0 * c1
    c3 = 2.0 * c1 - 1.0
    
    # The 4th order formula has 7 terms: A, B, A, B, A, B, A
    # We need to map this to our 12-stage format.
    # Let's create a 12-stage symmetric formula inspired by these coefficients.
    
    # Simple approach: use the 4th-order coefficients as the base
    # and replicate/scale to fill 12 stages.
    
    # For now, return a reasonable starting point
    # We'll refine with optimization
    a = [1.0/12] * 12
    b = [1.0/12] * 12
    
    return a, b

def coefficients(stages):
    """Design 12-stage product formula coefficients for Heisenberg chain."""
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill
    
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_XXZ_6site()
    dim = A.shape[0]
    
    # Precompute eigendecompositions
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(A + B)
    
    t_evals = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    U_exacts = []
    for t in t_evals:
        U_exact = eigvecs_H @ np.diag(np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
        U_exacts.append(U_exact)
    
    def compute_error_fast(a, b):
        """Fast error computation using precomputed data."""
        total = 0.0
        for idx, t in enumerate(t_evals):
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                expB = eigvecs_B @ np.diag(np.exp(-1j * b[k] * t * eigvals_B)) @ eigvecs_B.conj().T
                expA = eigvecs_A @ np.diag(np.exp(-1j * a[k] * t * eigvals_A)) @ eigvecs_A.conj().T
                U = expB @ expA @ U
            diff = U - U_exacts[idx]
            s = np.linalg.svd(diff, compute_uv=False)
            total += s[0]
        return total / len(t_evals)
    
    # Start with known good coefficients
    # Use 4th-order Suzuki-Trotter inspired coefficients
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(-0.25)))
    c2 = 1.0 - 2.0 * c1
    c3 = 2.0 * c1 - 1.0
    
    # Create a 12-stage symmetric formula
    # Pattern: b1,a1,b2,a2,b3,a3,b4,a4,b5,a5,b6,a6 with symmetry
    # a_k = a_{13-k}, b_k = b_{13-k}
    
    # Try different initializations
    candidates = []
    
    # Candidate 1: Uniform
    candidates.append(([1.0/12]*12, [1.0/12]*12))
    
    # Candidate 2: 4th-order inspired (symmetric)
    # Use c1, c2, c3 pattern scaled to sum to 1
    # For 12 stages with symmetry, we have 6 unique values each
    # Let's try: [c1, c2, c3, c3, c2, c1] for both a and b, normalized
    a_cand = [c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1]
    b_cand = a_cand[:]
    sa = sum(a_cand)
    sb = sum(b_cand)
    a_cand = [x/sa for x in a_cand]
    b_cand = [x/sb for x in b_cand]
    candidates.append((a_cand, b_cand))
    
    # Candidate 3: Strang with 6 substeps
    candidates.append(([1.0/6]*12, [1.0/6]*12))
    
    # Candidate 4: 6th order inspired
    c1_6 = 1.0 / (2.0 * (1.0 - 2.0**(-1.0/6.0)))
    c2_6 = 1.0 - 2.0 * c1_6
    a_cand4 = [c1_6, c2_6, c1_6, c1_6, c2_6, c1_6, c1_6, c2_6, c1_6, c1_6, c2_6, c1_6]
    b_cand4 = a_cand4[:]
    sa4 = sum(a_cand4)
    a_cand4 = [x/sa4 for x in a_cand4]
    b_cand4 = a_cand4[:]
    candidates.append((a_cand4, b_cand4))
    
    # Evaluate all candidates
    best_a = None
    best_b = None
    best_err = float('inf')
    
    for a_cand, b_cand in candidates:
        err = compute_error_fast(a_cand, b_cand)
        if err < best_err:
            best_err = err
            best_a = a_cand[:]
            best_b = b_cand[:]
    
    # Local optimization using Nelder-Mead style or simple hill climbing
    # Use symmetric parameterization: 6 free params for a, 6 for b
    # with sum constraint
    
    def symmetrize_normalize(a6, b6):
        """Create full 12-element arrays from 6 unique values, enforce symmetry and sum=1."""
        a = [a6[0], a6[1], a6[2], a6[3], a6[4], a6[5], a6[5], a6[4], a6[3], a6[2], a6[1], a6[0]]
        b = [b6[0], b6[1], b6[2], b6[3], b6[4], b6[5], b6[5], b6[4], b6[3], b6[2], b6[1], b6[0]]
        sa = sum(a)
        sb = sum(b)
        if sa != 0:
            a = [x/sa for x in a]
        if sb != 0:
            b = [x/sb for x in b]
        return a, b
    
    # Extract current 6-param representation
    a6_cur = [best_a[0], best_a[1], best_a[2], best_a[3], best_a[4], best_a[5]]
    b6_cur = [best_b[0], best_b[1], best_b[2], best_b[3], best_b[4], best_b[5]]
    
    step_size = 0.03
    max_iter = 500
    no_improve_count = 0
    max_no_improve = 20
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try perturbing each of the 12 free parameters (6 for a, 6 for b)
        for param_type in ['a', 'b']:
            for param_idx in range(6):
                for sign in [1, -1]:
                    a6_test = a6_cur[:]
                    b6_test = b6_cur[:]
                    
                    if param_type == 'a':
                        a6_test[param_idx] += sign * step_size
                    else:
                        b6_test[param_idx] += sign * step_size
                    
                    a_test, b_test = symmetrize_normalize(a6_test, b6_test)
                    err_test = compute_error_fast(a_test, b_test)
                    
                    if err_test < best_err - 1e-12:
                        a6_cur = a6_test[:]
                        b6_cur = b6_test[:]
                        best_a, best_b = a_test, b_test
                        best_err = err_test
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            step_size *= 0.85
            if step_size < 1e-6 or no_improve_count >= max_no_improve:
                break
    
    # Final normalization and clamping
    sa = sum(best_a)
    sb = sum(best_b)
    if sa != 0:
        best_a = [x/sa for x in best_a]
    if sb != 0:
        best_b = [x/sb for x in best_b]
    
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    return best_a, best_b
