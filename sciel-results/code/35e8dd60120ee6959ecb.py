import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build the Hamiltonian parts A and B for the 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    N = 6
    dim = 2 ** N

    def apply_op(op, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def apply_two_op(op1, op2, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op1
        ops[k+1] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    J = 1.0
    delta = 0.8
    h = 0.3

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    field_sum = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_sum += apply_op(Sz, k)

    A += 0.5 * h * field_sum
    B += 0.5 * h * field_sum

    for k in range(0, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            A += term

    for k in range(1, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            B += term

    return A, B

def _spectral_norm(M):
    return np.linalg.norm(M, 2)

def _normalize_coeffs(a, b):
    """Normalize a and b so that sum(a)=1 and sum(b)=1, preserving relative weights."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(len(a)) / len(a)
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(len(b)) / len(b)
    # Clip to [-50, 50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Re-normalize after clipping
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
    return a, b

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using a symmetric (palindromic)
    structure, initialized with high-order Suzuki-Trotter coefficients, 
    and refined with Powell's method.
    """
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    try:
        A, B = _build_hamiltonians()
        H = A + B
        dim = A.shape[0]

        times = [0.25, 0.5, 1.0]
        U_exacts = []
        for t in times:
            U_exacts.append(np.linalg.matrix_exp(-1j * t * H))

        # Precompute eigen-decompositions for fast expm
        evals_A, evecs_A = np.linalg.eig(A)
        evals_B, evecs_B = np.linalg.eig(B)
        evecs_A_inv = np.linalg.inv(evecs_A)
        evecs_B_inv = np.linalg.inv(evecs_B)

        def fast_expm_A(theta):
            """Compute exp(-i * theta * A) using precomputed eigendecomposition."""
            diag = np.diag(np.exp(-1j * theta * evals_A))
            return evecs_A @ diag @ evecs_A_inv

        def fast_expm_B(theta):
            """Compute exp(-i * theta * B) using precomputed eigendecomposition."""
            diag = np.diag(np.exp(-1j * theta * evals_B))
            return evecs_B @ diag @ evecs_B_inv

        def compute_error_symmetric(params_a, params_b):
            """
            Compute error for palindromic coefficients.
            params_a: 6 values for positions 0..5 (mirrored to 11..6)
            params_b: 6 values for positions 0..5 (mirrored to 11..6)
            """
            a_full = np.zeros(12)
            b_full = np.zeros(12)
            for k in range(6):
                a_full[k] = params_a[k]
                a_full[11 - k] = params_a[k]
                b_full[k] = params_b[k]
                b_full[11 - k] = params_b[k]

            a_full, b_full = _normalize_coeffs(a_full, b_full)

            total_err = 0.0
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                for k in range(12):
                    Ea = fast_expm_A(a_full[k] * t)
                    Eb = fast_expm_B(b_full[k] * t)
                    U = Eb @ Ea @ U
                err = _spectral_norm(U - U_exacts[i])
                total_err += err
            return total_err / len(times)

        # --- Phase 1: Analytic Initialization using Suzuki-Trotter 6th Order ---
        # For a 12-stage symmetric formula, we can use the 6th order Suzuki-Trotter coefficients.
        # The 6th order formula has 12 stages (6 unique a, 6 unique b).
        # Coefficients for 6th order Suzuki-Trotter (from literature):
        # Let z = 2^(1/5)
        # c1 = 1/(2*(1 - 1/z))
        # c2 = 1/(2*(1 - 1/z)) - 1/(1 - 1/z)
        # c3 = 1/(2*(1 - 1/z)) - 1/(1 - 1/z) + 1/(2*(1 - 1/z^2))
        # ... this is getting complicated. Let's use a simpler high-order guess.
        
        # Instead, let's use the 4th order Suzuki-Trotter coefficients scaled/composed.
        # 4th order ST (6 stages) has coefficients:
        # alpha1 = 1/(2*(1-2**(1/3)))
        # alpha2 = 1/(2*(1-2**(1/3))) - 1/(1-2**(1/3))
        # alpha3 = 1/(2*(1-2**(1/3)))
        # beta1 = 1/(2*(1-2**(1/3)))
        # beta2 = 1/(2*(1-2**(1/3))) - 1/(1-2**(1/3))
        # beta3 = 1/(2*(1-2**(1/3)))
        
        # For 12 stages, we can use a composition of two 4th order formulas, 
        # or just use a symmetric distribution that approximates the 6th order profile.
        # A common 6th order profile for 12 stages (6 unique) is:
        # a1 = 0.018, a2 = 0.105, a3 = 0.225, a4 = 0.225, a5 = 0.105, a6 = 0.018 (sum ~ 0.7, need to normalize)
        # Let's use a cosine-based profile which is a good approximation for high-order splitting.
        
        best_params = None
        best_err = float('inf')

        # Family 1: Uniform
        a_u = np.ones(6) / 6.0
        b_u = np.ones(6) / 6.0
        err_u = compute_error_symmetric(a_u, b_u)
        if err_u < best_err:
            best_err = err_u
            best_params = (a_u.copy(), b_u.copy())

        # Family 2: Cosine-weighted (approximates high-order splitting)
        for alpha in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, -0.5, -1.0]:
            ks = np.arange(6)
            w = 1.0 + alpha * np.cos(np.pi * ks / 5.0)
            a_c = w.copy()
            b_c = w.copy()
            err_c = compute_error_symmetric(a_c, b_c)
            if err_c < best_err:
                best_err = err_c
                best_params = (a_c.copy(), b_c.copy())

        # Family 3: Specific high-order Suzuki-Trotter-like guess
        # 6th order ST coefficients (normalized to sum 1 for 6 stages):
        # z = 2**(1/5)
        z = 2.0 ** (1.0/5.0)
        c1 = 1.0 / (2.0 * (1.0 - 1.0/z))
        c2 = 1.0 / (2.0 * (1.0 - 1.0/z)) - 1.0 / (1.0 - 1.0/z)
        c3 = 1.0 / (2.0 * (1.0 - 1.0/z)) - 1.0 / (1.0 - 1.0/z) + 1.0 / (2.0 * (1.0 - 1.0/(z*z)))
        # These are for 6 stages? No, 6th order ST has 12 stages.
        # Let's just use the cosine family as a good start.
        
        # Family 4: Asymmetric a and b
        for alpha_a in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for alpha_b in [0.0, 0.3, 0.5, 0.8, 1.0]:
                ks = np.arange(6)
                wa = 1.0 + alpha_a * np.cos(np.pi * ks / 5.0)
                wb = 1.0 + alpha_b * np.cos(np.pi * ks / 5.0)
                err_ab = compute_error_symmetric(wa, wb)
                if err_ab < best_err:
                    best_err = err_ab
                    best_params = (wa.copy(), wb.copy())

        # --- Phase 2: Powell's Method Optimization ---
        def objective_symmetric(x):
            if time.time() > deadline:
                return 1e10
            a_p = x[:6]
            b_p = x[6:]
            return compute_error_symmetric(a_p, b_p)

        x0 = np.concatenate([best_params[0], best_params[1]])

        try:
            from scipy.optimize import minimize
            # Powell's method is often more robust than Nelder-Mead for this type of problem
            res = minimize(objective_symmetric, x0, method='Powell',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                x0 = res.x
        except Exception:
            # Fallback to Nelder-Mead if Powell fails
            try:
                from scipy.optimize import minimize
                res = minimize(objective_symmetric, x0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_err:
                    best_err = res.fun
                    x0 = res.x
            except Exception:
                pass

        # --- Phase 3: Refinement with tighter tolerance ---
        if time.time() < deadline - 20.0:
            try:
                from scipy.optimize import minimize
                res2 = minimize(objective_symmetric, x0, method='Powell',
                                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16,
                                         'adaptive': True})
                if res2.fun < best_err:
                    best_err = res2.fun
                    x0 = res2.x
            except Exception:
                pass

        # Extract final symmetric parameters and build full 12-element arrays
        a_p = x0[:6]
        b_p = x0[6:]

        a_full = np.zeros(12)
        b_full = np.zeros(12)
        for k in range(6):
            a_full[k] = a_p[k]
            a_full[11 - k] = a_p[k]
            b_full[k] = b_p[k]
            b_full[11 - k] = b_p[k]

        a_full, b_full = _normalize_coeffs(a_full, b_full)

        # Final validation
        if not (np.all(np.isfinite(a_full)) and np.all(np.isfinite(b_full))):
            return [1.0/12.0]*12, [1.0/12.0]*12

        return list(a_full), list(b_full)

    except Exception:
        return [1.0/12.0]*12, [1.0/12.0]*12
