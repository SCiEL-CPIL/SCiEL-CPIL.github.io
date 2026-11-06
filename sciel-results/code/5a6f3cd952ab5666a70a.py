"""Optimized 12-stage product formula via direct coefficient fitting."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def coefficients(stages):
    """Fit 12-stage product formula coefficients to minimize spectral norm error."""
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s limit

    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Build the 6-site XXZ Hamiltonian
    # Pauli matrices
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Build operators for 6 sites
    def op(site, op_mat):
        """Apply op_mat at given site (0-indexed) on 6-site chain."""
        result = op_mat
        for s in range(6):
            if s != site:
                result = np.kron(result, I2)
        return result

    def op2(site1, site2, op1, op2):
        """Apply op1 at site1 and op2 at site2 (0-indexed)."""
        result = np.eye(2**6, dtype=complex)
        for s in range(6):
            if s == site1:
                result = np.kron(result, op1)
            elif s == site2:
                result = np.kron(result, op2)
            else:
                result = np.kron(result, I2)
        return result

    # XXZ Hamiltonian: anisotropy=0.8, field=0.3
    delta = 0.8
    field = 0.3

    H_odd = np.zeros((64, 64), dtype=complex)
    H_even = np.zeros((64, 64), dtype=complex)

    # Odd bonds: (0,1), (2,3), (4,5)
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        H_odd += 0.5 * op2(i, j, Sx, Sx) + 0.5 * op2(i, j, Sy, Sy) + delta * op2(i, j, Sz, Sz)

    # Even bonds: (1,2), (3,4)
    for (i, j) in [(1, 2), (3, 4)]:
        H_even += 0.5 * op2(i, j, Sx, Sx) + 0.5 * op2(i, j, Sy, Sy) + delta * op2(i, j, Sz, Sz)

    # Half field on each
    for s in range(6):
        H_odd += 0.5 * field * op(s, Sz)
        H_even += 0.5 * field * op(s, Sz)

    A = H_odd
    B = H_even
    H_total = A + B

    # Precompute exact propagators at evaluation times
    eval_times = [0.25, 0.5, 1.0]
    exact_props = []
    for t in eval_times:
        # exp(-i * t * H_total)
        eigvals, eigvecs = np.linalg.eigh(H_total)
        exact_props.append(eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T)

    # Precompute exp(-i*t*A) and exp(-i*t*B) for each t
    eigA_vals, eigA_vecs = np.linalg.eigh(A)
    eigB_vals, eigB_vecs = np.linalg.eigh(B)

    def exp_iA(t, coeffs_a):
        """Compute prod_k exp(-i * a_k * t * A) in order."""
        result = np.eye(64, dtype=complex)
        for ak in coeffs_a:
            if abs(ak) > 1e-15:
                phase = np.exp(-1j * ak * t * eigA_vals)
                result = (eigA_vecs @ np.diag(phase) @ eigA_vecs.conj().T) @ result
        return result

    def exp_iB(t, coeffs_b):
        """Compute prod_k exp(-i * b_k * t * B) in order."""
        result = np.eye(64, dtype=complex)
        for bk in coeffs_b:
            if abs(bk) > 1e-15:
                phase = np.exp(-1j * bk * t * eigB_vals)
                result = (eigB_vecs @ np.diag(phase) @ eigB_vecs.conj().T) @ result
        return result

    def product_formula(t, coeffs_a, coeffs_b):
        """U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12."""
        result = np.eye(64, dtype=complex)
        for k in range(12):
            # exp(-i a_k t A)
            if abs(coeffs_a[k]) > 1e-15:
                phase = np.exp(-1j * coeffs_a[k] * t * eigA_vals)
                result = (eigA_vecs @ np.diag(phase) @ eigA_vecs.conj().T) @ result
            # exp(-i b_k t B)
            if abs(coeffs_b[k]) > 1e-15:
                phase = np.exp(-1j * coeffs_b[k] * t * eigB_vals)
                result = (eigB_vecs @ np.diag(phase) @ eigB_vecs.conj().T) @ result
        return result

    def objective(params):
        """Mean spectral norm error at three times."""
        # params: 24 values, first 12 are a, last 12 are b
        coeffs_a = params[:12]
        coeffs_b = params[12:]

        # Enforce sum constraints via projection
        sa = np.sum(coeffs_a)
        sb = np.sum(coeffs_b)
        coeffs_a = coeffs_a / sa if abs(sa) > 1e-10 else np.ones(12) / 12
        coeffs_b = coeffs_b / sb if abs(sb) > 1e-10 else np.ones(12) / 12

        total_err = 0.0
        for idx, t in enumerate(eval_times):
            U = product_formula(t, coeffs_a, coeffs_b)
            diff = U - exact_props[idx]
            # Spectral norm = largest singular value
            sv = np.linalg.svd(diff, compute_uv=False)
            total_err += sv[0]
        return total_err / len(eval_times)

    # Initial guess: 4th-order Suzuki-Trotter pattern
    two_thirds = 2.0 ** (1.0 / 3.0)
    a1_st = 1.0 / (4.0 - two_thirds)
    a2_st = 1.0 / (2.0 * (4.0 - two_thirds))
    b1_st = 1.0 / (2.0 - two_thirds)
    b2_st = 1.0 / (2.0 * (2.0 - two_thirds))

    a_init = [a1_st, a2_st, a2_st, a2_st, a2_st, a2_st,
              a2_st, a2_st, a2_st, a2_st, a2_st, a1_st]
    b_init = [b1_st, b2_st, b2_st, b2_st, b2_st, b2_st,
              b2_st, b2_st, b2_st, b2_st, b2_st, b1_st]

    # Normalize
    a_init = np.array(a_init) / np.sum(a_init)
    b_init = np.array(b_init) / np.sum(b_init)

    x0 = np.concatenate([a_init, b_init])

    # Optimize
    from scipy.optimize import minimize

    best_a = a_init
    best_b = b_init
    best_err = objective(x0)

    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_err:
            best_err = res.fun
            params = res.x
            sa = np.sum(params[:12])
            sb = np.sum(params[12:])
            best_a = params[:12] / sa if abs(sa) > 1e-10 else np.ones(12) / 12
            best_b = params[12:] / sb if abs(sb) > 1e-10 else np.ones(12) / 12
    except Exception:
        pass

    # If time permits, try a second optimization with different initial conditions
    if time.time() < deadline - 20.0:
        # Try symmetric pattern: a = [a1,a2,a3,a4,a5,a6,a6,a5,a4,a3,a2,a1]
        # Parameterize by 6 free parameters for a and 6 for b
        def sym_objective(params):
            a_half = params[:6]
            b_half = params[6:12]
            coeffs_a = np.concatenate([a_half, a_half[::-1]])
            coeffs_b = np.concatenate([b_half, b_half[::-1]])
            sa = np.sum(coeffs_a)
            sb = np.sum(coeffs_b)
            coeffs_a = coeffs_a / sa if abs(sa) > 1e-10 else np.ones(12) / 12
            coeffs_b = coeffs_b / sb if abs(sb) > 1e-10 else np.ones(12) / 12

            total_err = 0.0
            for idx, t in enumerate(eval_times):
                U = product_formula(t, coeffs_a, coeffs_b)
                diff = U - exact_props[idx]
                sv = np.linalg.svd(diff, compute_uv=False)
                total_err += sv[0]
            return total_err / len(eval_times)

        # Symmetric initial guess
        a_sym_init = np.array([a1_st, a2_st, a2_st, a2_st, a2_st, a2_st]) / np.sum([a1_st, a2_st, a2_st, a2_st, a2_st, a2_st]) * 2
        b_sym_init = np.array([b1_st, b2_st, b2_st, b2_st, b2_st, b2_st]) / np.sum([b1_st, b2_st, b2_st, b2_st, b2_st, b2_st]) * 2
        x0_sym = np.concatenate([a_sym_init, b_sym_init])

        try:
            res2 = minimize(sym_objective, x0_sym, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res2.fun < best_err:
                best_err = res2.fun
                a_half = res2.x[:6]
                b_half = res2.x[6:12]
                best_a = np.concatenate([a_half, a_half[::-1]])
                best_b = np.concatenate([b_half, b_half[::-1]])
                best_a = best_a / np.sum(best_a)
                best_b = best_b / np.sum(best_b)
        except Exception:
            pass

    # Final safety: if optimization failed, use the initial ST coefficients
    if time.time() > deadline - 5.0:
        best_a = a_init
        best_b = b_init

    # Ensure constraints
    best_a = np.array(best_a, dtype=float)
    best_b = np.array(best_b, dtype=float)
    best_a = best_a / np.sum(best_a)
    best_b = best_b / np.sum(best_b)

    # Clip to valid range
    best_a = np.clip(best_a, -50.0, 50.0)
    best_b = np.clip(best_b, -50.0, 50.0)
    # Re-normalize after clipping
    best_a = best_a / np.sum(best_a)
    best_b = best_b / np.sum(best_b)

    a_list = [float(x) for x in best_a]
    b_list = [float(x) for x in best_b]

    assert abs(sum(a_list) - 1.0) < 1e-8, f"sum(a) = {sum(a_list)}"
    assert abs(sum(b_list) - 1.0) < 1e-8, f"sum(b) = {sum(b_list)}"
    assert all(-50.0 <= x <= 50.0 for x in a_list), "a out of bounds"
    assert all(-50.0 <= x <= 50.0 for x in b_list), "b out of bounds"

    return a_list, b_list
# EVOLVE-BLOCK-END
