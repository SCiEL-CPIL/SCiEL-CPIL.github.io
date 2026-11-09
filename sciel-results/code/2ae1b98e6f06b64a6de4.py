"""Improved growth-index fit with minimax epigraph and iterative reweighting."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to minimize worst-case relative error on f and D."""
    t_start = time.time()
    deadline = t_start + 82.0  # 8s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa) - physically reasonable range
    Om0_vals = np.array([0.20, 0.24, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    wa_vals = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

    # Scale factor grid for fitting - denser where growth is sensitive
    a_vals = np.linspace(0.05, 0.99, 60)

    # Build design matrix, targets, and metadata
    rows = []
    targets = []
    # Store (Om0, w0, wa, a) for each row to compute D later
    meta = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor

                    if E2 <= 0 or E2 < 1e-12:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    if Om_a <= 0 or Om_a >= 1.0:
                        continue
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)

                    if u <= 0.01:
                        continue

                    # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)

                    # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    meta.append((Om0, w0, wa, a))

                    if time.time() > deadline - 15.0:
                        break
                if time.time() > deadline - 15.0:
                    break
            if time.time() > deadline - 15.0:
                break
        if time.time() > deadline - 15.0:
            break

    if len(rows) < 50:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(rows)
    b = np.array(targets)
    n = len(rows)

    # Helper: compute gamma from coefficients
    def compute_gamma(c):
        return A @ c

    # Helper: compute relative errors for f and D
    def compute_errors(c):
        """Return array of relative errors for f and D at each point."""
        gamma_pred = A @ c
        gamma_true = b

        # f = Om(a)^gamma, so ln(f) = gamma * ln(Om(a))
        # Relative error in f: |f_pred - f_true| / |f_true|
        # f_true = Om_a^gamma_true, f_pred = Om_a^gamma_pred
        # ln(f_pred/f_true) = (gamma_pred - gamma_true) * ln(Om_a)
        # So relative error ≈ |gamma_pred - gamma_true| * |ln(Om_a)| for small errors

        # Get Om_a for each point
        Om_a_arr = np.zeros(n)
        for i, (Om0, w0, wa, a) in enumerate(meta):
            a_cubed = a**3
            de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
            E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
            if E2 > 0:
                Om_a_arr[i] = Om0 * a_cubed / E2

        # Relative error in f
        ln_Om = np.log(np.maximum(Om_a_arr, 1e-10))
        f_rel_err = np.abs(gamma_pred - gamma_true) * np.abs(ln_Om)

        # For D: D(a)/D(1) = exp(-∫_a^1 f(exp(x)) dx)
        # The relative error in D depends on the integral of f errors
        # Approximate: if f has relative error ε(x), then D has error ~ ∫ ε(x) f(x) dx / D
        # For a rough estimate, weight by the cumulative effect
        # Use a simple approximation: D_rel_err ≈ max over sub-intervals of accumulated f error

        # More practical: compute D directly for each point
        D_rel_errs = np.zeros(n)
        for i, (Om0, w0, wa, a) in enumerate(meta):
            if a >= 1.0:
                D_rel_errs[i] = 0.0
                continue
            # Compute f_true and f_pred at a grid of x from a to 1
            x_grid = np.linspace(np.log(a), 0.0, 30)
            a_grid = np.exp(x_grid)

            f_true_vals = np.zeros(len(a_grid))
            f_pred_vals = np.zeros(len(a_grid))
            for j, aj in enumerate(a_grid):
                a_cubed_j = aj**3
                de_factor_j = aj**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-aj))
                E2_j = Om0 * a_cubed_j + (1.0 - Om0) * de_factor_j
                if E2_j <= 0:
                    f_true_vals[j] = 0.0
                    f_pred_vals[j] = 0.0
                    continue
                Om_a_j = Om0 * a_cubed_j / E2_j
                if Om_a_j <= 0:
                    f_true_vals[j] = 0.0
                    f_pred_vals[j] = 0.0
                    continue
                q_j = 1.0 - Om_a_j
                u_j = 1.0 + w0 + wa * (1.0 - aj)
                gamma_true_j = (5.0/7.0) * (Om_a_j / u_j)**(4.0/7.0) if u_j > 0.01 else 0.55
                gamma_pred_j = (c[0] + c[1]*q_j + c[2]*q_j*q_j +
                                c[3]*u_j + c[4]*wa*aj + c[5]*u_j*q_j)
                f_true_vals[j] = Om_a_j**gamma_true_j
                f_pred_vals[j] = Om_a_j**gamma_pred_j

            # Integrate
            int_true = np.trapz(f_true_vals, x_grid)
            int_pred = np.trapz(f_pred_vals, x_grid)

            D_true = np.exp(-int_true)
            D_pred = np.exp(-int_pred)

            if abs(D_true) > 1e-10:
                D_rel_errs[i] = abs(D_pred - D_true) / abs(D_true)
            else:
                D_rel_errs[i] = 0.0

        # Combined relative error: max of f and D errors at each point
        # The evaluator takes max over both observables, so we take max per point
        combined_err = np.maximum(f_rel_err, D_rel_errs)
        return combined_err

    # Helper: check gamma range
    def check_gamma_range(c):
        test_Om0 = Om0_vals[:3]
        test_w0 = w0_vals[:3]
        test_wa = wa_vals[:3]
        test_a = np.linspace(0.05, 0.99, 15)
        for Om0 in test_Om0:
            for w0 in test_w0:
                for wa in test_wa:
                    for a in test_a:
                        a_cubed = a**3
                        de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                        E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                        if E2 <= 0:
                            continue
                        Om_a = Om0 * a_cubed / E2
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - a)
                        gamma = (c[0] + c[1]*q + c[2]*q*q +
                                 c[3]*u + c[4]*wa*a + c[5]*u*q)
                        if gamma < 0.1 or gamma > 1.2:
                            return False
        return True

    # Initial guess
    coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Phase 1: Iterative reweighting toward worst-case (minimax surrogate)
    # Use log-sum-exp as smooth max surrogate
    for iteration in range(6):
        if time.time() > deadline - 20.0:
            break

        errors = compute_errors(coeffs)
        if len(errors) == 0:
            break

        # Log-sum-exp for smooth max
        max_err = np.max(errors)
        if max_err < 1e-12:
            break

        # Weights: softmax of errors (emphasize worst points)
        # Use temperature to control sharpness
        temp = max_err * 0.5
        if temp < 1e-6:
            temp = 1e-6
        w = np.exp(errors / temp)
        w = w / np.sum(w)

        # Weighted least squares on gamma
        Aw = A * w[:, None]
        bw = b * w
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        except Exception:
            break

        # Clip to [-10, 10]
        coeffs = np.clip(coeffs, -10.0, 10.0)

        # Verify gamma range
        if not check_gamma_range(coeffs):
            # Soft penalty approach
            def obj_with_penalty(c):
                pred = A @ c
                fit_err = np.sum(w * (pred - b)**2)
                pen = 0.0
                test_Om0 = Om0_vals[:2]
                test_w0 = w0_vals[:2]
                test_wa = wa_vals[:2]
                test_a = np.linspace(0.1, 0.99, 8)
                for Om0 in test_Om0:
                    for w0 in test_w0:
                        for wa in test_wa:
                            for a in test_a:
                                a_cubed = a**3
                                de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                                if E2 <= 0:
                                    continue
                                Om_a = Om0 * a_cubed / E2
                                q = 1.0 - Om_a
                                u = 1.0 + w0 + wa * (1.0 - a)
                                gamma = (c[0] + c[1]*q + c[2]*q*q +
                                         c[3]*u + c[4]*wa*a + c[5]*u*q)
                                if gamma < 0.1:
                                    pen += (0.1 - gamma)**2
                                elif gamma > 1.2:
                                    pen += (gamma - 1.2)**2
                return fit_err + 1000.0 * pen

            bounds = [(-10, 10)] * 6
            try:
                result = minimize(obj_with_penalty, coeffs, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 50})
                if result.success:
                    coeffs = np.clip(result.x, -10.0, 10.0)
            except Exception:
                pass

            if not check_gamma_range(coeffs):
                coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Phase 2: Direct minimax via epigraph formulation
    # Minimize t subject to: |gamma_pred_i - gamma_true_i| * |ln(Om_i)| <= t
    #                        and D_rel_err_i <= t
    #                        and 0.1 <= gamma_i <= 1.2
    if time.time() < deadline - 10.0:
        # Precompute ln_Om for f error weighting
        Om_a_arr = np.zeros(n)
        for i, (Om0, w0, wa, a) in enumerate(meta):
            a_cubed = a**3
            de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
            E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
            if E2 > 0:
                Om_a_arr[i] = Om0 * a_cubed / E2
        ln_Om_abs = np.abs(np.log(np.maximum(Om_a_arr, 1e-10)))

        # Use a subset of points for the epigraph solve (too many constraints)
        # Select worst 200 points from current solution
        errors_current = compute_errors(coeffs)
        if len(errors_current) > 200:
            idx_worst = np.argsort(errors_current)[-200:]
        else:
            idx_worst = np.arange(len(errors_current))

        A_sub = A[idx_worst]
        b_sub = b[idx_worst]
        ln_Om_sub = ln_Om_abs[idx_worst]
        meta_sub = [meta[i] for i in idx_worst]

        def epigraph_obj(vars_):
            """vars_ = [c0..c5, t]"""
            c = vars_[:6]
            t = vars_[6]
            gamma_pred = A_sub @ c
            # f relative error
            f_err = np.abs(gamma_pred - b_sub) * ln_Om_sub
            # Constraint violations
            viol_f = np.maximum(0, f_err - t)
            # D error (approximate with f error scaled)
            # For speed, use f error as proxy for D error in the optimization
            # The D error is typically smaller or comparable
            viol_total = np.sum(viol_f)
            # Gamma range penalty
            gamma_range_viol = 0.0
            gamma_vals = gamma_pred
            gamma_range_viol += np.sum(np.maximum(0, 0.1 - gamma_vals)**2)
            gamma_range_viol += np.sum(np.maximum(0, gamma_vals - 1.2)**2)
            return t + 10.0 * viol_total + 100.0 * gamma_range_viol

        x0 = np.append(coeffs, np.max(errors_current) + 0.1)
        bounds_ep = [(-10, 10)] * 6 + [(0.0, 5.0)]
        try:
            result = minimize(epigraph_obj, x0, method='L-BFGS-B',
                              bounds=bounds_ep, options={'maxiter': 100, 'ftol': 1e-12})
            if result.success or result.fun < epigraph_obj(x0):
                new_coeffs = np.clip(result.x[:6], -10.0, 10.0)
                if check_gamma_range(new_coeffs):
                    coeffs = new_coeffs
        except Exception:
            pass

    # Phase 3: Final polish with high-precision reweighting
    if time.time() < deadline - 3.0:
        # Multiple passes of reweighting with tighter focus
        for polish_iter in range(3):
            if time.time() > deadline - 3.0:
                break
            errors = compute_errors(coeffs)
            if len(errors) == 0:
                break
            max_err = np.max(errors)
            if max_err < 1e-8:
                break
            # Very sharp weighting on worst points
            temp = max_err * 0.1
            if temp < 1e-8:
                temp = 1e-8
            w = np.exp(errors / temp)
            w = w / np.sum(w)
            Aw = A * w[:, None]
            bw = b * w
            try:
                coeffs_new, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=1e-14)
            except Exception:
                break
            coeffs_new = np.clip(coeffs_new, -10.0, 10.0)
            if check_gamma_range(coeffs_new):
                coeffs = coeffs_new

    # Final validation
    if not check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END
