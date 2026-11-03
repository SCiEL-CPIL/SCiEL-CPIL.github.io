# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Precompute constants for analytic integration
    # I_k = int x^(2k) exp(-x^2) dx from -inf to inf = 0.5 * Gamma(k + 0.5)
    # For P degree 12, P^2 degree 24 -> max k=12
    # For P', degree 11, (P')^2 degree 22 -> max k=11
    I_k = np.zeros(13)
    I_k[0] = np.sqrt(np.pi) / 2.0
    for k in range(1, 13):
        I_k[k] = I_k[k-1] * (k - 0.5)

    # Probabilists' Hermite polynomials in monomial form
    # Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    He0_coeffs = np.zeros(7)
    He0_coeffs[0] = 1.0

    He4_coeffs = np.zeros(7)
    He4_coeffs[0] = 3.0
    He4_coeffs[1] = -6.0
    He4_coeffs[2] = 1.0

    He8_coeffs = np.zeros(7)
    He8_coeffs[0] = 105.0
    He8_coeffs[1] = -420.0
    He8_coeffs[2] = 210.0
    He8_coeffs[3] = -28.0
    He8_coeffs[4] = 1.0

    He12_coeffs = np.zeros(7)
    He12_coeffs[0] = 9240.0
    He12_coeffs[1] = -12096.0
    He12_coeffs[2] = 30240.0
    He12_coeffs[3] = -15120.0
    He12_coeffs[4] = 1605.0
    He12_coeffs[5] = -42.0
    He12_coeffs[6] = 1.0

    def get_P_coeffs(c0, c4, c8):
        # P(0) = c0 + 3*c4 + 105*c8 + 9240*c12 = 0
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 9240.0
        P_coeffs = c0 * He0_coeffs + c4 * He4_coeffs + c8 * He8_coeffs + c12 * He12_coeffs
        return P_coeffs

    def compute_integrals(a):
        """
        Compute I0, I1, I2 from monomial coefficients a of P(x)
        P(x) = sum a_j x^(2j)
        I0 = int P^2 exp(-x^2) dx
        I1 = int x^2 P^2 exp(-x^2) dx
        I2 = int (P')^2 exp(-x^2) dx
        """
        I0 = 0.0
        for i in range(7):
            if a[i] == 0: continue
            for j in range(i, 7):
                if a[j] == 0: continue
                term = a[i] * a[j] * I_k[i+j]
                if i != j:
                    term *= 2.0
                I0 += term

        I1 = 0.0
        for i in range(7):
            if a[i] == 0: continue
            for j in range(i, 7):
                if a[j] == 0: continue
                term = a[i] * a[j] * I_k[i+j+1]
                if i != j:
                    term *= 2.0
                I1 += term

        I2 = 0.0
        for i in range(1, 7):
            if a[i] == 0: continue
            bi = 2.0 * i * a[i]
            for j in range(i, 7):
                if a[j] == 0: continue
                bj = 2.0 * j * a[j]
                term = bi * bj * I_k[i+j-1]
                if i != j:
                    term *= 2.0
                I2 += term

        return I0, I1, I2

    def objective(params):
        if time.time() > deadline:
            return 1e10
        c0, c4, c8 = params
        a = get_P_coeffs(c0, c4, c8)
        I0, I1, I2 = compute_integrals(a)
        if I0 < 1e-12:
            return 1e10
        return (I1 * I2) / (I0 * I0)

    def analytic_gradient(params):
        if time.time() > deadline:
            return np.zeros(3)
        c0, c4, c8 = params
        a = get_P_coeffs(c0, c4, c8)
        I0, I1, I2 = compute_integrals(a)
        if I0 < 1e-12:
            return np.zeros(3)

        # Gradient of (I1*I2)/(I0^2) with respect to c0, c4, c8
        # d/dc_k = (dI1/dc_k * I2 * I0^2 - I1*I2*2*I0*dI0/dc_k) / I0^4
        #        = (dI1/dc_k * I2 * I0 - 2*I1*I2*dI0/dc_k) / I0^3

        # Need dI0/dc_k, dI1/dc_k, dI2/dc_k
        # I0 = sum a_i a_j I_k[i+j], so dI0/dc_k = 2 * sum_j (da_k/dc_k)_j a_j I_k[k+j]
        # where da_k/dc_m is the derivative of coefficient a_k with respect to c_m

        # a = M * [c0, c4, c8, c12] where c12 = -1/9240*(c0 + 3*c4 + 105*c8)
        # So da_k/dc_0 = He0_coeffs[k] - He12_coeffs[k]/9240
        #    da_k/dc_4 = He4_coeffs[k] - 3*He12_coeffs[k]/9240
        #    da_k/dc_8 = He8_coeffs[k] - 105*He12_coeffs[k]/9240

        da_dc0 = He0_coeffs - He12_coeffs / 9240.0
        da_dc4 = He4_coeffs - 3.0 * He12_coeffs / 9240.0
        da_dc8 = He8_coeffs - 105.0 * He12_coeffs / 9240.0

        def dI_dcoeff(da_dc, I_k_arr):
            """dI/dc where I = sum a_i a_j I_k_arr[i+j] (or similar)"""
            # dI/dc = 2 * sum_j da_j a_j I_k_arr[j+k] for each k
            # But we need the vector [dI/dc0, dI/dc4, dI/dc8]
            # dI/dc_m = 2 * sum_{i,j} (da_i/dc_m) a_j I_k_arr[i+j] (with symmetry)
            # Actually: I = sum_{i,j} a_i a_j B[i,j] where B[i,j] = I_k_arr[i+j]
            # dI/dc_m = sum_{i,j} (da_i/dc_m) a_j B[i,j] + sum_{i,j} a_i (da_j/dc_m) B[i,j]
            #         = 2 * sum_{i,j} (da_i/dc_m) a_j B[i,j] (since B is symmetric)
            result = np.zeros(3)
            for m, da in enumerate([da_dc0, da_dc4, da_dc8]):
                s = 0.0
                for i in range(7):
                    if da[i] == 0: continue
                    for j in range(7):
                        if a[j] == 0: continue
                        s += da[i] * a[j] * I_k_arr[i+j]
                result[m] = 2.0 * s
            return result

        dI0_dc = dI_dcoeff(None, I_k)  # placeholder, we'll compute properly below

        # Let's just compute dI0_dc, dI1_dc, dI2_dc directly
        def compute_dI0_da():
            # dI0/da_i = 2 * sum_j a_j I_k[i+j]
            dI0_da = np.zeros(7)
            for i in range(7):
                if a[i] == 0: continue
                s = 0.0
                for j in range(7):
                    if a[j] == 0: continue
                    s += a[j] * I_k[i+j]
                dI0_da[i] = 2.0 * s
            return dI0_da

        def compute_dI1_da():
            # dI1/da_i = 2 * sum_j a_j I_k[i+j+1]
            dI1_da = np.zeros(7)
            for i in range(7):
                if a[i] == 0: continue
                s = 0.0
                for j in range(7):
                    if a[j] == 0: continue
                    s += a[j] * I_k[i+j+1]
                dI1_da[i] = 2.0 * s
            return dI1_da

        def compute_dI2_da():
            # I2 = sum_{i,j=1}^6 (2i a_i)(2j a_j) I_k[i+j-1]
            # dI2/da_k = 2 * (2k) * sum_{j=1}^6 (2j a_j) I_k[k+j-1] for k=1..6
            # = 4k * sum_j (2j a_j) I_k[k+j-1]
            dI2_da = np.zeros(7)
            for k in range(1, 7):
                if a[k] == 0: continue
                s = 0.0
                for j in range(1, 7):
                    if a[j] == 0: continue
                    s += (2.0 * j * a[j]) * I_k[k+j-1]
                dI2_da[k] = 4.0 * k * s
            return dI2_da

        dI0_da = compute_dI0_da()
        dI1_da = compute_dI1_da()
        dI2_da = compute_dI2_da()

        # Now chain rule: dI_m/dc_p = sum_i dI_m/da_i * da_i/dc_p
        dI0_dc = np.zeros(3)
        dI1_dc = np.zeros(3)
        dI2_dc = np.zeros(3)
        for p in range(3):
            da_dc = [da_dc0, da_dc4, da_dc8][p]
            dI0_dc[p] = np.dot(dI0_da, da_dc)
            dI1_dc[p] = np.dot(dI1_da, da_dc)
            dI2_dc[p] = np.dot(dI2_da, da_dc)

        # J = I1*I2/I0^2
        # dJ/dc_p = (dI1/dc_p * I2 * I0 - 2*I1*I2*dI0/dc_p) / I0^3
        denom = I0**3
        grad = np.zeros(3)
        for p in range(3):
            grad[p] = (dI1_dc[p] * I2 * I0 - 2.0 * I1 * I2 * dI0_dc[p]) / denom
        return grad

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    seed = np.array([c0_init, c4_init, c8_init])

    best_score = objective(seed)
    best_coeffs = seed.copy()

    def time_ok(margin=10.0):
        return (time.time() - start_time) < (deadline - margin)

    rng = np.random.default_rng(42)

    # Primary optimization: L-BFGS-B with analytic gradient
    if time_ok(5.0):
        try:
            res = minimize(
                objective,
                seed,
                jac=analytic_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Random restarts with L-BFGS-B
    for _ in range(30):
        if not time_ok(5.0):
            break
        # Random perturbation around seed and current best
        if rng.random() < 0.5:
            start = seed + rng.normal(0, 0.1, 3)
        else:
            start = best_coeffs + rng.normal(0, 0.05, 3)

        try:
            res = minimize(
                objective,
                start,
                jac=analytic_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Additional global exploration: random points in a larger box
    for _ in range(50):
        if not time_ok(3.0):
            break
        start = np.array([
            rng.uniform(-3.0, 0.3),
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.01, 0.01)
        ])
        try:
            val = objective(start)
            if val < best_score:
                best_score = val
                best_coeffs = start.copy()
        except Exception:
            pass

    # Final polish with high precision
    if time_ok(5.0):
        try:
            res = minimize(
                objective,
                best_coeffs,
                jac=analytic_gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
