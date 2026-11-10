import numpy as np
import scipy.optimize
import scipy.integrate
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # ============================================================
    # STRATEGY: Local Taylor expansion around a=1
    # Derive exact f(1), f'(1), f''(1) and D'(1), D''(1) analytically
    # for a central cosmology, then solve for 6 coefficients that
    # match these 6 Taylor conditions.
    # ============================================================

    # Central cosmology for the Taylor match
    om_m0_c = 0.30
    w0_c = -1.0
    wa_c = 0.0

    # --- Analytic derivatives of Omega_m(a) at a=1 ---
    # E^2(a) = om_m0 * a^-3 + om_l0 * a^{-3(1+w0)} * exp(-3*wa*(a-1))
    # Let A = om_m0, B = om_l0, alpha = -3(1+w0), beta = -3*wa
    # DE term: B * a^alpha * exp(beta*(a-1))
    # d/da [a^alpha * exp(beta*(a-1))] at a=1 = alpha + beta
    # d2/da2 at a=1 = alpha*(alpha-1) + 2*alpha*beta + beta^2

    A = om_m0_c
    B = 1.0 - om_m0_c
    alpha = -3.0 * (1.0 + w0_c)
    beta = -3.0 * wa_c

    # E^2(1) = A + B = 1
    E2_1 = A + B  # = 1.0

    # dE2/da at a=1:
    # d/da [A*a^-3] = -3A
    # d/da [B*a^alpha*exp(beta*(a-1))] = B*(alpha + beta)
    dE2_1 = -3.0 * A + B * (alpha + beta)

    # d2E2/da2 at a=1:
    # d2/da2 [A*a^-3] = 6A
    # d2/da2 [B*a^alpha*exp(beta*(a-1))] = B*(alpha*(alpha-1) + 2*alpha*beta + beta^2)
    d2E2_1 = 6.0 * A + B * (alpha * (alpha - 1.0) + 2.0 * alpha * beta + beta ** 2)

    # Omega_m = A*a^-3 / E^2
    # Om_m(1) = A
    # dOm_m/da at 1: (d/d a [A a^-3] * E2 - A a^-3 * dE2/da) / E2^2  at a=1
    #   = (-3A * 1 - A * dE2_1) / 1  = -3A - A*dE2_1
    dOm_1 = -3.0 * A - A * dE2_1

    # d2Om_m/da2 at 1: use quotient rule more carefully
    # Om = N/D where N = A*a^-3, D = E2
    # Om' = (N'D - ND')/D^2
    # Om'' = (N''D - ND')/D^2 - 2(N'D - ND')D'/D^3
    #      = N''/D - N D'/D^2 - 2(N'D - ND')D'/D^3
    # At a=1: D=1, N=A, N'=-3A, N''=6A
    N_1 = A
    Np_1 = -3.0 * A
    Npp_1 = 6.0 * A
    D_1 = E2_1
    Dp_1 = dE2_1
    Dpp_1 = d2E2_1

    Om_1 = N_1 / D_1
    Om_p1 = (Np_1 * D_1 - N_1 * Dp_1) / D_1 ** 2
    Om_pp1 = (Npp_1 * D_1 - N_1 * Dpp_1) / D_1 ** 2 - 2.0 * (Np_1 * D_1 - N_1 * Dp_1) * Dp_1 / D_1 ** 3

    # --- f(a) = Om_m(a)^gamma(a) ---
    # We need f(1), f'(1), f''(1) in terms of the coefficients.
    # At a=1: Om=om_m0_c, q=1-om_m0_c, u=1+w0_c+wa_c*(1-1)=1+w0_c
    # gamma(1) = c0 + c1*q1 + c2*q1^2 + c3*u1 + c4*wa_c*1 + c5*u1*q1
    # where q1 = 1-om_m0_c, u1 = 1+w0_c

    q1 = 1.0 - om_m0_c
    u1 = 1.0 + w0_c  # wa*(1-a)=0 at a=1

    # Derivatives of gamma at a=1:
    # q = 1 - Om_m, so q' = -Om_m', q'' = -Om_m''
    # u = 1 + w0 + wa*(1-a), so u' = -wa, u'' = 0
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # gamma' = c1*q' + 2*c2*q*q' + c3*u' + c4*wa + c5*(u'*q + u*q')
    # gamma'' = c1*q'' + 2*c2*(q'^2 + q*q'') + c3*u'' + c5*(u''*q + 2*u'*q' + u*q'')
    #         = c1*q'' + 2*c2*(q'^2 + q*q'') + c5*(2*u'*q' + u*q'')

    qp1 = -Om_p1
    qpp1 = -Om_pp1
    up1 = -wa_c
    upp1 = 0.0

    # f = Om^gamma => ln f = gamma * ln Om
    # f' = f * (gamma' * ln Om + gamma * Om'/Om)
    # f'' = f * [(gamma'*lnOm + gamma*Om'/Om)^2 + gamma''*lnOm + gamma'*Om'/Om + gamma*(Om''/Om - (Om'/Om)^2)]

    lnOm_1 = np.log(Om_1)
    Om_ratio_1 = Om_p1 / Om_1
    Om_ratio2_1 = Om_pp1 / Om_1 - (Om_p1 / Om_1) ** 2

    # Let g = gamma(1), gp = gamma'(1), gpp = gamma''(1)
    # f(1) = Om^g
    # f'(1) = f(1) * (gp * lnOm + g * Om_ratio)
    # f''(1) = f(1) * [(gp*lnOm + g*Om_ratio)^2 + gpp*lnOm + gp*Om_ratio + g*Om_ratio2]

    # We need the true f(1), f'(1), f''(1) from the ODE.
    # The ODE: D'' + (3/2)(1+w) D' - (3/2) Om_m D = 0
    # f = D'/D => f' = D''/D - (D'/D)^2 = -(3/2)(1+w)f + (3/2)Om - f^2
    # At a=1: f(1) = Om_m0^gamma_true, but we need the TRUE f from the ODE.
    # Actually f(1) is determined by the ODE initial condition.
    # For the true solution, f(1) is whatever the ODE gives.
    # But we can compute f_true(1) by integrating the ODE numerically.

    # Let's numerically integrate the ODE to get true f and D at a=1 and nearby.
    # This gives us the target values for the Taylor match.

    def om_m_of_a(a, om0, w0, wa):
        """Compute Omega_m(a) for CPL dark energy."""
        if a <= 0:
            return om0
        de = (1.0 - om0) * (a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a - 1.0))
        m = om0 * (a ** -3.0)
        return m / (m + de)

    def ode_system(x, y, om0, w0, wa):
        a = np.exp(x)
        om_m = om_m_of_a(a, om0, w0, wa)
        w_a = w0 + wa * (1.0 - a)
        dD = y[1]
        dDp = -1.5 * (1.0 + w_a) * y[1] + 1.5 * om_m * y[0]
        return [dD, dDp]

    # Integrate the ODE for the central cosmology to get true f and D
    # We need f(1), f'(1), f''(1), D'(1), D''(1)
    # f(1) = D'(1)/D(1) = D'(1) since D(1)=1
    # We need to determine D'(1) from the ODE. The ODE is second order,
    # so we need D(1)=1 and D'(1)=f(1). But f(1) is not known a priori.
    # However, for the growth factor, the standard convention is D(1)=1 and
    # f(1) = Om_m0^gamma where gamma is the "true" growth index at a=1.
    # Actually, the ODE is homogeneous, so D'(1) is a free parameter.
    # The standard is to set D(1)=1 and f(1) = D'(1).
    # We can determine f(1) by requiring the solution to be the growing mode.
    # For a flat universe, f(1) ≈ Om_m0^0.55 is a good approximation.
    # But for exact Taylor matching, we need the TRUE f(1).
    # Let's use a very accurate numerical integration.

    # Actually, for the ODE D'' + (3/2)(1+w)D' - (3/2)Om D = 0,
    # at a=1 (x=0): D''(0) + (3/2)(1+w0)D'(0) - (3/2)Om_m0 D(0) = 0
    # With D(0)=1: D''(0) = -(3/2)(1+w0)D'(0) + (3/2)Om_m0
    # And f(0) = D'(0)/D(0) = D'(0)
    # f'(0) = D''(0)/D(0) - (D'(0)/D(0))^2 = D''(0) - f(0)^2
    #       = -(3/2)(1+w0)f(0) + (3/2)Om_m0 - f(0)^2

    # The issue is that f(0) = D'(0) is a free parameter of the ODE.
    # The "true" growth factor is the one that matches the linear perturbation theory.
    # For a flat universe, the growing mode at late times gives f(1) ≈ Om_m0^0.55.
    # But we can determine it more precisely by integrating from early times.

    # Let's integrate from a small value to a=1 and match.
    # Actually, the standard approach: integrate the ODE from a=0.01 to a=1
    # with initial conditions D(a_small) and D'(a_small) chosen so that
    # the solution is the growing mode. Then normalize D(1)=1.

    # Simpler approach: use the known fact that for the growing mode,
    # f(1) can be computed by integrating the ODE from a=0.01 to 1
    # with D(0.01)=1, D'(0.01)=Om(0.01)^0.55 (approximate), then
    # normalize and read off f(1).

    # Even simpler: just use f(1) = Om_m0^0.55 as a starting point,
    # integrate the ODE forward from a=1 to check consistency, and
    # use the ODE to get f'(1) and f''(1) from f(1).

    # Let me just numerically integrate to get all the derivatives.
    # I'll integrate from x=0 (a=1) to x=-5 (a≈0.0067) with
    # initial conditions D(0)=1, D'(0)=f_seed, then compute
    # f at several nearby points and fit a polynomial to get derivatives.

    f_seed = om_m0_c ** 0.55

    # Integrate ODE from 0 to -3 (a from 1 to ~0.05)
    x_eval = np.linspace(0, -3.0, 1000)
    sol = scipy.integrate.solve_ivp(
        lambda x, y: ode_system(x, y, om_m0_c, w0_c, wa_c),
        [0, -3.0], [1.0, f_seed],
        method='DOP853', rtol=1e-12, atol=1e-14,
        dense_output=True, max_step=0.01
    )

    if not sol.success:
        # Fallback: use constant gamma
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Evaluate at fine grid near x=0 to get derivatives
    x_fine = np.linspace(0, -0.1, 200)
    y_fine = sol.sol(x_fine)
    D_fine = y_fine[0]
    Dp_fine = y_fine[1]
    f_fine = Dp_fine / D_fine

    # Fit f(x) = f0 + f1*x + f2*x^2/2 near x=0
    # Use the first few points
    n_fit = 50
    x_fit = x_fine[:n_fit]
    f_fit = f_fine[:n_fit]
    # Polynomial fit: f ≈ a0 + a1*x + a2*x^2
    coeffs_fit = np.polyfit(x_fit, f_fit, 2)
    f0_true = coeffs_fit[2]  # f(0)
    f1_true = coeffs_fit[1]  # f'(0) (derivative w.r.t. x = ln a)
    f2_true = 2.0 * coeffs_fit[0]  # f''(0)

    # Similarly for D(x):
    D_fit = np.polyfit(x_fit, D_fine[:n_fit], 2)
    D0_true = D_fit[2]  # should be 1
    D1_true = D_fit[1]  # D'(0) = f(0)
    D2_true = 2.0 * D_fit[0]  # D''(0)

    # Now set up the Taylor matching conditions.
    # We want our ansatz to match:
    # 1. f_ansatz(1) = f0_true
    # 2. f_ansatz'(1) = f1_true  (derivative w.r.t. x = ln a)
    # 3. f_ansatz''(1) = f2_true
    # 4. D_ansatz(1) = 1 (automatic since exp(0)=1)
    # 5. D_ansatz'(1) = D1_true  (which should equal f0_true)
    # 6. D_ansatz''(1) = D2_true

    # Note: D_ansatz(x) = exp(-∫_0^x f(e^t) dt)
    # D_ansatz'(x) = -f(e^x) * D_ansatz(x)  => D'(0) = -f(0)*1 = -f0
    # Wait, let me re-check the sign convention.
    # D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx) = exp(∫_0^{ln a} f(e^x) dx)  ??? 
    # Let me re-read the task: "D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)"
    # So D(a)/D(1) = exp(-∫_{ln a}^{0} f(e^x) dx)
    # Let I(x) = ∫_{ln a}^{0} f(e^t) dt where ln a = x
    # D = exp(-I(x))
    # I'(x) = -f(e^x)  (by Leibniz rule, d/dx ∫_x^0 g(t)dt = -g(x))
    # D' = -I' * D = f(e^x) * D
    # So D'(0) = f(0) * D(0) = f0 * 1 = f0  ✓
    # D'' = f'(e^x)*D + f(e^x)*D' = f'*D + f^2*D  (at x=0: f' is df/dx)
    # D''(0) = f1_true + f0_true^2

    # Let me verify: D''(0) from our numerical fit should be ≈ f1 + f0^2
    # D2_true should be close to f1_true + f0_true^2

    # Now, f_ansatz = Om_m(x)^gamma(x) where x = ln a, a = e^x
    # We need df/dx and d2f/dx2 at x=0.
    # Let L = ln Om, so f = e^{gamma * L}
    # df/dx = f * (gamma' * L + gamma * L')  where primes are d/dx
    # d2f/dx2 = f * [(gamma'*L + gamma*L')^2 + gamma''*L + gamma'*L' + gamma*L'']

    # At x=0 (a=1):
    # L(0) = ln(om_m0_c)
    # L'(0) = Om_m'/Om_m = Om_p1/Om_1 (derivative w.r.t. a, but we need w.r.t. x=ln a)
    # d/da Om_m at a=1 is Om_p1, so d/dx Om_m = a * d/da Om_m = 1 * Om_p1 = Om_p1
    # L'(0) = Om_p1 / Om_1
    # L''(0) = d/dx (Om'/Om) = a*d/da(Om'/Om) = (Om''/Om - (Om'/Om)^2) at a=1
    #        = Om_pp1/Om_1 - (Om_p1/Om_1)^2

    L0 = np.log(Om_1)
    L1 = Om_p1 / Om_1
    L2 = Om_pp1 / Om_1 - (Om_p1 / Om_1) ** 2

    # gamma(x) = c0 + c1*q(x) + c2*q(x)^2 + c3*u(x) + c4*wa_c*e^x + c5*u(x)*q(x)
    # q(x) = 1 - Om_m(x)
    # u(x) = 1 + w0_c + wa_c*(1 - e^x)

    # At x=0: q0 = q1 = 1-om_m0_c, u0 = u1 = 1+w0_c
    # q'(x) = -Om_m'(x) => q'(0) = -Om_p1 (w.r.t. x)
    # q''(x) = -Om_m''(x) => q''(0) = -Om_pp1
    # u'(x) = -wa_c * e^x => u'(0) = -wa_c
    # u''(x) = -wa_c * e^x => u''(0) = -wa_c

    q0 = q1
    u0 = u1
    qp0 = -Om_p1
    qpp0 = -Om_pp1
    up0 = -wa_c
    upp0 = -wa_c

    # gamma(0) = c0 + c1*q0 + c2*q0^2 + c3*u0 + c4*wa_c*1 + c5*u0*q0
    # gamma'(0) = c1*qp0 + 2*c2*q0*qp0 + c3*up0 + c4*wa_c*1 + c5*(up0*q0 + u0*qp0)
    # gamma''(0) = c1*qpp0 + 2*c2*(qp0^2 + q0*qpp0) + c3*upp0 + c5*(upp0*q0 + 2*up0*qp0 + u0*qpp0)

    # Define the three conditions for f:
    # f(0) = Om_1^gamma(0) = f0_true
    # f'(0) = f(0) * (gamma'(0)*L0 + gamma(0)*L1) = f1_true
    # f''(0) = f(0) * [(gamma'*L0 + gamma*L1)^2 + gamma''*L0 + gamma'*L1 + gamma*L2] = f2_true

    # And for D:
    # D(0) = 1 (automatic)
    # D'(0) = f(0) = f0_true (same as condition 1)
    # D''(0) = f'(0) + f(0)^2 = f1_true + f0_true^2 = D2_true (should be consistent)

    # So we have 3 independent conditions from f, and D conditions are dependent.
    # We have 6 unknowns (c0..c5) and only 3 independent equations.
    # We need 3 more conditions. Let's add:
    # - Match gamma(0) to a reasonable value (e.g., 0.55 or the implied value)
    # - Match q-dependence: make gamma linear in q to first order
    # - Or: minimize residual error on a secondary cosmology

    # Better approach: use 3 conditions from f Taylor + 3 from D Taylor
    # But D conditions are dependent on f conditions. So we only have 3 independent.
    # We need a different strategy for the remaining 3 degrees of freedom.

    # ALTERNATIVE: Use the 3 f-Taylor conditions to determine 3 of the 6 coefficients,
    # then use the remaining 3 to minimize error on a secondary cosmology.

    # Let's set up a least-squares problem:
    # Variables: c0, c1, c2, c3, c4, c5
    # Residuals: 
    #   r1 = f_ansatz(0) - f0_true
    #   r2 = f_ansatz'(0) - f1_true  
    #   r3 = f_ansatz''(0) - f2_true
    #   r4 = f_ansatz on secondary cosmology (a few points)
    #   r5 = D_ansatz on secondary cosmology (a few points)
    #   r6 = gamma bounds penalty

    # Actually, let me just do a full optimization but with the Taylor conditions
    # as strong constraints (large weight) plus a secondary cosmology fit.

    # Define the secondary cosmology for generalization
    om_m0_s = 0.32
    w0_s = -0.8
    wa_s = 0.3

    # Grid for secondary cosmology
    a_sec = np.logspace(np.log10(0.1), 0.0, 50)
    x_sec = np.log(a_sec)

    # Integrate ODE for secondary cosmology
    f_seed_s = om_m0_s ** 0.55
    sol_s = scipy.integrate.solve_ivp(
        lambda x, y: ode_system(x, y, om_m0_s, w0_s, wa_s),
        [0, -3.0], [1.0, f_seed_s],
        method='DOP853', rtol=1e-12, atol=1e-14,
        dense_output=True, max_step=0.01
    )

    D_sec_ref = np.ones_like(a_sec)
    f_sec_ref = np.ones_like(a_sec)
    if sol_s.success:
        y_sec = sol_s.sol(x_sec)
        D_sec_ref = y_sec[0]
        f_sec_ref = y_sec[1] / y_sec[0]

    # Objective function for optimization
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        # Check bounds
        if np.any(np.abs(coeffs) > 10):
            return 1e10

        # Taylor conditions at central cosmology
        g0 = c0 + c1*q0 + c2*q0**2 + c3*u0 + c4*wa_c + c5*u0*q0
        gp0 = c1*qp0 + 2*c2*q0*qp0 + c3*up0 + c4*wa_c + c5*(up0*q0 + u0*qp0)
        gpp0 = c1*qpp0 + 2*c2*(qp0**2 + q0*qpp0) + c3*upp0 + c5*(upp0*q0 + 2*up0*qp0 + u0*qpp0)

        # f(0) = Om_1^g0
        if g0 < 0.1 or g0 > 1.2:
            return 1e10
        f0 = Om_1 ** g0
        f1 = f0 * (gp0 * L0 + g0 * L1)
        f2 = f0 * ((gp0 * L0 + g0 * L1)**2 + gpp0 * L0 + gp0 * L1 + g0 * L2)

        r1 = f0 - f0_true
        r2 = f1 - f1_true
        r3 = f2 - f2_true

        # Secondary cosmology errors
        om_m_sec = np.array([om_m_of_a(a, om_m0_s, w0_s, wa_s) for a in a_sec])
        q_sec = 1.0 - om_m_sec
        u_sec = 1.0 + w0_s + wa_s * (1.0 - a_sec)
        gamma_sec = c0 + c1*q_sec + c2*q_sec**2 + c3*u_sec + c4*wa_s*a_sec + c5*u_sec*q_sec

        if np.any(gamma_sec < 0.1) or np.any(gamma_sec > 1.2):
            return 1e10

        f_sec_approx = om_m_sec ** gamma_sec
        # D integration
        dx = x_sec[1] - x_sec[0]
        integrals = np.zeros_like(x_sec)
        for i in range(1, len(x_sec)):
            integrals[i] = integrals[i-1] + 0.5 * (f_sec_approx[i-1] + f_sec_approx[i]) * dx
        # Reverse: x_sec goes from log(0.1) to 0, we need integral from 0 to x
        # integrals[i] = ∫_{x_sec[0]}^{x_sec[i]} f dx
        # We want ∫_{x}^{0} f dx = integrals[-1] - integrals[i]
        D_sec_approx = np.exp(-(integrals[-1] - integrals))

        err_f_sec = (f_sec_approx - f_sec_ref) / f_sec_ref
        err_D_sec = (D_sec_approx - D_sec_ref) / D_sec_ref

        # Weighted sum of squares
        weight_taylor = 100.0  # Strong weight on Taylor conditions
        weight_sec = 1.0

        obj = weight_taylor * (r1**2 + r2**2 + r3**2) + weight_sec * (np.sum(err_f_sec**2) + np.sum(err_D_sec**2))
        return obj

    # Initial guess: constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Optimize
    best_coeffs = x0.copy()
    best_obj = objective(x0)

    try:
        result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B',
                                         bounds=[(-10, 10)] * 6,
                                         options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < best_obj:
            best_coeffs = result.x
            best_obj = result.fun

        # Polish with Nelder-Mead
        if time.time() < deadline - 15.0:
            result2 = scipy.optimize.minimize(objective, best_coeffs, method='Nelder-Mead',
                                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result2.fun < best_obj:
                best_coeffs = result2.x
                best_obj = result2.fun

    except Exception:
        pass

    # Final validation
    c = np.clip(best_coeffs, -10, 10)

    # Check gamma bounds on a dense grid for multiple cosmologies
    test_cosmos = [(0.25, -1.2, -0.5), (0.3, -1.0, 0.0), (0.35, -0.8, 0.5),
                   (0.2, -1.5, -1.0), (0.4, -0.5, 1.0)]
    test_a = np.logspace(np.log10(0.05), 0.0, 100)
    valid = True
    for (om0, w0, wa) in test_cosmos:
        om_m_t = np.array([om_m_of_a(a, om0, w0, wa) for a in test_a])
        q_t = 1.0 - om_m_t
        u_t = 1.0 + w0 + wa * (1.0 - test_a)
        gamma_t = c[0] + c[1]*q_t + c[2]*q_t**2 + c[3]*u_t + c[4]*wa*test_a + c[5]*u_t*q_t
        if np.any(gamma_t < 0.1) or np.any(gamma_t > 1.2):
            valid = False
            break

    if not valid:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return c.tolist()
# EVOLVE-BLOCK-END
