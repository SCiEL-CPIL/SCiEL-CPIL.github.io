import numpy as np
import time
from scipy.optimize import minimize

def _softplus(x):
    """Numerically stable softplus: log(1+exp(x)) without overflow."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def equation(b, s, temp, pH, params):
    p = params

    # Parameters with stable softplus
    K_s      = _softplus(p[0]) + 1e-6
    T_opt    = p[1]
    T_width  = _softplus(p[2]) + 1e-3
    pH_opt   = p[3]
    pH_width = _softplus(p[4]) + 1e-3
    mu_max   = _softplus(p[5])
    b_slope  = p[6]
    b_shift  = p[7]
    mu_base  = p[8]
    s_inh    = _softplus(p[9]) + 1e-6  # Substrate inhibition strength (Haldane)

    # Normalizations to keep parameters O(1) for BFGS starting at 1
    # b: [1, 5] -> [0, 1]
    bn = (b - 1.0) / 4.0
    tn = (temp - 20.0) / 10.0
    phn = (pH - 7.0) / 5.0

    # Monod with Haldane Inhibition
    # s is in [1, 40], K_s ~ 1, s_inh ~ 10
    # s^2 / s_inh can be up to 1600/10 = 160.
    # Denominator: K_s + s + s^2/s_inh. Max ~ 1 + 40 + 160 = 201.
    # This is safe.
    sfac = s / (K_s + s + (s**2) / (s_inh + 1e-9))

    # Temperature Bell
    # tn ~ [0, 1], T_opt ~ 1.7 (37C), T_width ~ 0.5
    # ((tn - T_opt) / T_width)^2 can be large if T_opt is far.
    # Clip exponent to avoid underflow/overflow in exp (though exp(-large) is just 0)
    t_arg = np.clip(-0.5 * ((tn - T_opt) / T_width)**2, -100, 0)
    tfac = np.exp(t_arg)

    # pH Bell
    # phn ~ [0, 1], pH_opt ~ 0, pH_width ~ 1
    p_arg = np.clip(-0.5 * ((phn - pH_opt) / pH_width)**2, -100, 0)
    pfac = np.exp(p_arg)

    # Density Inhibition
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = 1.0 / (1.0 + np.exp(arg))

    return mu_max * sfac * tfac * pfac * dfac + mu_base

def _objective(params, b, s, temp, pH, y):
    try:
        pred = equation(b, s, temp, pH, params)
        if not np.all(np.isfinite(pred)):
            return 1e30
        return np.mean((pred - y)**2)
    except Exception:
        return 1e30

def _gradient(params, b, s, temp, pH, y):
    """
    Analytic gradient of MSE w.r.t. params.
    """
    p = params
    n = len(p)
    grad = np.zeros(n)

    # Recompute intermediates
    K_s      = _softplus(p[0]) + 1e-6
    T_opt    = p[1]
    T_width  = _softplus(p[2]) + 1e-3
    pH_opt   = p[3]
    pH_width = _softplus(p[4]) + 1e-3
    mu_max   = _softplus(p[5])
    b_slope  = p[6]
    b_shift  = p[7]
    mu_base  = p[8]
    s_inh    = _softplus(p[9]) + 1e-6

    bn = (b - 1.0) / 4.0
    tn = (temp - 20.0) / 10.0
    phn = (pH - 7.0) / 5.0

    # sfac = s / (K_s + s + s^2/s_inh)
    s2 = s**2
    denom_s = K_s + s + s2 / (s_inh + 1e-9)
    sfac = s / denom_s

    # tfac = exp(-0.5 * ((tn - T_opt)/T_width)^2)
    t_arg = -0.5 * ((tn - T_opt) / T_width)**2
    tfac = np.exp(np.clip(t_arg, -100, 0))

    # pfac = exp(-0.5 * ((phn - pH_opt)/pH_width)^2)
    p_arg = -0.5 * ((phn - pH_opt) / pH_width)**2
    pfac = np.exp(np.clip(p_arg, -100, 0))

    # dfac = 1 / (1 + exp(b_slope*(bn - b_shift)))
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = 1.0 / (1.0 + np.exp(arg))

    pred = mu_max * sfac * tfac * pfac * dfac + mu_base
    err = pred - y
    # dMSE/dpred = 2 * err / N
    d_pred = 2.0 * err / len(y)

    # d(pred)/d(mu_max) = sfac*tfac*pfac*dfac
    d_mu_max = sfac * tfac * pfac * dfac
    # d(pred)/d(mu_base) = 1
    d_mu_base = np.ones_like(mu_base)

    # d(pred)/d(K_s) = mu_max * tfac*pfac*dfac * d(sfac)/d(K_s)
    # d(sfac)/d(K_s) = -s / (K_s + s + s^2/s_inh)^2
    d_sfac_dK = -s / (denom_s**2)
    d_K_s = mu_max * tfac * pfac * dfac * d_sfac_dK

    # d(pred)/d(s_inh) = mu_max * tfac*pfac*dfac * d(sfac)/d(s_inh)
    # d(sfac)/d(s_inh) = s * s^2 / (s_inh + 1e-9)^2 / denom_s^2
    d_sfac_ds_inh = s * s2 / ((s_inh + 1e-9)**2 * denom_s**2)
    d_s_inh = mu_max * tfac * pfac * dfac * d_sfac_ds_inh

    # d(pred)/d(T_opt) = mu_max * sfac*pfac*dfac * d(tfac)/d(T_opt)
    # d(tfac)/d(T_opt) = tfac * (tn - T_opt) / T_width^2
    d_tfac_dT_opt = tfac * (tn - T_opt) / (T_width**2)
    d_T_opt = mu_max * sfac * pfac * dfac * d_tfac_dT_opt

    # d(pred)/d(T_width) = mu_max * sfac*pfac*dfac * d(tfac)/d(T_width)
    # d(tfac)/d(T_width) = tfac * (tn - T_opt)^2 / T_width^3
    d_tfac_dT_width = tfac * (tn - T_opt)**2 / (T_width**3)
    d_T_width = mu_max * sfac * pfac * dfac * d_tfac_dT_width

    # d(pred)/d(pH_opt) = mu_max * sfac*tfac*dfac * d(pfac)/d(pH_opt)
    d_pfac_dpH_opt = pfac * (phn - pH_opt) / (pH_width**2)
    d_pH_opt = mu_max * sfac * tfac * dfac * d_pfac_dpH_opt

    # d(pred)/d(pH_width) = mu_max * sfac*tfac*dfac * d(pfac)/d(pH_width)
    d_pfac_dpH_width = pfac * (phn - pH_opt)**2 / (pH_width**3)
    d_pH_width = mu_max * sfac * tfac * dfac * d_pfac_dpH_width

    # d(pred)/d(b_slope) = mu_max * sfac*tfac*pfac * d(dfac)/d(b_slope)
    # d(dfac)/d(b_slope) = dfac * (1 - dfac) * (bn - b_shift)
    d_dfac_db_slope = dfac * (1.0 - dfac) * (bn - b_shift)
    d_b_slope = mu_max * sfac * tfac * pfac * d_dfac_db_slope

    # d(pred)/d(b_shift) = mu_max * sfac*tfac*pfac * d(dfac)/d(b_shift)
    d_dfac_db_shift = -dfac * (1.0 - dfac) * b_slope
    d_b_shift = mu_max * sfac * tfac * pfac * d_dfac_db_shift

    # d(pred)/d(mu_max) = sfac*tfac*pfac*dfac
    # d(pred)/d(mu_base) = 1

    # Chain rule: dMSE/dparam = sum(d_pred * dpred/dparam)
    grad[0] = np.sum(d_pred * d_K_s)
    grad[1] = np.sum(d_pred * d_T_opt)
    grad[2] = np.sum(d_pred * d_T_width)
    grad[3] = np.sum(d_pred * d_pH_opt)
    grad[4] = np.sum(d_pred * d_pH_width)
    grad[5] = np.sum(d_pred * d_mu_max)
    grad[6] = np.sum(d_pred * d_b_slope)
    grad[7] = np.sum(d_pred * d_b_shift)
    grad[8] = np.sum(d_pred * d_mu_base)
    grad[9] = np.sum(d_pred * d_s_inh)

    return grad

def fit_params(b, s, temp, pH, y, timeout=140.0):
    start_time = time.time()
    deadline = start_time + timeout

    # Initial guesses
    # K_s ~ 1, T_opt ~ 1.7 (37C), T_width ~ 0.5, pH_opt ~ 0, pH_width ~ 1, mu_max ~ 1.5, b_slope ~ 1, b_shift ~ 0.5, mu_base ~ 0, s_inh ~ 10
    x0 = np.array([0.0, 1.7, 0.5, 0.0, 1.0, 0.4, 1.0, 0.5, 0.0, 2.3])

    best_x = x0
    best_f = _objective(x0, b, s, temp, pH, y)

    # Try a few random restarts if time permits, otherwise just BFGS from x0
    # Since we have 150s limit, and BFGS is fast, we can try 2-3 restarts.

    restarts = [
        x0,
        x0 + np.array([0.5, 0.1, 0.2, 0.1, 0.2, 0.2, 0.5, 0.2, 0.0, 0.5]),
        x0 + np.array([-0.5, -0.1, -0.2, -0.1, -0.2, -0.2, -0.5, -0.2, 0.0, -0.5]),
        x0 + np.array([1.0, 0.2, 0.3, 0.2, 0.3, 0.3, 1.0, 0.3, 0.0, 1.0]),
        x0 + np.array([-1.0, -0.2, -0.3, -0.2, -0.3, -0.3, -1.0, -0.3, 0.0, -1.0])
    ]

    for x_init in restarts:
        if time.time() > deadline - 10:  # Safety margin
            break

        try:
            # Use L-BFGS-B for better large-scale convergence
            res = minimize(_objective, x_init, args=(b, s, temp, pH, y), method='L-BFGS-B',
                           jac=_gradient,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})

            if res.success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

        # Fallback to BFGS if L-BFGS-B failed or didn't improve
        if time.time() > deadline - 15:
            break

        try:
            res = minimize(_objective, x_init, args=(b, s, temp, pH, y), method='BFGS',
                           jac=_gradient,
                           options={'maxiter': 200, 'gtol': 1e-6})

            if res.success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final polish with tighter tolerance if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_x, args=(b, s, temp, pH, y), method='L-BFGS-B',
                           jac=_gradient,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x
