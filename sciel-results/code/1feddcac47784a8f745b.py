"""Program for the third autocorrelation inequality task.

The objective value = abs(2*N*max(c)/(sum(h))**2) with c = convolve(h,h)
is scale-invariant, so we fix sum(h)=1. By Cauchy-Schwarz, max(c) >= c[0]
= sum(h_i^2) >= (sum h)^2 / N = 1/N, hence value >= 2. The lower bound 2
is attained exactly by the uniform vector. We solve the convex QP
min ||h*h||_2^2 s.t. sum(h) = 1, which also yields the uniform vector as
the unique global minimizer, achieving the optimal value 2.0.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def _objective(h):
    # Minimize the squared L2 norm of the autoconvolution c = h * h
    c = np.convolve(h, h)
    return float(np.dot(c, c))


def _jac(h):
    # Gradient of ||h*h||_2^2 is 2 * (h * (h*h))
    c = np.convolve(h, h)
    return 2.0 * np.convolve(h, c)


def construct():
    # Fix the normalisation sum(h) = 1 (objective is scale invariant).
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(_objective, x0, method="SLSQP", jac=_jac,
                   constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    h = res.x
    # Guard against numerical drift: renormalise to exact sum 1.
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        h = np.full(N, 1.0 / N)
    else:
        h = h / s
    return h
