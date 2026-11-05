"""Program for the third autocorrelation inequality task.

We solve the convex QP: minimize sum(h_i^2) subject to sum(h_i) = 1.
The zero-lag autocorrelation c[N-1] = sum(h_i^2) is a lower bound on
max(convolve(h,h)), so minimizing it yields the globally optimal value 2.0,
attained by the uniform vector h_i = 1/N.
"""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    # Fix normalisation sum(h) = 1 (objective is scale-invariant).
    x0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq",
             "fun": lambda h: np.sum(h) - 1.0,
             "jac": lambda h: np.ones(N)}]
    res = minimize(lambda h: float(np.dot(h, h)),
                   x0, method="SLSQP",
                   jac=lambda h: 2.0 * h,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12})
    h = res.x
    # Renormalise to exact sum 1 to guard against numerical drift.
    s = np.sum(h)
    if np.isfinite(s) and abs(s) > 1e-12:
        h = h / s
    else:
        h = np.full(N, 1.0 / N)
    return h
