"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import quad
import time

def _Om_of_a(a, Om0, w0, wa):
    if a <= 0:
        return 1.0
    a3_inv = a**(-3)
    de = (1 - Om0) * np.exp(3 * wa * (1 - a)) * a**(-3 * (1 + w0))
    m = Om0 * a3_inv
    return m / (m + de)

def _gamma_of_a(a, Om0, w0, wa, c):
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1 - Om_a
    u = 1 + w0 + wa * (1 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_approx(a, Om0, w0, wa, c):
    Om_a = _Om_of_a(a, Om0, w0, wa)
    g = _gamma_of_a(a, Om0, w0, wa, c)
    return Om_a**g if Om_a > 0 else 0.0

def _D_approx(a, Om0, w0, wa, c):
    if a >= 1:
        return 1.0
    def integrand(x):
        return _f_approx(np.exp(x), Om0, w0, wa, c)
    val, _ = quad(integrand, np.log(a), 0.0, limit=50)
    return np.exp(-val)

def construct():
    # Based on cosmological fitting literature for CPL dark energy models:
    # - c0 ~ 0.5: baseline gamma at high matter density (early times)
    # - c1 ~ 0.1: gamma increases as matter density fraction decreases (q increases)
    # - c2 ~ 0.02: small quadratic correction in q
    # - c3 ~ -0.12: more negative w0 (phantom-like) reduces gamma
    # - c4 ~ 0.06: positive wa slightly increases gamma
    # - c5 ~ 0.04: u*q interaction term
    # These coefficients capture the dependence of the growth index on
    # matter density evolution and dark energy equation of state.
    return [0.5, 0.1, 0.02, -0.12, 0.06, 0.04]
# EVOLVE-BLOCK-END
