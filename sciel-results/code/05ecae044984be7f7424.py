"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


_H4 = np.array([16.0, -48.0, 12.0])  # coeffs for x^4, x^2, x^0
_H8 = np.array([256.0, -1344.0, 2688.0, -1344.0, 112.0])  # x^8..x^0
_H12 = np.array([4096.0, -32256.0, 84672.0, -107520.0, 64512.0, -14336.0, 984.0])  # x^12..x^0


def _build_P(c0, c1, c2):
    """Build the degree-5 polynomial P(z) = q(sqrt(z)) in descending powers of z."""
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # H0 = 1, H4 = 16x^4-48x^2+12, H8 = 256x^8-1344x^6+2688x^4-1344x^2+112
    # H12 = 4096x^12-32256x^10+84672x^8-107520x^6+64512x^4-14336x^2+984
    # g(0) = c0 + 12*c1 + 112*c2 + 984*c3 = 0 => c3 = -(c0+12c1+112c2)/984
    c3 = -(c0 + 12.0 * c1 + 112.0 * c2) / 984.0

    # g(x) as polynomial in x^2: g(x) = sum_{k=0}^{6} a_k * x^{2k}
    # a_0 = c0 + 12*c1 + 112*c2 + 984*c3 = 0 (by construction)
    # a_1 = -48*c1 - 1344*c2 - 14336*c3
    # a_2 = 16*c1 + 2688*c2 + 64512*c3
    # a_3 = -1344*c2 - 107520*c3
    # a_4 = 256*c2 + 84672*c3
    # a_5 = -32256*c3
    # a_6 = 4096*c3
    a1 = -48.0 * c1 - 1344.0 * c2 - 14336.0 * c3
    a2 = 16.0 * c1 + 2688.0 * c2 + 64512.0 * c3
    a3 = -1344.0 * c2 - 107520.0 * c3
    a4 = 256.0 * c2 + 84672.0 * c3
    a5 = -32256.0 * c3
    a6 = 4096.0 * c3

    # q(x) = g(x)/x^2 = a1 + a2*z + a3*z^2 + a4*z^3 + a5*z^4 + a6*z^5 where z=x^2
    # P(z) = a6*z^5 + a5*z^4 + a4*z^3 + a3*z^2 + a2*z + a1
    return np.array([a6, a5, a4, a3, a2, a1])


def _largest_sign_change_root(P):
    """Find the largest real root of P where P changes sign (odd multiplicity)."""
    # Find all real roots
    roots = np.roots(P)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real

    if len(real_roots) == 0:
        return -np.inf

    # For each real root, check if P changes sign (odd multiplicity)
    # A root changes sign if the multiplicity is odd
    # We can check by evaluating P just to the left and right of the root
    valid_roots = []
    for z in real_roots:
        # Check sign change: evaluate P at z-eps and z+eps
        eps = 1e-8 * max(1.0, abs(z))
        left = np.polyval(P, z - eps)
        right = np.polyval(P, z + eps)
        if left * right < 0:
            valid_roots.append(z)

    if len(valid_roots) == 0:
        return -np.inf

    return max(valid_roots)


def _objective(params):
    c0, c1, c2 = params
    P = _build_P(c0, c1, c2)
    z_max = _largest_sign_change_root(P)
    if z_max <= 0:
        # If no positive sign-changing root, this is great (value would be negative or zero)
        # But we need g to have a negative limit at infinity handled by sign flip
        # Actually if z_max <= 0, then r^2 <= 0, value <= 0, which is ideal
        return z_max
    return z_max


def _seed_value():
    c0, c1, c2 = -113.0 / 100, 1.0 / 25, 1.0 / 3240
    P = _build_P(c0, c1, c2)
    z_max = _largest_sign_change_root(P)
    return z_max / (2.0 * np.pi)


def construct():
    deadline = time.time() + 180.0  # Safety margin before 210s kill

    # Seed point
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best_params = seed.copy()
    best_val = _objective(best_params)

    # Multi-start Nelder-Mead
    n_starts = 20
    rng = np.random.RandomState(42)

    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break

        if i == 0:
            x0 = seed.copy()
        else:
            # Perturb the seed with Gaussian noise, scaled to reasonable range
            x0 = seed + rng.normal(0, 0.5, size=3)

        try:
            res = minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                         'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue

    # High-precision polish: use BFGS or L-BFGS-B with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Final verification
    if time.time() < deadline - 5.0:
        try:
            # Try a few more local refinements around best
            for _ in range(5):
                if time.time() > deadline - 5.0:
                    break
                perturbed = best_params + rng.normal(0, 1e-4, size=3)
                try:
                    res3 = minimize(
                        _objective, perturbed, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                                 'adaptive': True}
                    )
                    if res3.fun < best_val:
                        best_val = res3.fun
                        best_params = res3.x.copy()
                except Exception:
                    continue
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
