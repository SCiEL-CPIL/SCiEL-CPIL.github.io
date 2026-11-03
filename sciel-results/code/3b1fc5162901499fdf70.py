import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time


def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
    # H_{2k}(0) = (-1)^k (2k)! / k!
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Precompute exact coefficients for H4, H8, H12
    # H4(x) = 16x^4 - 48x^2 + 12
    H4_coeff = {0: 12.0, 2: -48.0, 4: 16.0}
    # H8(x) = 256x^8 - 4608x^6 + 23040x^4 - 32256x^2 + 1680
    H8_coeff = {0: 1680.0, 2: -32256.0, 4: 23040.0, 6: -4608.0, 8: 256.0}
    # H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5806080x^6 + 13271040x^4 - 12355200x^2 + 3991680
    H12_coeff = {0: 3991680.0, 2: -12355200.0, 4: 13271040.0, 6: -5806080.0,
                 8: 1032192.0, 10: -98304.0, 12: 4096.0}

    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    def _poly_mul(A, B):
        """Multiply two polynomials represented as dict {power: coeff}."""
        result = {}
        for p1, c1 in A.items():
            for p2, c2 in B.items():
                p = p1 + p2
                result[p] = result.get(p, 0.0) + c1 * c2
        return result

    def _even_integrals(max_power):
        """Precompute ∫ x^{2k} e^{-πx²} dx for k=0..max_power//2"""
        integrals = {}
        for k in range(max_power // 2 + 1):
            n = 2 * k
            if n == 0:
                integrals[n] = 1.0  # ∫ e^{-πx²} dx = 1
            else:
                # (n-1)!! / (2π)^{n/2}
                double_fact = 1.0
                for j in range(1, n, 2):
                    double_fact *= j
                integrals[n] = double_fact / (2 * np.pi) ** (n / 2.0)
        return integrals

    # Precompute integrals up to degree 24 (max from P²)
    max_deg = 24
    E_integrals = _even_integrals(max_deg)

    def objective(params):
        a0, a4, a8 = params
        a12 = compute_a12(a0, a4, a8)

        # Build P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 as dict {power: coeff}
        P = {}
        # H0 = 1
        P[0] = P.get(0, 0.0) + a0 * 1.0
        for p, c in H4_coeff.items():
            P[p] = P.get(p, 0.0) + a4 * c
        for p, c in H8_coeff.items():
            P[p] = P.get(p, 0.0) + a8 * c
        for p, c in H12_coeff.items():
            P[p] = P.get(p, 0.0) + a12 * c

        # ||g||₂² = ∫ P(x)² e^{-πx²} dx
        P2 = _poly_mul(P, P)
        norm_sq = 0.0
        for power, coeff in P2.items():
            if power % 2 == 0 and power <= max_deg:
                norm_sq += coeff * E_integrals[power]

        # ||ĝ||₂² = ∫ |P(iξ)|² e^{-πξ²} dξ
        # P(iξ) = sum_k a_k (iξ)^k. Since only even powers:
        # P(iξ) = sum_{k even} a_k i^k ξ^k
        # |P(iξ)|² = P(iξ) * conj(P(iξ)) = P(iξ) * P(-iξ)
        # For even powers: i^k * (-i)^k = (-1)^k = 1 (since k is even)
        # So |P(iξ)|² = sum_{k,j even} a_k a_j ξ^{k+j}
        # This is just P(ξ) * P(ξ) evaluated... wait, no.
        # P(iξ) = sum_k a_k i^k ξ^k
        # P(-iξ) = sum_k a_k (-i)^k ξ^k
        # Product = sum_{k,j} a_k a_j (i^k)(-i)^j ξ^{k+j}
        # For even k,j: i^k (-i)^j = i^k * (-1)^j * i^j = (-1)^j * i^{k+j}
        # Since k,j even, k+j even, so i^{k+j} is real (±1)
        # Actually simpler: for real coefficients and even powers only,
        # P(iξ) = sum_{m} a_{2m} (iξ)^{2m} = sum_m a_{2m} (-1)^m ξ^{2m}
        # So |P(iξ)|² = [sum_m a_{2m} (-1)^m ξ^{2m}]²
        # Let Q(ξ²) = sum_m a_{2m} (-1)^m (ξ²)^m, then |P(iξ)|² = Q(ξ²)²
        # And ∫ |P(iξ)|² e^{-πξ²} dξ = ∫ Q(ξ²)² e^{-πξ²} dξ

        # Build Q(t) where t = ξ²: Q(t) = sum_m a_{2m} (-1)^m t^m
        Q = {}
        for p, c in P.items():
            if p % 2 == 0:
                m = p // 2
                sign = (-1.0) ** m
                Q[m] = Q.get(m, 0.0) + c * sign

        Q2 = _poly_mul(Q, Q)
        # ∫ Q(ξ²)² e^{-πξ²} dξ = ∫ sum_{j} Q2[j] (ξ²)^j e^{-πξ²} dξ
        # = sum_j Q2[j] ∫ ξ^{2j} e^{-πξ²} dξ
        norm_sq_hat = 0.0
        for j, coeff in Q2.items():
            power = 2 * j
            if power <= max_deg:
                norm_sq_hat += coeff * E_integrals[power]

        # The upper bound C4 is related to the product
        # For the uncertainty principle, the relevant quantity is ||g||₂² * ||ĝ||₂²
        # But we need to be careful about normalization
        # The bound on C4 is this product (lower is better)
        return norm_sq * norm_sq_hat

    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])

    best_result = None
    best_score = np.inf

    # Try multiple starting points
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
    ]

    for sp in start_points:
        if time.time() > deadline:
            break

        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()

        except Exception:
            continue

    # Final polish with tighter tolerances if time permits
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113/100, 1/25, 1/3240])

    return best_result
