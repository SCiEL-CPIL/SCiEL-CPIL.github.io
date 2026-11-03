import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint P(0)=0
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Coefficients for H0, H4, H8, H12 as arrays indexed by power//2
    # H0(x) = 1
    # H4(x) = 12 - 48x^2 + 16x^4
    # H8(x) = 1680 - 32256x^2 + 23040x^4 - 4608x^6 + 256x^8
    # H12(x) = 3991680 - 12355200x^2 + 13271040x^4 - 5806080x^6 + 1032192x^8 - 98304x^10 + 4096x^12
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12
    # ∫_{-∞}^{∞} x^{2k} e^{-πx²} dx = (2k-1)!! / (2π)^k
    # Note: The Gaussian is e^{-πx²}, so the normalization is different from e^{-x²/2}.
    # Let's verify the integral formula.
    # ∫ e^{-πx²} dx = 1.
    # ∫ x² e^{-πx²} dx = 1/(2π).
    # Formula: (2k-1)!! / (2π)^k.
    # k=0: 1 / 1 = 1. Correct.
    # k=1: 1 / (2π) = 1/(2π). Correct.
    # k=2: 3 / (2π)^2 = 3/(4π²).
    # Check: ∫ x⁴ e^{-πx²} dx. Using ∫ x^{2n} e^{-ax²} = (2n-1)!!/ (2a)^n √(π/a).
    # Here a=π. √(π/π)=1. So (2n-1)!! / (2π)^n.
    # Yes, the formula is correct.

    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Q: Q[m] gets coefficient from P[2m] * (-1)^m
    # Fourier transform of x^{2m} e^{-πx²} is (i)^{2m} (2m-1)!! / (2π)^m e^{-πξ²} = (-1)^m ...
    # Actually, let's derive Q coefficients carefully.
    # g(x) = P(x) e^{-πx²} = (∑ P_m x^{2m}) e^{-πx²}
    # \hat{g}(ξ) = ∫ P(x) e^{-πx²} e^{-2πixξ} dx
    # Using the property that FT of x^n f(x) is i^n (d^n/dξ^n) \hat{f}(ξ) is not quite right for the coefficient extraction.
    # Let's use the specific property for Gaussian moments.
    # ∫ x^{2m} e^{-π(x-ξ)^2} ... no, standard FT definition.
    # Let h(x) = x^{2m} e^{-πx²}.
    # \hat{h}(ξ) = ∫ x^{2m} e^{-πx²} e^{-2πixξ} dx.
    # This is related to the derivative of the Gaussian FT.
    # Alternatively, use the orthogonality of Hermite polynomials.
    # The problem statement implies a specific structure where the Fourier transform coefficients are related by signs.
    # The seed program uses Q = P * signs where signs[m] = (-1)^m.
    # Let's verify this.
    # If P(x) = H_n(x), \hat{H_n}(ξ) = (-1)^n H_n(ξ) (for probabilists' Hermite? No, physicists' with e^{-x²/2}?).
    # For e^{-πx²}, the Hermite polynomials are orthogonal.
    # The specific relation in the seed code is:
    # norm_sq_hat = ||Q||_2^2 where Q[m] = P[m] (-1)^m.
    # This implies that the "Fourier coefficients" in the Hermite basis are just the original coefficients multiplied by (-1)^m.
    # This is consistent with the property that the Fourier transform of the m-th Hermite polynomial (scaled appropriately) is (-1)^m times itself.
    # So we trust the seed's formulation: Q = P * [1, -1, 1, -1, 1, -1, 1].

    signs = np.array([(-1.0) ** m for m in range(7)])

    def get_P(params):
        a0, a4, a8 = params
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        P = np.zeros(7)
        # H0 contributes to P[0]
        P[0] += a0 * H0_arr[0]
        # H4 contributes to P[0], P[1], P[2]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        # H8 contributes to P[0]..P[4]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        # H12 contributes to P[0]..P[6]
        for m in range(7):
            P[m] += a12 * H12_arr[m]
        return P

    def objective_product(params):
        P = get_P(params)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        
        if norm_sq <= 0 or norm_sq_hat <= 0:
            return 1e30
        return norm_sq * norm_sq_hat

    def objective_log(params):
        P = get_P(params)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        
        if norm_sq <= 0 or norm_sq_hat <= 0:
            return 1e30
        return np.log(norm_sq) + np.log(norm_sq_hat)

    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = None
    best_score = np.inf

    # 1. Global Search using Differential Evolution
    # Define bounds. The coefficients can vary, but the optimal values are likely small.
    # Based on x0, a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003.
    # Let's try a wider range to ensure we don't miss a better basin.
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                objective_product, 
                bounds, 
                seed=42, 
                maxiter=1000, 
                tol=1e-10,
                polish=True,  # Use L-BFGS-B for polish
                update_interval=50,
                workers=1
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception as e:
            print(f"DE failed: {e}")

    # 2. Local Search from multiple starting points using Nelder-Mead
    start_points = [
        x0,
        best_result if best_result is not None else x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
        np.array([-1.1, 0.08, 0.003]),
        np.array([-0.9, 0.06, 0.004]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        if sp is None:
            continue
        try:
            result = minimize(
                objective_product,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue

    # 3. High-Precision Polish with BFGS on the log-objective
    # The log-objective is smoother and better conditioned for BFGS.
    if best_result is not None and time.time() < deadline - 10:
        try:
            # Use the best result from previous steps to start the log-optimization
            # We need to ensure we are in the same basin.
            result = minimize(
                objective_log,
                best_result,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-14, 'disp': False}
            )
            # Convert log-score back to product-score for comparison
            # score = exp(log_score)
            # But we can just compare log_scores if we track them, or exp them.
            # Let's just evaluate the product at the new point.
            new_score = objective_product(result.x)
            if new_score < best_score:
                best_score = new_score
                best_result = result.x.copy()
        except Exception as e:
            print(f"BFGS failed: {e}")

    # 4. Final ultra-precise Nelder-Mead polish on the product objective
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective_product,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
