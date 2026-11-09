import numpy as np
import scipy.special
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_genlaguerre, gamma

def construct():
    # Deadline guard: 630s limit, use 580s as safety margin
    start_time = time.time()
    deadline = start_time + 580.0

    def time_remaining():
        return deadline - time.time()

    try:
        m, n = 12, 1
        N_ROOTS = 12

        # The seed roots are [4, 8, ..., 48].
        seed_roots = np.array([4.0 * (i + 1) for i in range(N_ROOTS)])

        def compute_bound(roots):
            """
            Computes a proxy for the uncertainty bound C4.
            Uses the function f(x) = exp(-x^2/4) * prod(x^2 - r_i).
            Minimizes sigma_x * sigma_k.
            """
            roots = np.sort(np.asarray(roots, dtype=float))
            if np.any(roots <= 0):
                return 1e10

            # Construct the polynomial Q(y) = prod(y - r_i)
            try:
                coeffs = np.poly(roots)
            except Exception:
                return 1e10

            # 1. Position Space Moments (x^2)
            # Use Gauss-Laguerre quadrature for high precision.
            # Integral: int_0^inf x^p exp(-x^2/2) Q(x^2)^2 dx
            # Substitution: u = x^2/2 => x = sqrt(2u), dx = du/sqrt(2u)
            # x^2 = 2u
            # exp(-x^2/2) = exp(-u)
            # Q(x^2) = Q(2u)
            # Integral becomes: int_0^inf (2u)^{p/2} exp(-u) Q(2u)^2 (du/sqrt(2u))
            # = 2^{p/2 - 0.5} int_0^inf u^{p/2 - 0.5} exp(-u) Q(2u)^2 du
            # This is a Laguerre integral with alpha = p/2 - 0.5.
            
            # We need p=0 (norm) and p=2 (mean_x2).
            # For p=0: alpha = -0.5.
            # For p=2: alpha = 0.5.
            
            # Since alpha depends on p, we need separate quadrature nodes.
            # However, we can use a single high-order quadrature with alpha=0.5
            # and re-weight for p=0, or just use two.
            # Let's use two separate quadratures for accuracy.
            
            # Quadrature for Norm (p=0, alpha=-0.5)
            # roots_genlaguerre(n, alpha)
            n_quad = 100
            nodes_0, weights_0 = roots_genlaguerre(n_quad, -0.5)
            # Q(2u)
            Q_vals_0 = np.polyval(coeffs, 2.0 * nodes_0)
            # Integral = 2^{-0.5} * sum( w_i * Q(2u_i)^2 )
            norm_sq = (2.0 ** -0.5) * np.sum(weights_0 * Q_vals_0**2)
            
            if norm_sq < 1e-15:
                return 1e10

            # Quadrature for Mean x^2 (p=2, alpha=0.5)
            nodes_2, weights_2 = roots_genlaguerre(n_quad, 0.5)
            Q_vals_2 = np.polyval(coeffs, 2.0 * nodes_2)
            # Integral = 2^{0.5} * sum( w_i * (2u_i) * Q(2u_i)^2 )
            # Wait, x^2 = 2u. So integrand is 2u * exp(-u) * Q(2u)^2
            # Term is u^{0.5} * u * ... = u^{1.5}.
            # The weight w_i corresponds to int u^{0.5} e^{-u} ...
            # So we multiply by u_i.
            mean_x2_int = (2.0 ** 0.5) * np.sum(weights_2 * (2.0 * nodes_2) * Q_vals_2**2)
            mean_x2 = mean_x2_int / norm_sq
            
            sigma_x = np.sqrt(mean_x2)

            # 2. Frequency Space Moments (k^2)
            # Fourier transform of f(x) = exp(-x^2/4) Q(x^2).
            # This is tricky to do analytically for arbitrary roots.
            # Numerical FFT is the standard approach.
            # We need to be careful with the grid.
            
            # The function decays like exp(-x^2/4).
            # At x=20, exp(-100) is tiny.
            # Let's use a grid [-L, L] with L=20.
            # N points.
            
            # To get good k resolution, we need large L and N.
            # dk = 2*pi / (2*L) = pi/L.
            # We need to integrate k^2 |F(k)|^2.
            
            # Let's use N=8192 for safety and speed balance.
            N_fft = 8192
            L = 20.0
            x_fft = np.linspace(-L, L, N_fft)
            dx = x_fft[1] - x_fft[0]
            
            # f(x) is even?
            # Q(x^2) is even. exp(-x^2/4) is even. So f(x) is even.
            # We can just compute on [0, L] and mirror, or use full grid.
            # Full grid is easier for FFT.
            
            y_fft = x_fft**2
            Q_fft = np.polyval(coeffs, y_fft)
            f_fft = np.exp(-x_fft**2 / 4.0) * Q_fft
            
            # FFT
            F = np.fft.fft(f_fft)
            F = np.fft.fftshift(F)
            k = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dx))
            
            # Parseval: int |f|^2 dx = int |F|^2 dk / (2*pi)^2 ?
            # Depending on convention.
            # Let's stick to the ratio, so constants might cancel or scale consistently.
            # Actually, if we change convention, sigma_x and sigma_k scale.
            # Standard: f(x) = 1/sqrt(2pi) int F(k) e^{ikx} dk.
            # Then int |f|^2 dx = int |F|^2 dk.
            # np.fft.fft does not include 1/sqrt(2pi) or 1/N normalization in the way we want for continuous approx.
            # F_discrete[k] ~ int f(x) e^{-ikx} dx.
            # So int |f|^2 dx ~ sum |F|^2 dk / (2*pi)?
            # Let's just compute the ratio:
            # sigma_k^2 = int k^2 |F(k)|^2 dk / int |F(k)|^2 dk
            
            F_sq = np.abs(F)**2
            norm_k = np.sum(F_sq) * dx # Approximating integral dk
            # Wait, dk = 2*pi / (N*dx) ? No, dk = 2*pi / (2*L) = pi/L.
            # The FFT frequencies are in cycles per unit.
            # k_rad = 2*pi * freq.
            # Let's use k_rad for the moment.
            k_rad = 2.0 * np.pi * k
            
            # The integral over k is sum |F|^2 * dk_rad.
            dk_rad = k_rad[1] - k_rad[0]
            
            norm_k_rad = np.sum(F_sq) * dk_rad
            if norm_k_rad < 1e-15:
                return 1e10
            
            k2_F_sq = k_rad**2 * F_sq
            mean_k2_rad = np.sum(k2_F_sq) * dk_rad / norm_k_rad
            
            sigma_k = np.sqrt(mean_k2_rad)
            
            # The uncertainty principle is usually sigma_x sigma_k >= 1/2 (for hbar=1).
            # The "bound" C4 is likely this product.
            return sigma_x * sigma_k

        # Optimization Strategy:
        # 1. Start with seed.
        # 2. Try Differential Evolution for global exploration (robust).
        # 3. Refine with L-BFGS-B.
        
        best_roots = seed_roots
        best_score = compute_bound(seed_roots)

        # Differential Evolution
        # Bounds: [0.1, 100.0]
        bounds = [(0.1, 100.0)] * N_ROOTS
        
        # Use a smaller population for speed, but enough for diversity.
        # maxiter should be limited by time.
        
        try:
            # Run DE for a few seconds
            de_start = time.time()
            # maxiter=100, popsize=15 -> 1500 evaluations.
            # Each eval is ~10ms? 15s.
            # Let's try maxiter=50 first.
            res_de = differential_evolution(
                compute_bound,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False # We will polish manually
            )
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_roots = res_de.x
        except Exception:
            pass

        # Refine with L-BFGS-B
        # Try from the best root found so far (seed or DE result)
        try:
            # Ensure roots are sorted for L-BFGS-B stability?
            # L-BFGS-B doesn't require sorted, but the function sorts them internally.
            # This makes the function non-smooth at permutations.
            # To handle this, we can enforce ordering via bounds or just accept the kink.
            # The kink is usually not a problem for L-BFGS-B if we start close to a local min.
            
            res_lbfgs = minimize(
                compute_bound,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_roots = res_lbfgs.x
        except Exception:
            pass

        # One more polish with tight tolerance
        if time_remaining() > 60:
            try:
                res_polish = minimize(
                    compute_bound,
                    best_roots,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res_polish.fun < best_score:
                    best_score = res_polish.fun
                    best_roots = res_polish.x
            except Exception:
                pass

        # Finalize
        best_roots = np.sort(best_roots)
        best_roots = np.maximum(best_roots, 1e-6)
        
        # Check validity
        if len(best_roots) != N_ROOTS or np.any(~np.isfinite(best_roots)):
            return [4.0 * (i + 1) for i in range(N_ROOTS)]

        return [float(r) for r in best_roots]

    except Exception:
        return [4.0 * (i + 1) for i in range(N_ROOTS)]

# Fallback if construct fails entirely
if __name__ == "__main__":
    construct()
