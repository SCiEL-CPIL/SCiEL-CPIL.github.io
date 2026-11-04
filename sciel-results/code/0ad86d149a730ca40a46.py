import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
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
    # ∫ x^{2k} e^{-πx²} dx = (2k-1)!! / (2^k π^k)
    # Note: The current program used (2k-1)!! / (2π)^k. 
    # Let's verify. ∫ e^{-ax^2} dx = sqrt(π/a).
    # ∫ x^2 e^{-πx^2} dx = 1/(2π) * sqrt(π/π) = 1/(2π).
    # Formula: (2k-1)!! / (2^k π^k).
    # k=1: 1!! / (2 * π) = 1/(2π). Correct.
    # k=2: 3!! / (4 * π^2) = 3/(4π^2).
    # Current program: 3!! / (2π)^2 = 3/(4π^2). Correct.
    
    E_int = np.zeros(13)
    E_int[0] = 1.0 / np.sqrt(1.0) # ∫ e^{-πx^2} dx = 1/sqrt(π/π) = 1? No.
    # ∫_{-∞}^{∞} e^{-πx^2} dx = 1.
    # So E_int[0] = 1.0 is correct.
    
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 ** k * np.pi ** k)

    # Signs for Q: Q[m] gets coefficient from P[2m] * (-1)^m
    # Fourier transform of x^{2m} e^{-πx^2} involves derivatives, which bring down powers of i.
    # Actually, it's easier to use the property that the Fourier transform of the Hermite-Gaussian
    # is a Hermite-Gaussian with phase (-i)^n.
    # However, we are working with polynomials P(x) times Gaussian.
    # Let g(x) = P(x) e^{-πx^2}.
    # \hat{g}(\xi) = \mathcal{F}[P(x) e^{-πx^2}](\xi).
    # This is equivalent to applying differential operators to the Gaussian.
    # Alternatively, we can just compute the moments of \hat{g} directly.
    # \int \xi^2 |\hat{g}(\xi)|^2 d\xi = \int (-\frac{d^2}{dx^2} g(x)) g(x) dx (up to signs/factors).
    # Actually, \int \xi^2 |\hat{g}|^2 d\xi = \int |g''(x)|^2 dx is NOT correct.
    # \int \xi^2 |\hat{g}|^2 d\xi = \langle g, -\frac{d^2}{dx^2} g \rangle.
    
    # Let's stick to the convolution method for the Fourier side, which is robust.
    # \hat{g}(\xi) = \int P(x) e^{-πx^2} e^{-2πix\xi} dx.
    # This is the Taylor expansion of the characteristic function.
    # \hat{g}(\xi) = \sum_{k} c_k \xi^{2k} ... ?
    # Actually, the current program's `Q` and `Q2` logic seems to handle the Fourier side.
    # Let's verify the `signs`.
    # \mathcal{F}[x^{2m} e^{-πx^2}](\xi) = (-i)^{2m} \frac{d^{2m}}{d\xi^{2m}} e^{-π\xi^2} ? No.
    # \mathcal{F}[x^n f(x)] = i^n \frac{d^n}{d\xi^n} \hat{f}(\xi).
    # So \mathcal{F}[x^{2m} e^{-πx^2}] = i^{2m} \frac{d^{2m}}{d\xi^{2m}} e^{-π\xi^2} = (-1)^m \frac{d^{2m}}{d\xi^{2m}} e^{-π\xi^2}.
    # The derivative of $e^{-π\xi^2}$ produces Hermite polynomials in $\xi$.
    # Specifically, $\frac{d^n}{dx^n} e^{-x^2} = (-1)^n H_n(x) e^{-x^2}$.
    # So the Fourier transform is a polynomial in $\xi$ times $e^{-π\xi^2}$.
    # The coefficients of this polynomial in $\xi$ will be related to the coefficients of P(x).
    
    # The current program uses `Q = P * signs`.
    # Let's trust the current program's `norm_sq_hat` calculation for the 0th moment.
    # For the 2nd moment $\int \xi^2 |\hat{g}|^2 d\xi$, we need to modify the integration weights.
    # $\int \xi^2 |\hat{g}(\xi)|^2 d\xi = \int \xi^2 Q(\xi)^2 e^{-2π\xi^2} d\xi$.
    # This requires moments of $\xi^2$ times the polynomial squared.
    
    # Let's compute the moments analytically for both sides.
    # Side 1: $\int x^{2k} |g(x)|^2 dx$.
    # $|g(x)|^2 = P(x)^2 e^{-2πx^2}$.
    # We need $\int x^{2k} P(x)^2 e^{-2πx^2} dx$.
    # This is $\sum_{i,j} P_i P_j \int x^{2k + 2i + 2j} e^{-2πx^2} dx$.
    # The integral is $E_{2π}[2k+2i+2j]$.
    
    # Side 2: $\int \xi^{2k} |\hat{g}(\xi)|^2 d\xi$.
    # $|\hat{g}(\xi)|^2 = \hat{P}(\xi)^2 e^{-2π\xi^2}$.
    # We need the polynomial $\hat{P}(\xi)$ first.
    # $\hat{g}(\xi) = \sum_{m} P_m \mathcal{F}[x^{2m} e^{-πx^2}](\xi)$.
    # $\mathcal{F}[x^{2m} e^{-πx^2}](\xi) = (-1)^m \frac{d^{2m}}{d\xi^{2m}} e^{-π\xi^2}$.
    # Let $H_{2m}(\sqrt{\pi}\xi) e^{-π\xi^2}$ be related to the derivative.
    # Actually, $\frac{d^n}{dx^n} e^{-\alpha x^2} = (-1)^n H_n(\sqrt{\alpha}x) e^{-\alpha x^2} / (2\sqrt{\alpha})^n$?
    # Let's use the property: $\mathcal{F}[H_n(x) e^{-πx^2}](\xi) = (-i)^n H_n(\xi) e^{-π\xi^2}$.
    # We can expand $P(x)$ in terms of Hermite polynomials $H_{2m}(x)$.
    # $P(x) = \sum c_{2m} H_{2m}(x)$.
    # Then $\hat{g}(\xi) = \sum c_{2m} (-i)^{2m} H_{2m}(\xi) e^{-π\xi^2} = \sum c_{2m} (-1)^m H_{2m}(\xi) e^{-π\xi^2}$.
    # So the coefficients for the Fourier side are just $c_{2m} (-1)^m$.
    # This matches the `signs` array in the current program!
    # $Q_m = P_m (-1)^m$.
    
    # So, we need to compute:
    # $N_0 = \int P(x)^2 e^{-2πx^2} dx$
    # $N_1 = \int x^2 P(x)^2 e^{-2πx^2} dx$
    # $M_0 = \int Q(\xi)^2 e^{-2π\xi^2} d\xi$
    # $M_1 = \int \xi^2 Q(\xi)^2 e^{-2π\xi^2} d\xi$
    
    # Precompute integrals $I_k(a) = \int x^{2k} e^{-a x^2} dx$.
    # Here a = 2π.
    a = 2.0 * np.pi
    # I_k(2π) = (2k-1)!! / (2^k (2π)^k) * sqrt(π/(2π)) = (2k-1)!! / (2^k (2π)^k) * 1/sqrt(2).
    # Wait. ∫ e^{-ax^2} dx = sqrt(π/a).
    # ∫ x^{2k} e^{-ax^2} dx = (2k-1)!! / (2^k a^k) * sqrt(π/a).
    
    def get_I(k, a):
        if k == 0:
            return np.sqrt(np.pi / a)
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        return df / (2.0 ** k * a ** k) * np.sqrt(np.pi / a)

    # Precompute I_k for k up to 24 (max power in P^2 is 24 if P is degree 12)
    # P has degree 12. P^2 has degree 24.
    # x^2 P^2 has degree 26.
    max_k = 13 # for x^2 * (x^12)^2 = x^26 -> k=13
    I_vals = np.array([get_I(k, a) for k in range(max_k + 1)])

    def compute_moments(P_coeffs, Q_coeffs):
        """
        P_coeffs: array of coefficients for x^{2m} in P(x)
        Q_coeffs: array of coefficients for x^{2m} in Q(x)
        Returns N0, N1, M0, M1
        """
        # Convolve to get coefficients of P(x)^2
        P2 = np.convolve(P_coeffs, P_coeffs)
        # P2[k] is coeff of x^{2k}
        # N0 = sum P2[k] * I_k
        # N1 = sum P2[k] * I_{k+1}
        
        N0 = 0.0
        N1 = 0.0
        for k in range(len(P2)):
            if k < len(I_vals):
                N0 += P2[k] * I_vals[k]
            if k + 1 < len(I_vals):
                N1 += P2[k] * I_vals[k + 1]
                
        Q2 = np.convolve(Q_coeffs, Q_coeffs)
        M0 = 0.0
        M1 = 0.0
        for k in range(len(Q2)):
            if k < len(I_vals):
                M0 += Q2[k] * I_vals[k]
            if k + 1 < len(I_vals):
                M1 += Q2[k] * I_vals[k + 1]
                
        return N0, N1, M0, M1

    def objective_from_coeffs(a0, a4, a8):
        """Compute the uncertainty product from three free coefficients."""
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]

        signs = np.array([(-1.0) ** m for m in range(7)])
        Q = P * signs

        N0, N1, M0, M1 = compute_moments(P, Q)
        
        if N0 <= 0 or M0 <= 0 or not np.isfinite(N0) or not np.isfinite(M0):
            return np.inf
            
        delta_x = np.sqrt(N1 / N0)
        delta_xi = np.sqrt(M1 / M0)
        
        return delta_x * delta_xi

    def objective(params):
        """Objective function for optimizer (takes array of 3 params)."""
        a0, a4, a8 = params
        return objective_from_coeffs(a0, a4, a8)

    def objective_spherical(theta, phi):
        """Spherical parameterization."""
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        return objective_from_coeffs(a0, a4, a8)

    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Convert seed to spherical
    seed_norm = np.linalg.norm(x0)
    if seed_norm > 0:
        x0_unit = x0 / seed_norm
        theta_init = np.arccos(np.clip(x0_unit[2], -1.0, 1.0))
        phi_init = np.arctan2(x0_unit[1], x0_unit[0])
        if phi_init < 0:
            phi_init += 2 * np.pi
    else:
        theta_init = np.pi / 2
        phi_init = 0.0

    best_score = objective_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init

    # Global Optimization: Differential Evolution
    if time.time() < deadline - 60:
        try:
            bounds_sph = [(1e-8, np.pi - 1e-8), (1e-8, 2 * np.pi - 1e-8)]
            de_result = differential_evolution(
                objective_spherical,
                bounds=bounds_sph,
                seed=42,
                maxiter=300,
                tol=1e-12,
                atol=1e-14,
                popsize=30,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass

    # Local Refinement: Nelder-Mead
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective_spherical,
                [best_theta, best_phi],
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_theta, best_phi = result.x
        except Exception:
            pass

    # Final Polish: Powell
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective_spherical,
                [best_theta, best_phi],
                method='Powell',
                options={'maxiter': 10000, 'xtol': 1e-16, 'ftol': 1e-19}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_theta, best_phi = result.x
        except Exception:
            pass

    # Convert back to Cartesian
    a0 = np.sin(best_theta) * np.cos(best_phi)
    a4 = np.sin(best_theta) * np.sin(best_phi)
    a8 = np.cos(best_theta)

    return np.array([a0, a4, a8])
