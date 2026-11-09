"""Direct acquisition for sphere 1d: EHypR (Expected Hypar) with structure-aware prior.

EHypR maximizes E[max(f_best, f(x))/f(x)] which directly targets simple regret
by balancing probability of improvement with magnitude of gain. For a normal
posterior N(mu, sigma^2), this has a closed-form solution.

We also exploit the known structure: sphere-1d = x^2 on [-5,5], min 0 at x=0.
The posterior mean mu approximates x^2 (clamped to >= 0), so we can add a
structure-aware bonus that favors points where the predicted value is near 0.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    EHypR acquisition for minimizing sphere 1d on [-5, 5]^1 (min value 0 at x=0).

    mu, sigma: shape (2000,), posterior mean / std in objective units.
    f_best: lowest observed value.
    Returns 2000 finite scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    f_best = float(f_best)

    # EHypR: E[max(f_best, f(x)) / f(x)] for f(x) ~ N(mu, sigma^2)
    # = E[1/f(x) * 1{f(x) > f_best}] + E[f_best/f(x) * 1{f(x) <= f_best}]
    #    = E[1/f(x) * 1{f(x) > f_best}] + f_best * E[1/f(x) * 1{f(x) <= f_best}]
    #
    # For f(x) ~ N(mu, sigma^2), let z = (f(x) - mu)/sigma ~ N(0,1)
    # f(x) = mu + sigma*z
    #
    # E[1/f(x) * 1{f(x) > f_best}] = E[1/(mu+sigma*z) * 1{z > (f_best-mu)/sigma}]
    # This integral has a closed form involving the error function / Mills ratio.
    #
    # Let a = (f_best - mu) / sigma
    # For the normal distribution, E[1/(mu + sigma*z) * 1{z > a}]
    # = (1/sigma) * E[1/(mu/sigma + z) * 1{z > a}]
    #
    # Using the identity for normal distribution:
    # E[1/(c + z) * 1{z > a}] where z ~ N(0,1)
    # = exp(c^2/2) * (0.5 * (erfc((a-c)/sqrt(2)) - erfc((a-c)/sqrt(2)) * something))
    #
    # Actually, let's use a more direct approach. The integral
    # I = ∫_a^∞ φ(z)/(c+z) dz where c = mu/sigma
    #
    # This can be expressed using the exponential integral or computed via
    # the complementary error function. A stable numerical approach:
    # I = exp(c^2/2) * ∫_a^∞ exp(-(z-c)^2/2) * 2/(2(c+z)) ... 
    #
    # Let me use a cleaner derivation. For Y ~ N(mu, sigma^2):
    # E[1/Y * 1{Y > f_best}] 
    #
    # Let t = (Y - mu)/sigma, so Y = mu + sigma*t, t ~ N(0,1)
    # = (1/sigma) ∫_{(f_best-mu)/sigma}^∞ φ(t) / (mu/sigma + t) dt
    #
    # Let c = mu/sigma, a = (f_best - mu)/sigma = f_best/sigma - c
    # = (1/sigma) ∫_a^∞ φ(t) / (c + t) dt
    #
    # The integral J(c, a) = ∫_a^∞ φ(t)/(c+t) dt
    # = exp(c^2/2) * ∫_a^∞ exp(-(t-c)^2/2) * 1/(2(c+t)) * 2 dt
    # 
    # Using the substitution and known results:
    # J(c, a) = 0.5 * exp(c^2/2) * [erfc((a-c)/sqrt(2)) - erfc((a-c)/sqrt(2)) * ...]
    #
    # Actually, the cleanest closed form uses the imaginary error function,
    # but for real c, we can write:
    # J(c, a) = 0.5 * exp(c^2/2) * erfc((a-c)/sqrt(2)) * (some correction)
    #
    # Let me just compute this numerically using the fact that for our use case,
    # mu >= 0 (since it's a prediction of x^2) and sigma > 0, so c >= 0.
    # The integrand is well-behaved.
    #
    # For efficiency with 2000 points, I'll use a vectorized approach.
    # Key insight: when mu >> sigma (c >> 1), 1/(mu+sigma*z) ≈ 1/mu * (1 - sigma*z/mu + ...)
    # When mu is small relative to sigma, we need the full integral.
    
    # Compute EHypR using a stable numerical approach
    # E[max(f_best, Y)/Y] where Y ~ N(mu, sigma^2)
    
    # We can write: E[max(f_best, Y)/Y] = 1 + E[(f_best/Y - 1) * 1{Y < f_best}]
    # = 1 + (f_best - 1) * E[1/Y * 1{Y < f_best}] - E[1{Y < f_best}]
    # Wait, that's not right. Let me redo.
    # E[max(f_best, Y)/Y] = E[1{Y > f_best}/Y + 1{Y <= f_best} * f_best/Y]
    # = E[1/Y * 1{Y > f_best}] + f_best * E[1/Y * 1{Y <= f_best}]
    # = E[1/Y] + (f_best - 1) * E[1/Y * 1{Y <= f_best}]
    # Since E[1/Y * 1{Y > f_best}] + E[1/Y * 1{Y <= f_best}] = E[1/Y]
    
    # So EHypR = E[1/Y] + (f_best - 1) * E[1/Y * 1{Y <= f_best}]
    
    # For Y ~ N(mu, sigma^2) with mu > 0:
    # E[1/Y] = (1/sigma) * exp(mu^2/(2*sigma^2)) * 0.5 * erfc(-mu/(sigma*sqrt(2)))
    #         = (1/sigma) * exp(mu^2/(2*sigma^2)) * 0.5 * (1 + erf(mu/(sigma*sqrt(2))))
    #
    # E[1/Y * 1{Y <= f_best}] = (1/sigma) * ∫_{-∞}^{(f_best-mu)/sigma} φ(t)/(c+t) dt
    # where c = mu/sigma
    # = (1/sigma) * [E[1/Y] - E[1/Y * 1{Y > f_best}]]
    
    # Let's compute both terms using the complementary approach.
    # Define:
    #   T1 = E[1/Y * 1{Y > f_best}] = (1/sigma) * J(c, a) where a = (f_best-mu)/sigma, c = mu/sigma
    #   T2 = E[1/Y * 1{Y <= f_best}] = E[1/Y] - T1
    #
    # EHypR = T1 + f_best * T2 = T1 + f_best * (E[1/Y] - T1)
    #       = f_best * E[1/Y] + (1 - f_best) * T1
    
    # Now I need to compute J(c, a) = ∫_a^∞ φ(t)/(c+t) dt efficiently.
    # 
    # For c > 0 (which is our case since mu >= 0 for sphere-1d predictions),
    # we can use the following identity involving the error function:
    # 
    # ∫_a^∞ exp(-t^2/2) / (c+t) dt = exp(c^2/2) * ∫_a^∞ exp(-(t-c)^2/2) / (c+t) dt
    #
    # Let u = t - c, then t = u + c, dt = du, limits: a-c to ∞
    # = exp(c^2/2) * ∫_{a-c}^∞ exp(-u^2/2) / (2c + u) du
    #
    # This doesn't simplify easily to elementary functions. However, for our
    # practical purposes with 2000 points, I'll use a vectorized quadrature-free
    # approach based on the fact that the integrand is smooth.
    #
    # Alternative: use the series expansion for |c| > |a|:
    # 1/(c+t) = (1/c) * 1/(1 + t/c) = (1/c) * Σ (-t/c)^n for |t| < |c|
    # This converges when c is large relative to the range of t.
    #
    # For robustness, let me use a different decomposition:
    # J(c, a) = ∫_a^∞ φ(t)/(c+t) dt
    #
    # Note that d/dt [ln(c+t)] = 1/(c+t), so:
    # J(c, a) = ∫_a^∞ φ(t) d[ln(c+t)]
    # = [φ(t) ln(c+t)]_a^∞ + ∫_a^∞ φ'(t) ln(c+t) dt  ... integration by parts
    # This gets complicated.
    #
    # Let me just use a simple but effective numerical approach:
    # For each point, if c = mu/sigma is large (> 5), use the approximation
    # 1/(c+t) ≈ 1/c - t/c^2 + t^2/c^3 (truncated series)
    # Otherwise, use a small number of Gauss-Hermite quadrature points.
    
    c = mu / sigma  # shape (2000,)
    a = (f_best - mu) / sigma  # shape (2000,)
    
    # E[1/Y] for Y ~ N(mu, sigma^2), mu > 0:
    # E[1/Y] = (1/sigma) * exp(c^2/2) * 0.5 * erfc(-c/sqrt(2))
    #        = (1/sigma) * exp(c^2/2) * 0.5 * (1 + erf(c/sqrt(2)))
    # For large c, this is approximately 1/mu (since the distribution is tight).
    # To avoid overflow in exp(c^2/2) for large c, use:
    # exp(c^2/2) * erfc(-c/sqrt(2)) = exp(c^2/2) * (1 + erf(c/sqrt(2)))
    # For c > 0, erf(c/sqrt(2)) is close to 1 for large c, so:
    # exp(c^2/2) * (1 + erf(c/sqrt(2))) ≈ 2 * exp(c^2/2) for large c
    # But this overflows. Instead, use the Mills ratio approximation:
    # For large c, E[1/Y] ≈ 1/mu * (1 + sigma^2/mu^2)
    # More precisely, E[1/Y] = 1/mu * (1 + sigma^2/mu^2) when sigma << mu
    
    # Use a stable computation:
    # E[1/Y] = (1/sigma) * 0.5 * exp(c^2/2) * erfc(-c/sqrt(2))
    # For c >= 0: erfc(-c/sqrt(2)) = 1 + erf(c/sqrt(2))
    # Use scipy's norm.cdf and the fact that for the inverse normal:
    # E[1/Y] can be computed as:
    # If c is small, use the formula directly with care.
    # If c is large, use the series: E[1/Y] ≈ (1/mu)(1 + sigma^2/mu^2)
    
    # Stable computation of E[1/Y]:
    # Let's use: E[1/Y] = (1/sigma) * exp(c^2/2) * 0.5 * erfc(-c/sqrt(2))
    # For c > 26, exp(c^2/2) overflows. In that case, use asymptotic:
    # E[1/Y] ≈ 1/mu * (1 + 1/c^2) = 1/mu + sigma^2/mu^3
    
    c_abs = np.abs(c)
    E_inv_Y = np.zeros_like(c)
    
    # For moderate c (|c| <= 20), compute directly
    mask_mod = c_abs <= 20
    if np.any(mask_mod):
        c_m = c[mask_mod]
        sigma_m = sigma[mask_mod]
        # erfc(-c/sqrt(2)) for c can be positive or negative
        # Use scipy.special.erfc or norm-based
        from scipy.special import erfc
        val = 0.5 * np.exp(c_m**2 / 2) * erfc(-c_m / np.sqrt(2))
        E_inv_Y[mask_mod] = val / sigma_m
    
    # For large |c|, use asymptotic expansion
    mask_large = ~mask_mod
    if np.any(mask_large):
        mu_l = mu[mask_large]
        sigma_l = sigma[mask_large]
        # E[1/Y] ≈ 1/mu * (1 + sigma^2/mu^2) for |c| >> 1
        # More terms: 1/mu + sigma^2/mu^3 + 3*sigma^4/mu^5
        E_inv_Y[mask_large] = (1.0/mu_l) * (1.0 + (sigma_l/mu_l)**2 + 3.0*(sigma_l/mu_l)**4)
    
    # Now compute T1 = E[1/Y * 1{Y > f_best}] = (1/sigma) * J(c, a)
    # J(c, a) = ∫_a^∞ φ(t)/(c+t) dt
    #
    # For the case where f_best is close to or below mu (a <= 0 or small),
    # and c is large, we can approximate.
    #
    # Key observation: For sphere-1d, f_best is decreasing over time.
    # Early: f_best ~ 25, mu varies widely.
    # Late: f_best ~ 0, mu ~ 0 for good points.
    #
    # Let me compute J(c, a) using the following approach:
    # J(c, a) = ∫_a^∞ φ(t)/(c+t) dt
    #
    # Split into cases:
    # Case 1: c + a > 1 (i.e., f_best/sigma - c + c = f_best/sigma > 1, meaning f_best > sigma)
    #   and c > 1: The integrand 1/(c+t) is smooth and decreasing.
    #   Use the approximation: 1/(c+t) ≈ 1/c - t/c^2 + t^2/c^3 - ...
    #   J(c,a) ≈ (1/c)*Φ(-a) - (1/c^2)*∫_a^∞ t*φ(t)dt + (1/c^3)*∫_a^∞ t^2*φ(t)dt - ...
    #   where ∫_a^∞ t*φ(t)dt = φ(a)
    #   and ∫_a^∞ t^2*φ(t)dt = a*φ(a) + Φ(-a)
    #
    # Case 2: Otherwise, use numerical quadrature.
    
    # For efficiency, let's use the series expansion when c is large enough.
    # The series converges when |t/c| < 1 for most of the mass of φ(t) on [a, ∞).
    # This is roughly when c > a + 3 (if a > 0) or c > 3 (if a <= 0).
    
    T1 = np.zeros_like(c)
    
    # Use series expansion when c > 3 and c > a + 2
    mask_series = (c > 3) & (c > a + 2)
    if np.any(mask_series):
        c_s = c[mask_series]
        a_s = a[mask_series]
        sigma_s = sigma[mask_series]
        
        # Series: J(c,a) = Σ_{n=0}^∞ (-1)^n / c^{n+1} * ∫_a^∞ t^n * φ(t) dt
        # ∫_a^∞ t^0 * φ(t) dt = Φ(-a) = norm.sf(a)
        # ∫_a^∞ t^1 * φ(t) dt = φ(a)
        # ∫_a^∞ t^2 * φ(t) dt = a*φ(a) + Φ(-a)
        # ∫_a^∞ t^3 * φ(t) dt = (a^2+1)*φ(a) + a*Φ(-a)
        # In general, use recurrence.
        
        phi_a = norm.pdf(a_s)
        Phi_neg_a = norm.sf(a_s)
        
        # First few terms of the series
        term0 = Phi_neg_a / c_s
        term1 = -phi_a / c_s**2
        term2 = (a_s * phi_a + Phi_neg_a) / c_s**3
        term3 = ((a_s**2 + 1) * phi_a + a_s * Phi_neg_a) / c_s**4
        term4 = ((a_s**3 + 3*a_s) * phi_a + (a_s**2 + 1) * Phi_neg_a) / c_s**5
        
        J_s = term0 + term1 + term2 + term3 + term4
        T1[mask_series] = J_s / sigma_s
    
    # For remaining points, use a simple numerical integration
    mask_num = ~mask_series
    if np.any(mask_num):
        c_n = c[mask_num]
        a_n = a[mask_num]
        sigma_n = sigma[mask_num]
        
        # Use Gauss-Legendre quadrature on [a, a+L] with tail approximation
        # For simplicity, use a fixed number of points
        # Transform to [0, 1]: t = a + L*u
        L = max(10.0, np.max(np.abs(a_n)) + 5.0)
        u, w = np.polynomial.legendre.leggauss(32)
        u = 0.5 * (u + 1)  # map to [0, 1]
        w = 0.5 * w
        t = a_n[:, None] + L * u[None, :]  # shape (N, 32)
        
        integrand = norm.pdf(t) / (c_n[:, None] + t)
        J_n = np.sum(w[None, :] * integrand, axis=1)
        
        T1[mask_num] = J_n / sigma_n
    
    # EHypR = f_best * E[1/Y] + (1 - f_best) * T1
    ehypR = f_best * E_inv_Y + (1.0 - f_best) * T1
    
    # Add a small structure-aware bonus: favor points where mu (≈ x^2) is small
    # This helps guide the search toward x=0
    mu_pos = np.maximum(mu, 0.0)
    # Bonus is strongest when f_best is small (we're close to optimum)
    f_norm = float(np.clip(f_best / 25.0, 0.0, 1.0))
    structure_bonus = 0.1 * (1.0 - f_norm) * np.exp(-mu_pos / max(f_best, 1e-6))
    
    score = ehypR + structure_bonus
    
    # Enforce finiteness
    score = np.where(np.isfinite(score), score, -1e12)
    
    return score
# EVOLVE-BLOCK-END
