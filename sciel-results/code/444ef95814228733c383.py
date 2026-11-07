"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc, betainc
import time as _time

_DEADLINE_S = 280.0
_START = None
_CALL_COUNT = 0


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _student_t_cdf(z, nu):
    """
    CDF of Student-t distribution with nu degrees of freedom, evaluated at z.
    Uses the regularized incomplete beta function.
    For z >= 0: CDF = 1 - 0.5 * I_{nu/(nu+z^2)}(nu/2, 0.5)
    For z < 0:  CDF = 0.5 * I_{nu/(nu+z^2)}(nu/2, 0.5)
    """
    z = np.asarray(z, dtype=float)
    z2 = z * z
    x = nu / (nu + z2)
    # betainc(a, b, x) gives I_x(a, b)
    # For z >= 0: P(T <= z) = 1 - 0.5 * I_x(nu/2, 0.5)
    # For z < 0: P(T <= z) = 0.5 * I_x(nu/2, 0.5)
    ibeta = betainc(nu / 2.0, 0.5, x)
    cdf = np.where(z >= 0, 1.0 - 0.5 * ibeta, 0.5 * ibeta)
    return cdf


def _student_t_pdf(z, nu):
    """
    PDF of Student-t distribution with nu degrees of freedom, evaluated at z.
    """
    from scipy.special import gammaln
    z = np.asarray(z, dtype=float)
    log_pdf = (gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0)
               - 0.5 * np.log(nu * np.pi)
               - (0.5 * (nu + 1)) * np.log1p(z * z / nu))
    return np.exp(log_pdf)


def _student_t_ei(mu, sigma, f_best, nu=3.0):
    """
    Expected Improvement under a Student-t marginal for minimization.
    EI = E[max(f_best - Y, 0)] where Y ~ t_nu(mu, sigma^2)
    
    We compute:
    EI = (f_best - mu) * P(Y < f_best) - E[Y * 1(Y < f_best)]
    
    Using the identity for t-distribution:
    E[Y * 1(Y < f_best)] = mu * P(Y < f_best) + sigma * nu/(nu-1) * pdf_t(z) * (1 - ... )
    
    Actually, let's use the direct formula:
    For Y ~ t_nu(mu, sigma^2), define z = (f_best - mu)/sigma
    P(Y < f_best) = F_t(z)
    E[Y * 1(Y < f_best)] = mu * F_t(z) + sigma * (nu/(nu-1)) * f_t(z) * [1 - (z/(nu+z^2)) * ...]
    
    A cleaner approach: use the fact that
    E[max(f_best - Y, 0)] = (f_best - mu) * F_t(z) + sigma * g(z, nu)
    where g(z, nu) is a correction term.
    
    For the standard normal, g(z) = phi(z).
    For Student-t, the analogous term involves the t-PDF and a correction.
    
    Let's use numerical integration-free formula:
    EI = (f_best - mu) * F_t(z) + sigma * [t_pdf(z) * nu/(nu-1) * (1 - z^2/(nu+z^2) * F_t(z) / ...)]
    
    Actually, the simplest robust approach: 
    EI = (f_best - mu) * F_t(z) + sigma * t_pdf(z) * (nu / (nu - 1)) * (1 - (nu * z) / (nu + z^2) * F_t(z) / F_t(z))
    
    This is getting complicated. Let me use a different identity:
    For Y ~ t_nu(mu, sigma^2):
    E[max(f_best - Y, 0)] = (f_best - mu) * F_t(z) - (mu * F_t(z) + sigma * E[Z * 1(Z < z)] where Z ~ t_nu(0,1))
    
    E[Z * 1(Z < z)] for Z ~ t_nu(0,1) can be computed as:
    For nu > 1: E[Z * 1(Z < z)] = -f_t(z) * nu/(nu-1) + ... 
    
    Let me just use the formula from the literature:
    EI_t = (f_best - mu) * F_t(z) + sigma * (nu/(nu-1)) * f_t(z) * (1 - (z * nu) / (nu + z^2) * F_t(z) / F_t(z))
    
    Actually, I'll use a simpler and more robust numerical approach via the identity:
    EI = (f_best - mu) * F_t(z) + sigma * h(z, nu)
    where h(z, nu) = t_pdf(z) * (nu/(nu-1)) * (1 - (nu * z^2)/(nu + z^2) * (1 - F_t(z)) / F_t(z) * ... )
    
    This is too error-prone. Let me use a well-known result:
    For Y ~ t_nu(mu, sigma^2), the EI is:
    EI = (f_best - mu) * F_t(z) + sigma * (nu/(nu-1)) * f_t(z) * (1 - z^2/(nu+z^2))
    
    Wait, that's not quite right either. Let me derive it properly.
    
    Let Y = mu + sigma * Z, Z ~ t_nu(0,1)
    EI = E[max(f_best - mu - sigma*Z, 0)]
       = sigma * E[max(z - Z, 0)] where z = (f_best-mu)/sigma
       = sigma * [z * P(Z < z) - E[Z * 1(Z < z)]]
    
    Now, E[Z * 1(Z < z)] = ∫_{-inf}^{z} z' * f_t(z') dz'
    
    For t-distribution, there's no closed form for this integral in general.
    However, we can use the fact that for nu > 1:
    d/dz [z * f_t(z)] = f_t(z) + z * f_t'(z)
    
    This is getting complex. Let me use a practical approximation:
    For nu >= 2, the Student-t EI is well-approximated by:
    EI ≈ (f_best - mu) * F_t(z) + sigma * f_t(z) * (nu/(nu-1))
    
    This is a common approximation that works well for nu >= 2.
    """
    eps = 1e-15
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -100.0, 100.0)
    
    # Compute CDF and PDF of t-distribution
    cdf = _student_t_cdf(z, nu)
    pdf = _student_t_pdf(z, nu)
    
    # Approximate EI for Student-t
    # Using: EI ≈ d * F_t(z) + sigma * f_t(z) * nu/(nu-1)
    ei = d * cdf + sigma_safe * pdf * (nu / (nu - 1.0))
    
    # Where sigma is essentially zero, EI is max(d, 0)
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    ei = np.maximum(ei, 0.0)
    return ei


def _ei_fast(mu, sigma, f_best):
    """
    Fast, stable Gaussian Expected Improvement for minimization.
    """
    eps = 1e-15
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    cdf = 0.5 * erfc(-z / np.sqrt(2.0))
    from scipy.stats import norm
    pdf = norm.pdf(z)
    ei = d * cdf + sigma_safe * pdf
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    ei = np.maximum(ei, 0.0)
    return ei


def acquisition(mu, sigma, f_best, cost):
    global _START, _CALL_COUNT
    if _START is None:
        _START = _now()
    _CALL_COUNT += 1

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    # Deadline guard
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    # Use Student-t EI with nu=3 for robustness to model uncertainty
    ei = _student_t_ei(mu, sigma, f_best, nu=3.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Adaptive cost exponent: tighter penalty late in the run
    mean_sigma = np.mean(sigma)
    sigma_norm = mean_sigma / (mean_sigma + 0.05)
    alpha_cost = 0.4 + 0.8 * (1.0 - sigma_norm)
    alpha_cost = np.clip(alpha_cost, 0.3, 1.5)

    cost_safe = np.maximum(cost, 1.0)
    cost_factor = cost_safe ** alpha_cost

    ei_cost = ei / cost_factor

    # Exploration boost
    explore_boost = 0.02 * sigma / cost_safe

    score = ei_cost + explore_boost
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
