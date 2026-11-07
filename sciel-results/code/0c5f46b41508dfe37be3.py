import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard and step tracking
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before the 330s hard limit
_call_count = 0

def _t_ei(f_best, mu, sigma, nu):
    """
    Compute Expected Improvement using a Student's t-distribution.
    
    For minimization, EI_t = E[max(f_best - X, 0)] where X ~ t_nu(mu, sigma^2)
    
    The t-distribution has heavier tails than the Gaussian, which makes the
    acquisition more robust to uncertainty in the GP posterior, especially
    with limited samples (30 evaluations total).
    
    Parameters:
    -----------
    f_best : float
        Best observed value so far.
    mu : array
        Mean of the t-distribution at each point.
    sigma : array
        Standard deviation (scale) of the t-distribution at each point.
    nu : float
        Degrees of freedom (higher = closer to Gaussian).
        
    Returns:
    --------
    ei_t : array
        Expected improvement values for each point.
    """
    # Standardize: X = mu + sigma * T where T ~ t_nu(0, 1)
    # EI = E[max(f_best - X, 0)] = E[max(f_best - mu - sigma*T, 0)]
    # Let a = (f_best - mu) / sigma
    # EI = sigma * E[max(a - T, 0)]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    a = (f_best - mu) / sigma_safe
    
    # E[max(a - T, 0)] where T ~ t_nu(0,1)
    # = a * P(T < a) + E[T * 1(T < a)]
    # P(T < a) = cdf_t(a; nu)
    # E[T * 1(T < a)] = -pdf_t(a; nu) * (1 + a^2/nu) / (nu-1) * cdf_t(a; nu) ... 
    # Actually, let's use the known formula:
    # For T ~ t_nu(0,1):
    # E[max(a - T, 0)] = a * F_t(a) + f_t(a) * (nu + a^2) / (nu - 1) * F_t(a)
    # Wait, that's not right either. Let me derive it properly.
    
    # E[max(a - T, 0)] = integral_{-inf}^{a} (a - t) f_t(t) dt
    # = a * F_t(a) - integral_{-inf}^{a} t * f_t(t) dt
    
    # For t-distribution, the first moment is 0, but the partial integral is:
    # integral_{-inf}^{a} t * f_t(t) dt = -f_t(a) * (1 + a^2/nu) / (nu-1)  [for nu > 1]
    # Actually, this is a known result for the t-distribution:
    # integral_{-inf}^{a} t * f_t(t) dt = -f_t(a) * (a^2 + nu) / (nu - 1)  [Wait, let me check]
    
    # The correct formula for the partial first moment of a t-distribution is:
    # integral_{-inf}^{a} t * f_t(t) dt = -f_t(a) * (a^2 + nu) / (nu - 1)  for nu > 1
    # Hmm, I need to be more careful. Let's use scipy.stats.t directly.
    
    from scipy.stats import t as t_dist
    
    # Compute CDF and PDF of the t-distribution at 'a'
    cdf_t = t_dist.cdf(a, df=nu)
    pdf_t = t_dist.pdf(a, df=nu)
    
    # For the t-distribution with nu degrees of freedom:
    # E[max(a - T, 0)] = a * F_t(a) + f_t(a) * (a^2 + nu) / (nu - 1) * F_t(a)
    # Wait, I think the correct formula is:
    # integral_{-inf}^{a} t * f_t(t) dt = -f_t(a) * (a^2 + nu) / (nu - 1)
    # So E[max(a - T, 0)] = a * F_t(a) + f_t(a) * (a^2 + nu) / (nu - 1)
    
    # Let me verify: when nu -> infinity, f_t -> phi (standard normal pdf) and F_t -> Phi
    # Then we get: a * Phi(a) + phi(a) * a^2 / 1 = a * Phi(a) + a^2 * phi(a)
    # But the standard EI formula is: a * Phi(a) + phi(a)
    # So there's a discrepancy. Let me re-derive.
    
    # Actually, for the standard normal:
    # integral_{-inf}^{a} t * phi(t) dt = -phi(a)
    # So E[max(a - T, 0)] = a * Phi(a) + phi(a)
    # Which matches the standard EI formula.
    
    # For the t-distribution:
    # integral_{-inf}^{a} t * f_t(t) dt = ?
    # Using integration by parts or known results:
    # The result is: -f_t(a) * (a^2 + nu) / (nu - 1)  is NOT correct for all nu.
    
    # Let me use a different approach. The expected value of max(a - T, 0) for T ~ t_nu
    # can be computed as:
    # E[max(a - T, 0)] = a * F_t(a) + (a^2 + nu) / (nu - 1) * f_t(a) * F_t(a)
    # No, that's not right either.
    
    # Let's just use numerical integration or a known closed form.
    # According to literature, for T ~ t_nu(0,1):
    # E[(a - T)^+] = a * F_t(a) + f_t(a) * (a^2 + nu) / (nu - 1)  [for nu > 1]
    
    # Wait, I think I'm confusing myself. Let me just compute it directly:
    # E[max(a - T, 0)] = integral_{-inf}^{a} (a - t) f_t(t) dt
    # = a * F_t(a) - integral_{-inf}^{a} t f_t(t) dt
    
    # The integral integral_{-inf}^{a} t f_t(t) dt for a t-distribution:
    # Using the fact that t f_t(t) = d/dt [something] ... 
    # Actually, for the t-distribution, we can use:
    # integral_{-inf}^{a} t f_t(t) dt = -f_t(a) * (a^2 + nu) / (nu - 1)  [I believe this is correct]
    
    # Let's test with nu=1 (Cauchy): f_t(a) = 1/(pi*(1+a^2))
    # Then -f_t(a) * (a^2+1)/(0) = undefined, which makes sense since Cauchy has no first moment.
    
    # For nu=2: f_t(a) = 1/(pi*(1+a^2)^{1.5}) ... wait, that's not right for nu=2.
    # f_t(a) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + a^2/nu)^{-(nu+1)/2}
    # For nu=2: f_t(a) = Gamma(1.5) / (sqrt(2*pi) * Gamma(1)) * (1 + a^2/2)^{-1.5}
    # = (sqrt(pi)/2) / (sqrt(2*pi)) * (1 + a^2/2)^{-1.5}
    # = 1/(2*sqrt(2)) * (1 + a^2/2)^{-1.5}
    
    # This is getting complicated. Let me just use the formula from the literature:
    # For T ~ t_nu(0,1), nu > 1:
    # E[(a - T)^+] = a * F_t(a) + f_t(a) * (a^2 + nu) / (nu - 1)
    
    # Actually, I found that the correct formula is:
    # integral_{-inf}^{a} t f_t(t) dt = -f_t(a) * (a^2 + nu) / (nu - 1)  is WRONG.
    
    # Let me just use a numerical approach or fall back to the Gaussian case.
    # For simplicity and reliability, let's use the following approximation:
    # For large nu, use the standard EI. For small nu, use a correction.
    
    # Actually, I'll use a different approach: simulate or use a known closed form.
    # After checking, the correct formula for the partial first moment of t is:
    # E[T * 1(T <= a)] = -f_t(a) * (a^2 + nu) / (nu - 1)  for nu > 1
    # This is actually correct! Let me verify with the normal case:
    # As nu -> infinity, f_t(a) -> phi(a), and (a^2 + nu)/(nu-1) -> 1
    # So E[T * 1(T <= a)] -> -phi(a), which is correct for the standard normal.
    
    # So: E[max(a - T, 0)] = a * F_t(a) + f_t(a) * (a^2 + nu) / (nu - 1)
    
    ei_t = sigma_safe * (a * cdf_t + pdf_t * (a**2 + nu) / (nu - 1))
    
    # Ensure finite values
    ei_t = np.where(np.isfinite(ei_t), ei_t, 0.0)
    
    return ei_t

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Student's t-based Expected Improvement (EI_t) 
    with an adaptive exploration bonus.
    
    Uses t-distribution based EI for more robust uncertainty quantification with 
    limited samples, and adds a weighted sigma term for exploration.
    The exploration weight decays over time to shift from exploration to exploitation.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement.
    """
    global _start_time, _call_count
    
    # Deadline guard: If we are running out of time, return a safe, valid score.
    if time.time() - _start_time > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-8)
        z = (f_best - mu_arr) / sigma_safe
        ei = (f_best - mu_arr) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate t-distribution based EI
    # Use nu=5 for moderate heavy-tailed behavior
    nu = 5.0
    ei_t = _t_ei(f_best, mu, sigma_safe, nu)
    
    # Adaptive exploration weight: starts high, decays over time
    step = _call_count
    exploration_weight = 0.3 - 0.25 * (step / 25.0)
    exploration_weight = max(exploration_weight, 0.05)
    
    exploration_term = exploration_weight * sigma_safe
    
    # Combine EI_t and exploration
    scores = ei_t + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    _call_count += 1
    
    return scores
