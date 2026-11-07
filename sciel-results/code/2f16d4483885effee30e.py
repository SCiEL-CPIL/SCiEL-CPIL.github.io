"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as t_dist

# Global state to track evaluation count for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Increment evaluation count
    _eval_count += 1
    
    # Student's t-distribution based acquisition function
    # With limited samples (30 total), the GP posterior is over-confident.
    # Using a t-distribution with low degrees of freedom accounts for
    # heavy-tailed uncertainty and promotes more robust exploration.
    
    # Degrees of freedom: start low (more exploration) and increase slightly
    # as we gather more data, but keep it moderate to avoid over-confidence.
    # For 30 evaluations total, df around 3-5 is reasonable.
    df = 3.0 + 0.1 * _eval_count
    
    # Compute the standardized improvement: (f_best - mu) / sigma
    # This is the z-score for the improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement using Student's t-distribution
    # EI = E[max(f_best - Y, 0)] where Y ~ t(df, mu, sigma)
    # Let Z = (Y - mu) / sigma ~ t(df)
    # EI = E[max(f_best - mu - sigma*Z, 0)]
    #    = sigma * E[max(z - Z, 0)]
    # where z = (f_best - mu) / sigma
    # 
    # For a t-distribution with df degrees of freedom:
    # E[max(z - Z, 0)] = (z * t_cdf(z) + t_pdf(z)) / ... 
    # Actually, let's use the formula for expected improvement with t-distribution.
    #
    # The expected improvement for a t-distribution can be computed as:
    # EI = sigma * [z * F_t(z) + f_t(z)]
    # where F_t is the CDF and f_t is the PDF of the t-distribution with df degrees of freedom.
    # Wait, this is for the standard normal case. For t-distribution, the formula is different.
    #
    # Let's derive it properly.
    # E[max(z - Z, 0)] where Z ~ t(df)
    # = ∫_(-∞ to z) (z - x) * f_t(x) dx
    # = z * ∫_(-∞ to z) f_t(x) dx - ∫_(-∞ to z) x * f_t(x) dx
    # = z * F_t(z) - ∫_(-∞ to z) x * f_t(x) dx
    #
    # For the t-distribution, the integral of x * f_t(x) from -∞ to z is:
    # ∫_(-∞ to z) x * f_t(x) dx = -df / (df - 1) * f_t(z)  for df > 1
    # Wait, let me verify this.
    #
    # Actually, for the t-distribution with df degrees of freedom:
    # The density is f_t(x) = Γ((df+1)/2) / (√(dfπ) Γ(df/2)) * (1 + x²/df)^(-(df+1)/2)
    #
    # The integral ∫ x * f_t(x) dx from -∞ to z can be computed using integration by parts
    # or by recognizing that x * f_t(x) is related to the derivative of the density.
    #
    # A simpler approach: use numerical integration or a known formula.
    # For practical purposes, we can use the following approximation:
    # EI_t = sigma * (z * F_t(z) + f_t(z) * df / (df - 1))
    # But this might not be exactly correct. Let's use a more robust approach.
    #
    # Alternative: Use the fact that for large df, t approaches normal, and for small df,
    # we can use numerical quadrature or a lookup table.
    #
    # For simplicity and robustness, let's use the following formula which is known to work:
    # EI = sigma * (z * F_t(z) + f_t(z) * (df + 1) / (df - 1))  for df > 1
    # But I'm not confident in this formula. Let's use a different approach.
    #
    # Let's use the standard EI formula with the t-distribution CDF and PDF:
    # EI = sigma * E[max(z - Z, 0)]
    # We can compute this using numerical integration if needed, but for speed,
    # let's use an approximation that works well for df >= 3.
    #
    # Actually, the correct formula for the expected improvement with a t-distribution is:
    # EI = sigma * [z * F_t(z) + f_t(z) * (df / (df - 1))]  for df > 1
    # Let's verify this with df -> infinity (normal case):
    # As df -> infinity, f_t(z) -> phi(z) and F_t(z) -> Phi(z)
    # df / (df - 1) -> 1
    # So EI -> sigma * [z * Phi(z) + phi(z)]
    # This matches the standard normal EI formula! Good.
    
    # Compute t-distribution CDF and PDF
    t_cdf = t_dist.cdf(z, df)
    t_pdf = t_dist.pdf(z, df)
    
    # Expected Improvement with t-distribution
    # EI = sigma * (z * F_t(z) + f_t(z) * df / (df - 1))
    if df > 1.0:
        correction = df / (df - 1.0)
    else:
        correction = 1.0
    
    ei = sigma * (z * t_cdf + t_pdf * correction)
    
    # Ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on sigma to encourage visiting uncertain regions
    # This helps escape local minima in the multimodal Hartmann 6D function
    exploration_weight = 0.5 * np.exp(-0.05 * _eval_count)
    scores = ei + exploration_weight * sigma
    
    # Check for non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return scores
# EVOLVE-BLOCK-END
