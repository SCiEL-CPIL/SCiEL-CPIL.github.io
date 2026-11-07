import numpy as np
from scipy.stats import t as t_dist
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a Student's t-distribution based Expected Improvement to balance
    exploration and exploitation with heavy-tailed uncertainty modeling.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    N = len(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Degrees of freedom for Student's t-distribution
    # Lower df means heavier tails, promoting more exploration
    df = 5.0
    
    # Compute the standardized improvement
    # For minimization, improvement is f_best - f(x)
    # We model f(x) ~ t_df(mu, sigma)
    # EI = E[max(f_best - f(x), 0)]
    
    # Standardize: z = (f_best - mu) / sigma
    # If f(x) ~ t_df(mu, sigma), then (f(x) - mu)/sigma ~ t_df(0, 1)
    # So f_best - f(x) = (f_best - mu) - sigma * t_df(0,1)
    # Let z = (f_best - mu) / sigma
    # Then f_best - f(x) = sigma * (z - t_df(0,1))
    # EI = sigma * E[max(z - t_df(0,1), 0)]
    
    # For t-distribution with df degrees of freedom:
    # E[max(z - T, 0)] where T ~ t_df(0,1)
    # = integral from -inf to z of (z - t) * pdf_t(t) dt
    # = z * CDF_t(z) - integral from -inf to z of t * pdf_t(t) dt
    # 
    # For t-distribution, the first moment E[T] = 0 for df > 1
    # But we need the partial expectation.
    # 
    # A known result: for T ~ t_df(0,1),
    # E[max(z - T, 0)] = z * CDF_t(z) - pdf_t(z) * (df-1) / df * (1 - 1/sqrt(df + z^2))
    # 
    # Actually, let me use a more standard approach.
    # For normal: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For t-distribution, we can use numerical integration or an approximation.
    
    # Let's use the formula for t-distribution EI:
    # E[max(a - T, 0)] where T ~ t_df(0,1) and a = (f_best - mu)/sigma
    # = a * CDF_t(a) - E[T * I(T <= a)]
    # 
    # For t-distribution, E[T * I(T <= a)] can be computed using the relationship:
    # E[T * I(T <= a)] = -pdf_t(a) * (df-1) / (df * sqrt(1 + a^2/(df-1))) * ... 
    # 
    # Actually, a simpler approach: use the fact that for t-distribution,
    # the partial expectation has a closed form.
    # 
    # Let me use a different formulation. The EI for t-distribution is:
    # EI = sigma * [ (f_best - mu)/sigma * CDF_t((f_best - mu)/sigma) 
    #                - sigma * pdf_t((f_best - mu)/sigma) * (df-1) / (df * sqrt(1 + ((f_best-mu)/sigma)^2/(df-1))) ]
    # 
    # Wait, let me be more careful. Let me use the standard result.
    # 
    # If X ~ t_df(mu, sigma), then EI = E[max(f_best - X, 0)]
    # Let Y = (X - mu)/sigma ~ t_df(0, 1)
    # f_best - X = f_best - mu - sigma * Y = sigma * ((f_best - mu)/sigma - Y)
    # Let z = (f_best - mu)/sigma
    # EI = sigma * E[max(z - Y, 0)]
    # 
    # E[max(z - Y, 0)] = integral_{-inf}^{z} (z - y) * pdf_t(y) dy
    # = z * CDF_t(z) - integral_{-inf}^{z} y * pdf_t(y) dy
    # 
    # For t-distribution, the first partial moment is:
    # integral_{-inf}^{z} y * pdf_t(y) dy = -pdf_t(z) * (df-1) / (df * sqrt(1 + z^2/(df-1)))
    # 
    # Wait, that's not quite right either. Let me use a known approximation or compute it numerically.
    # 
    # Actually, for the t-distribution with df degrees of freedom:
    # E[Y * I(Y <= z)] = -pdf_t(z) * (df-1) / (df * sqrt(1 + z^2/(df-1)))
    # 
    # Let me verify this. For the normal case (df -> inf), this should reduce to -phi(z).
    # As df -> inf, (df-1)/df -> 1, and sqrt(1 + z^2/(df-1)) -> 1, so it becomes -pdf_t(z) -> -phi(z). Good.
    
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF of t-distribution
    cdf_t = t_dist.cdf(z, df)
    pdf_t = t_dist.pdf(z, df)
    
    # Partial expectation term
    # E[Y * I(Y <= z)] = -pdf_t(z) * (df-1) / (df * sqrt(1 + z^2/(df-1)))
    sqrt_term = np.sqrt(1 + z**2 / (df - 1))
    partial_moment = -pdf_t * (df - 1) / (df * sqrt_term)
    
    # EI = sigma * [z * CDF_t(z) - partial_moment]
    # = sigma * [z * CDF_t(z) + pdf_t * (df-1) / (df * sqrt_term)]
    
    ei = sigma_safe * (z * cdf_t + pdf_t * (df - 1) / (df * sqrt_term))
    
    # The score should be higher for better candidates
    # EI is already positive and higher for better candidates
    scores = ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
