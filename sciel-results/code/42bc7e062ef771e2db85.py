import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Approximate Knowledge Gradient (KG) acquisition function.
    
    This implementation uses the approximation where the minimum of the other 
    candidates is treated as their posterior mean.
    
    KG(x_i) = f_best - E[min(y_i, min_{j != i} mu_j)]
    """
    n = len(mu)
    
    # Find the minimum mean and its index
    min_idx = np.argmin(mu)
    min_mu = mu[min_idx]
    
    # We need min_{j != i} mu_j for each i.
    # If i is not the global min index, min_{j != i} mu_j is just min_mu.
    # If i IS the global min index, min_{j != i} mu_j is the second minimum.
    
    # Find the second minimum
    # Mask out the global min and find the min of the rest
    mu_masked = mu.copy()
    mu_masked[min_idx] = np.inf
    second_min_mu = np.min(mu_masked)
    
    # Initialize scores
    scores = np.zeros(n)
    
    # For all i != min_idx:
    # m_{-i} = min_mu
    # For i == min_idx:
    # m_{-i} = second_min_mu
    
    # Vectorized calculation for all i
    # m_minus_i array
    m_minus_i = np.full(n, min_mu)
    m_minus_i[min_idx] = second_min_mu
    
    # Calculate E[min(y_i, m_{-i})]
    # y_i ~ N(mu_i, sigma_i^2)
    # C = m_minus_i
    # E[min(Y, C)] = C * Phi((C-mu)/sigma) + mu * Phi((mu-C)/sigma) - sigma * phi((C-mu)/sigma)
    
    # Handle small sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (m_minus_i - mu) / sigma_safe
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # E[min(Y, C)]
    e_min = m_minus_i * cdf_z + mu * (1 - cdf_z) - sigma_safe * pdf_z
    
    # KG = f_best - E[min(Y, C)]
    scores = f_best - e_min
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
