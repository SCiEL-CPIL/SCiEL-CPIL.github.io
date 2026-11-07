"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Student-t Expected Improvement (EIST)
    # The posterior predictive is approximated by a Student-t distribution.
    # nu is the degrees of freedom. Lower nu = heavier tails = more exploration.
    # For a standard GP with N data points, nu is often N or N+1.
    # Since we don't have N explicitly, we use a fixed nu that balances
    # exploitation and exploration. nu=1 is Cauchy (very heavy), nu=5 is moderate.
    # Let's try nu=4 for a robust but not overly aggressive exploration.
    nu = 4.0
    
    # z-score for the improvement
    delta = f_best - mu
    z = delta / sigma_safe
    
    # The CDF and PDF of the Student-t distribution
    # We need E[max(0, f_best - Y)] where Y ~ t_nu(mu, sigma)
    # Let Z ~ t_nu(0, 1). Then Y = mu + sigma * Z.
    # Improvement I = max(0, f_best - (mu + sigma*Z)) = max(0, delta - sigma*Z)
    # E[I] = delta * P(Z < delta/sigma) + sigma * E[Z * 1(Z < delta/sigma)]
    
    # P(Z < z) is the CDF of t_nu
    cdf_t = student_t.cdf(z, nu)
    
    # E[Z * 1(Z < z)] for Z ~ t_nu(0,1)
    # This is related to the PDF. Specifically, for standard normal, E[Z 1(Z<z)] = -phi(z).
    # For Student-t, E[Z 1(Z<z)] = -nu * t_pdf(z, nu) / (nu - 1) ? 
    # Let's verify the identity.
    # f(z) = C * (1 + z^2/nu)^(-(nu+1)/2)
    # Integral of z * f(z) from -inf to z.
    # Note that d/dz [ (1 + z^2/nu)^(-(nu-1)/2) ] = -(nu-1)/2 * 2z/nu * (1 + z^2/nu)^(-(nu+1)/2)
    # = -(nu-1)/nu * z * (1 + z^2/nu)^(-(nu+1)/2)
    # So z * (1 + z^2/nu)^(-(nu+1)/2) = -nu/(nu-1) * d/dz [ (1 + z^2/nu)^(-(nu-1)/2) ]
    # The integral is -nu/(nu-1) * C * [ (1 + z^2/nu)^(-(nu-1)/2) ]_{-inf}^z
    # At -inf, term is 0. At z, term is (1 + z^2/nu)^(-(nu-1)/2).
    # So Integral = -nu/(nu-1) * C * (1 + z^2/nu)^(-(nu-1)/2)
    # We know t_pdf(z) = C * (1 + z^2/nu)^(-(nu+1)/2) = C * (1 + z^2/nu)^(-1/2) * (1 + z^2/nu)^(-(nu-1)/2)
    # So C * (1 + z^2/nu)^(-(nu-1)/2) = t_pdf(z) / (1 + z^2/nu)^(-1/2) = t_pdf(z) * sqrt(1 + z^2/nu)
    # Thus E[Z 1(Z<z)] = -nu/(nu-1) * t_pdf(z) * sqrt(1 + z^2/nu)
    
    pdf_t = student_t.pdf(z, nu)
    sqrt_term = np.sqrt(1.0 + z**2 / nu)
    exp_z_ind = - (nu / (nu - 1.0)) * pdf_t * sqrt_term
    
    # EIST = delta * cdf_t + sigma * exp_z_ind
    eist = delta * cdf_t + sigma_safe * exp_z_ind
    
    # Cost-aware score
    # We divide by cost to normalize the expected improvement per unit cost.
    # We also add a small exploration bonus to ensure we don't completely ignore
    # high-uncertainty regions if EI is zero (e.g., when mu > f_best everywhere).
    # However, EIST is always positive if sigma > 0.
    
    # To handle the case where sigma is very small (exploitation), the division by cost
    # might penalize cheap points too much if they have low EI.
    # Let's stick to EIST / cost.
    
    score = eist / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
