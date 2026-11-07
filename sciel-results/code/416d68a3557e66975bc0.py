import numpy as np
from scipy.special import erf

def _norm_cdf_vec(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def _norm_pdf_vec(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def _ei(mu, sigma, fb):
    s = np.maximum(sigma, 1e-8)
    z = (fb - mu) / s
    return (fb - mu) * _norm_cdf_vec(z) + s * _norm_pdf_vec(z)

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)
    
    scores = _ei(mu, sigma, fb)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
