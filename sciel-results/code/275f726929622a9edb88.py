import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
        start_time = time.time()
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        sigma_safe = np.maximum(sigma, 1e-12)
        cost_safe = np.maximum(cost, 1e-9)
        
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        # EI for minimization
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        ei = np.maximum(ei, 0.0)
        
        # PoI for minimization
        poi = cdf_z
        
        # UCB for minimization (Maximize beta*sigma - mu)
        beta = 2.0
        ucb = beta * sigma - mu
        
        # Normalize scores to [0, 1] roughly to allow fair comparison in Max
        # EI normalization: Divide by max EI or f_best
        ei_max = np.max(ei)
        if ei_max > 1e-9:
            ei_norm = ei / ei_max
        else:
            ei_norm = np.zeros_like(ei)
            
        # UCB normalization: Min-Max
        ucb_min = np.min(ucb)
        ucb_max = np.max(ucb)
        ucb_range = ucb_max - ucb_min
        if ucb_range > 1e-9:
            ucb_norm = (ucb - ucb_min) / ucb_range
        else:
            ucb_norm = np.zeros_like(ucb)
            
        # PoI is already [0, 1]
        
        # Cost adjust
        score_ei = ei_norm / cost_safe
        score_poi = poi / cost_safe
        score_ucb = ucb_norm / cost_safe
        
        # Ensemble: Max of the three
        scores = np.maximum(score_ei, np.maximum(score_poi, score_ucb))
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        if time.time() - start_time > 325:
            return np.zeros_like(mu)
            
        return scores
