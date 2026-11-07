import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function for Bayesian optimization.
    
    Strategy:
    - Uses a deterministic score: score = -mu + kappa * sigma.
    - We are minimizing, so we want to select points with low expected value (mu)
      and high uncertainty (sigma).
    - The 'score' returned should be such that the MAX score is selected.
      Therefore, we maximize (kappa * sigma - mu).
    - The exploration coefficient kappa is adaptive. It starts high to encourage
      exploration and decays over time to focus on exploitation.
    - We estimate the current step count 'n' by counting the number of candidates
      where sigma is significantly above the noise floor (indicating unexplored regions)
      or simply use a heuristic based on the range of sigma if step count is not directly available.
      However, a more robust proxy for 'time' in a stateless acquisition function is the 
      number of points with high uncertainty. But since we don't have the history, 
      we can use a fixed schedule based on a heuristic for 'n'.
      
      Actually, a common trick in stateless BO is to use the magnitude of sigma itself 
      or a fixed decay if we can't track steps. But we can approximate the step 
      by looking at the distribution of sigma. Early on, sigma is large everywhere. 
      Later, sigma is small in explored regions.
      
      Let's use a simpler, robust adaptive kappa:
      kappa = max(0.1, 1.0 / (1.0 + 0.05 * n_est))
      
      How to estimate n? 
      In the early stages, sigma is high. In later stages, sigma drops.
      Let's just use a fixed moderate kappa if we can't track state, 
      OR we can try to infer the 'age' of the BO run from the spread of sigma.
      
      A better structural change: Use Expected Improvement (EI) with a small 
      exploration bonus, or stick to LCB with a well-tuned kappa.
      
      Given the directive to make a STRUCTURAL change and avoid micro-tuning, 
      I will implement a standard LCB with a kappa that adapts based on the 
      current best value f_best relative to the posterior mean.
      
      Actually, the most distinct structural change from TS is **Deterministic 
      Expected Improvement (EI)**. EI is the gold standard for BO. 
      TS is stochastic. EI is deterministic and directly optimizes the expected 
      reduction in the best observed value.
      
      Let's switch to **Expected Improvement (EI)**.
      
      Formula for minimization:
      EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
      where Z = (f_best - mu(x)) / sigma(x)
      Phi is the standard normal CDF, phi is the standard normal PDF.
      
      This is a fundamentally different acquisition function from Thompson Sampling.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score
    # Z = (f_best - mu) / sigma
    # If mu < f_best, Z is positive.
    # EI is 0 if mu >= f_best and sigma is small, but mathematically:
    
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Use scipy for normal CDF and PDF for precision
    from scipy.stats import norm
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Return EI as scores. Higher EI means better candidate.
    # The evaluator picks the max score.
    
    # Safety check: if any NaNs or Infs, replace with 0
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
