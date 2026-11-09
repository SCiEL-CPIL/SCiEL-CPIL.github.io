import numpy as np

def acquisition(mu, sigma, f_best, cost):
          sigma_safe = np.maximum(sigma, 1e-8)
          
          # UCB for minimization:
          # We want to minimize f(x).
          # The lower confidence bound is mu - kappa * sigma.
          # We want to maximize the potential for being low.
          # So we score by -mu + kappa * sigma.
          kappa = 1.5
          
          ucb_score = -mu + kappa * sigma_safe
          
          # Cost awareness
          score = ucb_score / cost
          
          # Ensure finite
          score = np.where(np.isfinite(score), score, 0.0)
          
          return score
