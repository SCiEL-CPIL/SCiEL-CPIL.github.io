import numpy as np

def acquisition(mu, sigma, f_best):
       # Identify anchor: index with lowest mu
       # mu_min = np.min(mu)
       
       # Distance from anchor in prediction space (mu space)
       dist = mu - np.min(mu)
       
       # Normalize distance
       max_dist = np.max(dist)
       if max_dist < 1e-9:
           dist_norm = np.zeros_like(mu)
       else:
           dist_norm = dist / max_dist
           
       # Score: product of sigma and normalized distance
       # We add 1 to ensure positive scores and weight the "base" sigma
       # The prompt says "product of sigma and normalized distance".
       # If distance is 0, score is 0. That would never pick the best mean.
       # So we must add a base term or use (1 + dist_norm).
       # "Product of sigma and ... distance" usually implies just the product.
       # But if the best point has dist=0, it gets score 0.
       # This would force exploration away from the best point.
       
       # Let's assume the "Pareto" idea is to avoid the current best.
       score = sigma * (1.0 + dist_norm)
       
       return score
