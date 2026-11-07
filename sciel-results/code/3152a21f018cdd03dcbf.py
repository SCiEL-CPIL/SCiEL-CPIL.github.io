import numpy as np

def acquisition(mu, sigma, f_best):
          # Thompson Sampling Acquisition Function
          # Samples from the GP posterior and returns the negative sample
          # to maximize the score (minimize the function value).
          
          # Clip sigma to avoid zero division or invalid standard deviation
          sigma = np.maximum(sigma, 1e-12)
          
          # Draw one sample for each candidate
          # mu and sigma have shape (2000,)
          samples = np.random.normal(loc=mu, scale=sigma)
          
          # We want to minimize the function, so the "best" candidate is the one
          # with the lowest sample value.
          # The evaluator selects the candidate with the HIGHEST score.
          # Therefore, we return the negative of the samples.
          scores = -samples
          
          return scores
