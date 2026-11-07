import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
       # Thompson Sampling
       # We approximate the posterior sample for each candidate.
       # To avoid randomness affecting the "best" selection in a way that might be unstable,
       # we can use the Expected Value of the Sampled Objective?
       # No, that's mean.
       
       # Let's use a deterministic proxy for TS:
       # The "optimism" of the posterior mean.
       
       # Actually, let's try a different structural change:
       # **Minimize the Upper Bound of the Expected Value**.
       
       # Let's try **Expected Improvement (EI) but with a different cost normalization**.
       # Current: EI / cost.
       # Alternative: EI / (cost ** 2).
       # This penalizes high cost more heavily.
       # Given the minimum is at high cost, this might hurt.
       # Alternative: EI * (1 / cost). Same thing.
       # Alternative: EI - lambda * cost.
       
       # Let's try **Subtractive Cost Penalty** instead of Divisive.
       # Score = EI - lambda * cost.
       # This changes the landscape shape.
       # Divisive makes the acquisition 0 where EI is 0.
       # Subtractive makes the acquisition negative where EI is small.
       
       # But we need to handle the scale.
       # EI is in objective units. Cost is ~1-2.
       # We need a lambda.
       # "No micro-tuning constants".
       
       # How about **Expected Improvement of the Cost-Adjusted Function**?
       # Define g(x) = f(x) * cost(x)? No.
       
       # Let's look at **Inspiration 1** again.
       # It defined `acq_weighted_ei_ucb`.
       
       # I will implement a **Hybrid EI-UCB** with a dynamic weight based on `f_best`.
       # When f_best is high (early), weight UCB more (explore).
       # When f_best is low (late), weight EI more (exploit).
       
       # This is a structural change: Adaptive Hybrid.
       
       # Calculate EI
       z = (f_best - mu) / (sigma + 1e-9)
       ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
       
       # Calculate UCB (for minimization, we want low mu and high sigma)
       # UCB_score = -mu + beta * sigma
       # We need to normalize EI and UCB to have similar scales.
       # EI is roughly O(f_best - mu).
       # UCB is O(mu).
       
       # Let's normalize by the max possible value?
       # Or just use a fixed beta?
       
       # Let's try **Probability of Improvement (PI)** again.
       # It is simpler and structurally distinct.
       
       z = (f_best - mu) / (sigma + 1e-9)
       pi = norm.cdf(z)
       
       # To make it cost-aware:
       # Score = pi / cost
       
       return pi / cost
