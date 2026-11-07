import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Structural change: Pure Lower Confidence Bound (LCB) for minimization
    # The score is defined as -mu + beta * sigma.
    # Maximizing this score balances finding low mean values (exploitation)
    # with high posterior uncertainty (exploration).
    # 
    # Rationale:
    # 1. The Weierstrass function is highly multimodal.
    # 2. The evaluation budget is small (30 total).
    # 3. Expected Improvement (EI) decays as f_best improves, potentially
    #    leading to premature convergence to local minima.
    # 4. LCB maintains a linear incentive for exploration (via sigma)
    #    independent of f_best magnitude, promoting better global coverage.
    
    # Beta is set to 2.5 to encourage strong exploration in the early
    # stages and to escape local minima in the later stages.
    beta = 2.5
    
    # Calculate score
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
