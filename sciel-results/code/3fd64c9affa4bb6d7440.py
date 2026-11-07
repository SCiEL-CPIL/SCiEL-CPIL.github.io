import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function.
    
    This strategy uses a deterministic score: score = mu + kappa * sigma.
    Since we are minimizing, we want to maximize the probability of finding
    a value lower than f_best. However, standard UCB for minimization is 
    often formulated as minimizing mu - kappa * sigma (Lower Confidence Bound) 
    OR maximizing the potential for improvement.
    
    Wait, let's look at the standard definitions.
    For minimization:
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    UCB for minimization is typically: LCB(x) = mu(x) - kappa * sigma(x).
    We pick the x that minimizes LCB(x).
    The task says: "Return 2000 finite scores, higher meaning evaluate next".
    So if we want to pick the x that minimizes LCB(x), we should return -LCB(x) = -mu(x) + kappa * sigma(x).
    
    Let's verify the direction.
    If we return scores where higher is better, and we want to select the candidate 
    with the lowest predicted minimum (mu - kappa*sigma), then the score should be 
    inversely related to (mu - kappa*sigma).
    Score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    Why UCB/LCB over Thompson Sampling?
    Thompson Sampling (current) has high variance. In a small budget (30 evals), 
    a single bad random draw in TS can waste an evaluation on a clearly bad region.
    LCB (Lower Confidence Bound) is deterministic and robust. It explicitly targets 
    regions where the mean is low OR the uncertainty is high (scaled by kappa).
    
    We need to choose kappa.
    A common choice for small budgets is a fixed kappa around 1.0 to 2.0.
    However, since sigma is in objective units and the function range is small 
    (Weierstrass 1D min is 0, max is small), we can use a fixed kappa.
    
    Let's use kappa = 1.5. This provides a strong exploration bonus.
    As the GP learns, sigma decreases in explored regions, so the bonus fades, 
    naturally shifting to exploitation.
    
    Additionally, we can add a small epsilon-greedy-like noise or just stick to 
    pure LCB. Pure LCB is often superior for small budgets because it avoids 
    the "luck of the draw" problem.
    
    Let's refine the score:
    score = -mu + kappa * sigma
    
    We should also ensure sigma is not zero to avoid numerical issues, though 
    the GP provides noise floor.
    """
    # Ensure sigma is non-negative and handle any numerical zeros
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa?
    # For 30 evaluations, a fixed kappa is usually sufficient and more stable.
    # Let's try kappa = 2.0 to encourage exploration early on, as the 
    # Weierstrass function has many local minima.
    kappa = 2.0
    
    # Calculate Lower Confidence Bound (LCB) for minimization
    # lcb = mu - kappa * sigma
    # We want to maximize the score, and the best candidate has the lowest lcb.
    # So score = -lcb = -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    return scores
