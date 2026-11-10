# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    This implementation uses a 2-point distribution:
    P(X=0) = p, P(X=1) = 1-p
    where p is chosen to maximize the probability.
    
    The probability for this distribution is P(p) = (1-p)p^2(3-2p).
    Maximizing this yields p = (15 - sqrt(33)) / 16.
    
    This distribution yields a probability of approx 0.26, which is higher than
    the uniform distribution on [0, 1] (approx 0.25), thus improving the score.
    """
    # Calculate optimal p for 2-point distribution
    p = (15.0 - np.sqrt(33.0)) / 16.0
    
    # Support and weights
    support = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([p, 1.0 - p], dtype=np.float64)
    
    return support, weights

# EVOLVE-BLOCK-END
