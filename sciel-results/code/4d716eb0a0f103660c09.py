# EVOLVE-BLOCK-START
import numpy as np
import time


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Uses a 2-point distribution {0, M} with optimized probability p.
    For {0, M} with P(X=0)=p, P(X=M)=1-p:
    P[X1+X2+X3 < 2*X4] = (1-p) * (p^3 + 3*p^2*(1-p))
    Optimized at p = (15 - sqrt(33))/16.
    """
    start_time = time.time()
    
    # Optimal 2-point distribution
    p = (15.0 - np.sqrt(33.0)) / 16.0
    # Use a large M to ensure strict inequality works cleanly
    M = 1.0  # Any positive M works; the structure is what matters
    
    support = np.array([0.0, M], dtype=np.float64)
    weights = np.array([p, 1.0 - p], dtype=np.float64)
    
    # Validate
    assert len(support) == len(weights)
    assert len(support) >= 1
    assert np.all(np.isfinite(support))
    assert np.all(np.isfinite(weights))
    assert np.all(support >= -1e-9)
    assert np.all(weights >= -1e-9)
    assert np.sum(weights) > 1e-9
    
    return support, weights


# EVOLVE-BLOCK-END
