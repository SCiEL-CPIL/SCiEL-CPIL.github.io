# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Truncated Fourier series basis with 10 fixed-frequency terms.
    The basis is:
      1 (constant),
      sin(w_k * x) and cos(w_k * x) for k = 1..4 (8 terms)
    Total: 1 + 8 = 9 terms. We add a 10th term: sin(5*w_1*x) to reach 10.
    
    Frequencies are chosen to span the domain [0.1, 6]:
      w_k = k * pi / 5.9  (fundamental ~ 0.532 rad/unit)
    
    This is linear in params, so BFGS solves a convex LSQ problem reliably.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Fixed frequencies: w_k = k * pi / 5.9 for k=1..4, plus 5*pi/5.9
    # Domain length ~ 5.9
    L = 5.9
    w1 = np.pi / L
    w2 = 2.0 * np.pi / L
    w3 = 3.0 * np.pi / L
    w4 = 4.0 * np.pi / L
    w5 = 5.0 * np.pi / L
    
    # Build basis: 10 columns
    # Col 0: constant 1
    # Cols 1-8: sin(w_k x), cos(w_k x) for k=1..4
    # Col 9: sin(w_5 x)
    
    c0 = np.ones_like(x)
    s1 = np.sin(w1 * x)
    c1 = np.cos(w1 * x)
    s2 = np.sin(w2 * x)
    c2 = np.cos(w2 * x)
    s3 = np.sin(w3 * x)
    c3 = np.cos(w3 * x)
    s4 = np.sin(w4 * x)
    c4 = np.cos(w4 * x)
    s5 = np.sin(w5 * x)
    
    basis = np.column_stack([c0, s1, c1, s2, c2, s3, c3, s4, c4, s5])
    
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
