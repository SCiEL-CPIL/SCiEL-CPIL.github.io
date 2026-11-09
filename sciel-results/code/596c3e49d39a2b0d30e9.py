"""
Hybrid Adaptive Basis:
1. Fixed Constant: Ensures global mean fitting capability.
2. Adaptive Orthogonal Basis: Uses QR decomposition on a rich candidate pool
   (Polynomials, Radial, Trig) to select 9 orthogonal features.
This improves conditioning and expressive power compared to fixed hand-picked bases.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 10.0


def _elapsed_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure 1-D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    n = x.size

    if not _elapsed_ok():
        # Safety fallback: return constant (mean estimator)
        return np.full(n, float(p[0]))

    # 1. Fixed Constant Basis
    b0 = np.ones(n)
    
    # 2. Generate Candidate Pool
    # We generate a diverse set of candidate features to span common function families.
    # x, y are in [-1, 1].
    
    # Polynomials (up to degree 5)
    # x^k, y^k
    x_pows = [x**k for k in range(1, 6)]
    y_pows = [y**k for k in range(1, 6)]
    
    # Interactions (x^i * y^j)
    # Keep low degree to avoid numerical instability
    interactions = []
    for i in range(1, 4):
        for j in range(1, 4):
            interactions.append((x**i) * (y**j))
            
    # Radial / Distance terms
    r2 = x**2 + y**2
    r = np.sqrt(r2 + 1e-9)
    radial = [r, r2, r2**2, r2**3] # r, r^2, r^4, r^6
    
    # Trigonometric
    trig = [
        np.sin(x), np.cos(x),
        np.sin(y), np.cos(y),
        np.sin(x+y), np.cos(x+y),
        np.sin(x-y), np.cos(x-y),
        np.sin(2*x), np.sin(2*y)
    ]
    
    # Combine all candidates
    candidates = x_pows + y_pows + interactions + radial + trig
    
    # Stack into matrix (n x M)
    # Note: We don't include the constant here because we handle it separately as b0.
    # However, QR on the candidates might pick up components correlated with 1.
    # To ensure orthogonality between the adaptive part and the constant,
    # we can project out the mean from the candidates first.
    
    # Project out mean from candidates to ensure orthogonality to b0
    # C = Candidates (n x M)
    C = np.column_stack(candidates)
    
    # Center columns
    C_mean = C.mean(axis=0, keepdims=True)
    C_centered = C - C_mean
    
    # QR Decomposition
    # We want 9 orthogonal vectors.
    # If M < 9, we just use what we have (but M is ~ 5+5+9+4+10 = 33, so plenty).
    try:
        Q, R = np.linalg.qr(C_centered, mode='reduced')
        # Take first 9 columns
        # Handle potential rank deficiency or small dimensions
        if Q.shape[1] >= 9:
            Q_adaptive = Q[:, :9]
        else:
            # Pad with zeros if not enough (unlikely)
            Q_adaptive = np.zeros((n, 9))
            Q_adaptive[:, :Q.shape[1]] = Q
    except np.linalg.LinAlgError:
        # Fallback to simple polynomial basis if QR fails
        # x, y, x^2, y^2, xy, sin(x), sin(y), sin(x+y), x*y
        Q_adaptive = np.column_stack([
            x, y, x**2, y**2, x*y,
            np.sin(x), np.sin(y), np.sin(x+y), x**2*y
        ])
        
    # Construct final Basis: [b0, Q_adaptive]
    # Shape: n x 10
    basis = np.column_stack((b0, Q_adaptive))
    
    # Prediction
    return basis @ p
# EVOLVE-BLOCK-END
