"""RBF BASIS FOR E. COLI GROWTH RATE."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

# Precompute fixed RBF centers in normalized space [0,1]^4
# Centers: 4 corners + 6 mid-edge/centroid points
# Corner centers
_c0 = np.array([0.0, 0.0, 0.0, 0.0])
_c1 = np.array([1.0, 1.0, 1.0, 1.0])
_c2 = np.array([0.0, 1.0, 0.0, 1.0])
_c3 = np.array([1.0, 0.0, 1.0, 0.0])
# Mid-edge / interior centers
_c4 = np.array([0.5, 0.5, 0.5, 0.5])  # centroid
_c5 = np.array([0.0, 0.0, 0.5, 0.5])  # mid temp/pH
_c6 = np.array([1.0, 1.0, 0.5, 0.5])  # mid temp/pH high b,s
_c7 = np.array([0.5, 0.0, 0.0, 0.0])  # low s,temp,pH
_c8 = np.array([0.5, 1.0, 1.0, 1.0])  # high s,temp,pH
_c9 = np.array([0.0, 0.5, 0.5, 0.0])  # low b, mid s/temp, low pH

# Stack centers: (10, 4)
_CENTERS = np.array([_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9])

# RBF bandwidth (inverse of width). Larger = narrower Gaussians.
# Tuned for normalized [0,1] space
_BW = 2.5

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-Center Radial Basis Function (RBF) Basis.
    
    Uses 10 Gaussian basis functions at fixed centers in normalized input space.
    Parameters are the weights for each RBF kernel.
    
    Input Ranges:
    b: [1, 5]
    s: [1, 40]
    temp: [1, 12]
    pH: [1, 5]
    """
    
    # Normalize inputs to [0, 1]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0
    
    # Clip to [0, 1] for stability
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    t_n = np.clip(t_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # Stack normalized inputs: (N, 4)
    X = np.stack([b_n, s_n, t_n, p_n], axis=-1)
    
    # Compute squared distances from each point to each center
    # X: (N, 1, 4), _CENTERS: (1, 10, 4) -> diff: (N, 10, 4)
    diff = X[:, np.newaxis, :] - _CENTERS[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)  # (N, 10)
    
    # Gaussian RBF: exp(-bw * sq_dist)
    rbf = np.exp(-_BW * sq_dist)  # (N, 10)
    
    # Linear combination with learned weights
    pred = rbf @ params
    
    return pred

# EVOLVE-BLOCK-END
