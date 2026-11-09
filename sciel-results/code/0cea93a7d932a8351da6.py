"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based nonlinear oscillator model.
    
    Uses 5 Gaussian RBFs in the 3D feature space (t, x, v) plus a linear term.
    Each RBF has a center and width determined by params, allowing localized
    nonlinear responses that a global polynomial cannot capture.
    
    Features: t in [30,50], x in [-0.2056,0.1749], v in [-0.3391,0.3624]
    
    params layout (10 values):
      p0:  center_t offset for RBF 1 (scaled to [30,50] range)
      p1:  center_x for RBF 1
      p2:  center_v for RBF 1
      p3:  center_t offset for RBF 2
      p4:  center_x for RBF 2
      p5:  center_v for RBF 2
      p6:  width for RBF 1
      p7:  width for RBF 2
      p8:  scale/amplitude factor
      p9:  constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to roughly [-1, 1] for stable RBF evaluation
    t_norm = (t - 40.0) / 10.0       # [30,50] -> [-1,1]
    x_norm = x / 0.21                 # ~[-1,1]
    v_norm = v / 0.36                 # ~[-1,1]

    # RBF centers in normalized space
    # Center t: map p0, p3 from [-1,1] to [30,50] then normalize
    c1_t = p0
    c1_x = p1
    c1_v = p2
    c2_t = p3
    c2_x = p4
    c2_v = p5

    # Widths (ensure positive via softplus-like: exp ensures > 0)
    w1 = np.exp(p6) + 1e-6
    w2 = np.exp(p7) + 1e-6

    # Squared distances in 3D normalized feature space
    d1_sq = (t_norm - c1_t)**2 + (x_norm - c1_x)**2 + (v_norm - c1_v)**2
    d2_sq = (t_norm - c2_t)**2 + (x_norm - c2_x)**2 + (v_norm - c2_v)**2

    # Gaussian RBFs
    rbf1 = np.exp(-d1_sq / (2.0 * w1**2))
    rbf2 = np.exp(-d2_sq / (2.0 * w2**2))

    # Linear baseline in normalized features
    linear = p8 * (t_norm + x_norm + v_norm)

    return linear + rbf1 + rbf2 + p9
# EVOLVE-BLOCK-END
