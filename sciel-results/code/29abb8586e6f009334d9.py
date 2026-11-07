# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model using learnable Gaussian bumps and sine waves.
    
    Structure:
    - Gaussian 1: Amplitude p0, Center p1, Width log-scale p2
    - Sine Wave 1: Amplitude p3, Frequency log-scale p4, Phase p5
    - Gaussian 2: Amplitude p6, Center p7, Width log-scale p8
    - Offset: p9
    
    This allows the optimizer to find localized features and periodicities
    that a fixed linear basis might miss.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack parameters
    A1, C1, W1_log = params[0], params[1], params[2]
    A2, F1_log, Ph1 = params[3], params[4], params[5]
    A3, C2, W2_log = params[6], params[7], params[8]
    Offset = params[9]
    
    # Transform to ensure positivity and stability
    # Widths and Frequencies are positive
    W1 = np.exp(W1_log) + 1e-6
    F1 = np.exp(F1_log) + 0.1
    W2 = np.exp(W2_log) + 1e-6
    
    # Calculate components
    # Gaussian 1
    g1 = A1 * np.exp(-0.5 * ((x - C1) / W1) ** 2)
    
    # Sine Wave 1
    s1 = A2 * np.sin(F1 * x + Ph1)
    
    # Gaussian 2
    g2 = A3 * np.exp(-0.5 * ((x - C2) / W2) ** 2)
    
    # Combine
    y = g1 + s1 + g2 + Offset
    
    return y
# EVOLVE-BLOCK-END
