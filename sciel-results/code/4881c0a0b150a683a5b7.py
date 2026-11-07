"""Symbolic regression with 1 input using a Fourier basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with 10 parameters.
    
    Uses a constant term plus 4 pairs of sine/cosine terms at harmonics
    1, 2, 3, 4. This captures smooth periodic or quasi-periodic structures
    efficiently, which is often better than local RBF kernels for global
    smooth functions.
    
    The 10 parameters are:
    - params[0]: constant (DC) term
    - params[1:2]: sin/cos at frequency 1
    - params[2:4]: sin/cos at frequency 2
    - params[4:6]: sin/cos at frequency 3
    - params[6:8]: sin/cos at frequency 4
    - params[8:10]: sin/cos at frequency 5
    
    x is mapped from [0.1, 6.0] to phase [0, 2*pi] for the Fourier terms.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # Map x from [0.1, 6.0] to phase in [0, 2*pi]
    x_min = 0.1
    x_max = 6.0
    n = len(x)
    
    # Base frequency: one full cycle across the domain
    # phase = 2*pi * (x - x_min) / (x_max - x_min)
    phase = 2.0 * np.pi * (x - x_min) / (x_max - x_min)
    
    # Build basis matrix with 10 columns
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Column 0: constant
    basis[:, 0] = 1.0
    
    # Columns 1-9: sine and cosine at harmonics 1 through 5
    for k in range(1, 6):
        col_sin = 2 * (k - 1) + 1  # odd indices: 1, 3, 5, 7, 9
        col_cos = 2 * (k - 1) + 2  # even indices: 2, 4, 6, 8, 10 -> but we only have up to 9
        # Actually: for k=1: sin at col 1, cos at col 2
        #           for k=2: sin at col 3, cos at col 4
        #           for k=3: sin at col 5, cos at col 6
        #           for k=4: sin at col 7, cos at col 8
        #           for k=5: sin at col 9, cos at col 10 -> out of range!
        # Let me fix: we have 10 params total, so:
        # param[0] = constant
        # param[1], param[2] = sin, cos at freq 1
        # param[3], param[4] = sin, cos at freq 2
        # param[5], param[6] = sin, cos at freq 3
        # param[7], param[8] = sin, cos at freq 4
        # param[9] = sin at freq 5 (only one more slot)
        # Actually, let's use 4 full harmonics + 1 extra:
        # Better: use harmonics 1, 2, 3, 4 with both sin and cos = 8 params + 2 extra
        # Let's do: constant + 4 pairs (sin, cos) at freqs 1-4 = 9 params, + 1 more sin at freq 5
        pass
    
    # Redo properly:
    # params[0]: constant
    # params[1], params[2]: sin, cos at freq 1
    # params[3], params[4]: sin, cos at freq 2
    # params[5], params[6]: sin, cos at freq 3
    # params[7], params[8]: sin, cos at freq 4
    # params[9]: sin at freq 5 (or we could use cos at freq 0.5, etc.)
    
    basis[:, 0] = 1.0
    basis[:, 1] = np.sin(phase)
    basis[:, 2] = np.cos(phase)
    basis[:, 3] = np.sin(2.0 * phase)
    basis[:, 4] = np.cos(2.0 * phase)
    basis[:, 5] = np.sin(3.0 * phase)
    basis[:, 6] = np.cos(3.0 * phase)
    basis[:, 7] = np.sin(4.0 * phase)
    basis[:, 8] = np.cos(4.0 * phase)
    basis[:, 9] = np.sin(5.0 * phase)
    
    return basis @ params
# EVOLVE-BLOCK-END
