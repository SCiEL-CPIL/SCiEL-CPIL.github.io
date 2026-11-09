"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    # Build a flexible basis of 10 functions
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        np.sin(x),
        np.cos(x),
        np.exp(0.5 * x),
        np.tanh(x),
        x * np.sin(x),
    ])

    # Normalize each basis function to unit RMS for better conditioning
    rms = np.sqrt(np.mean(basis ** 2, axis=0))
    rms = np.where(rms > 1e-10, rms, 1.0)
    basis = basis / rms

    # params has length 10; use first 10 as coefficients
    coeffs = params[:10]

    return basis @ coeffs
# EVOLVE-BLOCK-END
