"""Symbolic regression using random Fourier features for a linear-in-parameters model."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed random Fourier feature frequencies (deterministic seed for reproducibility)
_RNG = np.random.RandomState(42)
# Use 8 random frequencies for sin/cos pairs, scaled to cover [-2,2] well
_FREQS = _RNG.uniform(0.5, 5.0, size=4)  # 4 frequencies -> 8 features (sin+cos each)
# Precompute phase offsets for richer basis
_PHASES = _RNG.uniform(0, 2 * np.pi, size=4)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using random Fourier features + polynomial basis.
    Linear in parameters, so BFGS from any start point converges reliably.

    params (length 10):
        p0: constant
        p1: coefficient for x
        p2: coefficient for x^2
        p3: coefficient for x^3
        p4: coefficient for x^4
        p5-p8: coefficients for sin(freq_i * x + phase_i)
        p9: coefficient for cos(freq_0 * x + phase_0)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    # Random Fourier features: sin(f_i * x + phi_i) for i=0..3
    # and one cos term for the first frequency
    sin_features = np.column_stack([
        np.sin(f * x + p) for f, p in zip(_FREQS, _PHASES)
    ])
    cos_feature = np.cos(_FREQS[0] * x + _PHASES[0])

    # Combine all terms linearly
    result = (
        params[0]
        + params[1] * x
        + params[2] * x2
        + params[3] * x3
        + params[4] * x4
        + params[5] * sin_features[:, 0]
        + params[6] * sin_features[:, 1]
        + params[7] * sin_features[:, 2]
        + params[8] * sin_features[:, 3]
        + params[9] * cos_feature
    )

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
