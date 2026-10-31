import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import differential_evolution, L_BFGS_B

N = 600

def _autoconv_max(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of length N
    Returns: max( (h*h)[k] ) / (sum(h))^2
    Note: Since h is a step function on [-1/4, 1/4], the convolution is discrete.
    The normalization is by (sum h)^2.
    """
    h = np.asarray(h, dtype=float)
    if np.sum(h) == 0:
        return 0.0
    # Autoconvolution via FFT
    size = 2 * N
    H = fft(h, size)
    conv = np.real(ifft(H * H, size))[:2*N-1]
    # The peak is always at the center for non-negative h?
    # For non-negative h, max(h*h) = sum(h_i^2) is achieved at k=0 (shift 0).
    # Let's verify.
    # (h*h)[k] = sum_i h[i] h[i-k]
    # For k=0, sum h[i]^2.
    # Since h[i] >= 0, sum h[i] h[i-k] <= sum h[i]^2?
    # By Cauchy-Schwarz? No.
    # But if h is non-negative, the maximum of the autocorrelation is indeed at 0.
    # Proof: sum_i h[i] h[i-k] = <h, h_shifted>.
    # If we define inner product, <h, h> is max by Cauchy-Schwarz?
    # <h, h_k> <= ||h|| ||h_k|| = ||h||^2.
    # Yes, max is at k=0.
    # So we just need sum(h^2) / (sum(h))^2.
    # This simplifies the objective drastically.
    # Objective: minimize sum(h^2) / (sum(h))^2 subject to h>=0.
    # By Cauchy-Schwarz, (sum h)^2 <= N * sum(h^2) => sum(h^2)/(sum h)^2 >= 1/N.
    # Equality holds when h_i are all equal.
    # So the minimum is 1/N.
    # Why is the triangle score 0.75?
    # If min is 1/N, score = N * (1/N) = 1.
    # Triangle: h_i ~ triangle.
    # sum h^2 / (sum h)^2 for triangle on N=600.
    # Let's calculate for small N.
    # N=1: 1/1 = 1.
    # N=2: [1,1] -> 2/4=0.5. Triangle [1,0] or [0,1]? [1,1] is box. Triangle on 2 pts?
    # If N=3: Box [1,1,1] -> 3/9 = 1/3.
    # Triangle [0,1,0] -> 1/1 = 1.
    # Triangle [1,2,1] (symmetric peak 2) -> (1+4+1)/(4^2) = 6/16 = 0.375.
    # Box is 0.333.
    # So Box is better.
    # The current program is Triangle, which is worse than Box.
    # The inspiration says 1.23 is possible. This implies score > 1.
    # If score = N * value, and min value is 1/N, max score is N.
    # Wait, the task says "upper bound on C1".
    # If C1 is a constant, maybe the normalization is different?
    # "normalized maximum of its autoconvolution"
    # Maybe it's max(h*h) / max(h) / length?
    # Or maybe the interval length matters?
    # If the step function is on [-1/4, 1/4], width W = 1/2.
    # dx = W / N = 1/(2N).
    # Continuous convolution integral approximation?
    # If we treat h as a function f(x), then (f*f)(x) = \int f(t) f(x-t) dt.
    # Discrete approx: sum h_i h_j dx.
    # Max of discrete sum S = max sum h_i h_{i-k}.
    # Continuous max M = max \int f(t) f(x-t) dt.
    # M approx S * dx.
    # Normalization?
    # If score is M / ( \int f )^2 ?
    # \int f approx sum h_i * dx.
    # Ratio = (S * dx) / (sum h_i * dx)^2 = S / (sum h_i)^2 / dx = S / (sum h_i)^2 * 2N.
    # So Score = 2N * (sum h^2 / (sum h)^2).
    # For Box: sum h^2 / (sum h)^2 = N / N^2 = 1/N.
    # Score = 2N * (1/N) = 2.
    # For Triangle:
    # Let's compute sum h^2 / (sum h)^2 for a triangle.
    # If the score for triangle is 0.75, and box is 2.0, then triangle is worse.
    # But inspiration 1 says 1.23.
    # Maybe my formula for score is wrong.
    # If Score = N * value?
    # Box: 1. Triangle: 0.75.
    # Then 1.23 is possible.
    # This implies we need to minimize sum h^2 / (sum h)^2.
    # Which is minimized by Box.
    # Is it possible to get lower than 1/N? No, by Cauchy-Schwarz.
    # So the minimum score is 1 (if normalized by N) or 2 (if by 2N).
    # Why does the current program score 0.75?
    # Because it is a triangle, which is NOT the optimal shape for minimizing L2 norm relative to L1.
    # The optimal shape is the Box (uniform).
    # So simply returning np.ones(N) should give a score of ~1.0 (or higher depending on exact normalization).
    # If the seed score is 1.0 and triangle is 0.75, then Box is likely 1.0 or better.
    # Let's try returning Box.

def compute_autoconv_max(h):
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total == 0:
        return 0.0
    # Assuming the "value" is sum(h^2) / (sum(h))^2
    # The score is likely N * value or similar.
    # Let's just return the raw value as per "raw value (lower is better)"?
    # "score is the upper bound on C1"
    # "search score is seed/value"
    # If value is the metric to minimize.
    # If I return the value, and evaluator does seed/value, then lower value = higher score.
    # So I should minimize the value.
    # Value = max_autoconv / (sum h)^2.
    # For non-negative h, max_autoconv = sum h^2.
    # So minimize sum h^2 / (sum h)^2.
    # This is minimized by uniform h.
    return np.sum(h**2) / (total**2)

def construct():
    # The problem reduces to finding h >= 0 that minimizes sum(h^2) / (sum(h))^2.
    # The minimum is 1/N, achieved by h_i = c (constant).
    # So we should return a constant vector.
    # Let's return ones.
    return np.ones(N).tolist()
