import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    safe_cost = np.maximum(cost, 1e-9)
    
    # Cost-aware Expected Hypar Improvement (EHI)
    # EHI uses tanh to model the trade-off between improvement and cost.
    # For minimization, improvement = f_best - f(x)
    # We want to maximize E[tanh((f_best - f(x)) / c) / cost]
    # where c is a scaling parameter for the tanh saturation.
    
    # The expected value of tanh((f_best - f(x)) / c) under f(x) ~ N(mu, sigma^2)
    # can be approximated. For a normal distribution, E[tanh((a - X)/c)] where X~N(mu, sigma^2)
    # We use a numerical approximation or the fact that for small sigma/c, 
    # E[tanh((f_best - mu)/c)] * exp(-sigma^2/(2c^2)) approximately.
    
    # Actually, let's use the exact expectation for EHI which involves:
    # E[tanh((f_best - f(x))/c)] = integral of tanh((f_best - z)/c) * N(z; mu, sigma^2) dz
    
    # A common approximation: if we let y = (f_best - f(x))/c, then y ~ N((f_best-mu)/c, sigma^2/c^2)
    # E[tanh(y)] for y ~ N(m, s^2) can be computed numerically or approximated.
    
    # For efficiency, use the approximation that for moderate s:
    # E[tanh(y)] ≈ tanh(m) * exp(-s^2/2) for small s, but this isn't quite right.
    
    # Better: use the identity that E[tanh((f_best - f(x))/c)] can be computed via
    # numerical integration or the following approximation:
    # Let m = (f_best - mu)/c, s = sigma/c
    # E[tanh(y)] where y ~ N(m, s^2)
    # For s << 1: ≈ tanh(m)
    # For s >> 1: ≈ 0 (due to symmetry of tanh around 0)
    
    # We can use a quadrature-free approximation or simply compute it numerically
    # Since we have 2000 points, let's use a vectorized approach with a few Gauss-Hermite points
    
    # Alternatively, use the fact that EHI is often approximated as:
    # EHI ≈ (1/cost) * E[tanh((f_best - f(x))/c)]
    
    # Let's compute E[tanh((f_best - f(x))/c)] using Gauss-Hermite quadrature with 5 points
    # y = (f_best - f(x))/c, f(x) ~ N(mu, sigma^2)
    # y ~ N((f_best-mu)/c, (sigma/c)^2)
    
    c = 0.5  # scaling parameter for tanh saturation
    
    m = (f_best - mu) / c
    s = safe_sigma / c
    
    # Gauss-Hermite quadrature with 5 nodes
    # Nodes and weights for 5-point GH
    gh_nodes = np.array([
        -1.495897871133126,
        -0.763943726042316,
        0.0,
        0.763943726042316,
        1.495897871133126
    ])
    gh_weights = np.array([
        0.019953242155620,
        0.094642873606734,
        0.239314335249683,
        0.094642873606734,
        0.019953242155620
    ])
    
    # E[tanh(y)] where y ~ N(m, s^2)
    # = (1/sqrt(pi)) * integral exp(-t^2) * tanh(m + sqrt(2)*s*t) dt
    # Using GH: ≈ (1/sqrt(pi)) * sum(w_i * tanh(m + sqrt(2)*s*x_i))
    
    sqrt_pi = np.sqrt(np.pi)
    sqrt_2 = np.sqrt(2.0)
    
    # Vectorized computation
    # m has shape (2000,), s has shape (2000,)
    # We need to compute for each point: sum_i w_i * tanh(m_j + sqrt_2 * s_j * x_i)
    
    # Reshape for broadcasting: m and s are (2000,), nodes are (5,)
    m_exp = m[:, np.newaxis]  # (2000, 1)
    s_exp = s[:, np.newaxis]  # (2000, 1)
    nodes = gh_nodes[np.newaxis, :]  # (1, 5)
    weights = gh_weights[np.newaxis, :]  # (1, 5)
    
    # Compute tanh values: (2000, 5)
    tanh_vals = np.tanh(m_exp + sqrt_2 * s_exp * nodes)
    
    # Weighted sum: (2000,)
    expected_tanh = np.sum(weights * tanh_vals, axis=1) / sqrt_pi
    
    # EHI score = expected_tanh / cost
    score = expected_tanh / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
