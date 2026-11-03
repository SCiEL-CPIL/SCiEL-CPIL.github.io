"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # The problem: minimize max_k (h * h)[k] / (sum(h)^2) over h >= 0
    # where (h*h)[k] is the discrete autoconvolution on 600 intervals.
    # Note: the autoconvolution of a length-N sequence has length 2N-1.
    # But the task says "normalized maximum of its autoconvolution".
    # Let's interpret: h is a vector of N nonnegative heights.
    # The autoconvolution c[m] = sum_i h[i] h[i+m] for m in 0..N-1 (or full 2N-1).
    # "Normalized" likely means divide by (sum h)^2.
    # We want to minimize max_m c[m] / (sum h)^2.
    
    # Let S = sum(h). By scaling invariance, we can fix S = 1.
    # Then we want to minimize max_m c[m] subject to sum(h)=1, h>=0.
    # This is a linear program:
    #   minimize t
    #   subject to:
    #     c[m] <= t  for all m
    #     sum(h) = 1
    #     h >= 0
    #
    # c[m] = sum_i h[i] h[i+m] is quadratic in h, so this is NOT a linear program.
    # The directive suggests using linprog, but the objective is quadratic.
    #
    # Wait, the directive says: "Formulate the problem as a linear program by introducing
    # a scalar upper bound variable for the normalized autoconvolution peak and solving
    # for the nonnegative step heights via scipy.optimize.linprog"
    #
    # This is only possible if we linearize. One way: fix the heights to be a step function
    # with specific structure, or use a different formulation.
    #
    # Actually, for a step function with N equal intervals, the autoconvolution is:
    # c[m] = sum_{i=0}^{N-1-m} h[i] h[i+m]
    # This is quadratic. LP cannot handle this directly.
    #
    # However, if we consider the continuous version or use a specific parametrization...
    #
    # Let me reconsider: maybe the "step function" means the function is constant on each
    # interval, and the autoconvolution is computed in a way that can be linearized?
    #
    # Alternative interpretation: Perhaps we can use the fact that for the purpose of
    # bounding C1 (first autocorrelation inequality), there's a known optimal solution.
    #
    # Let's try a different approach: since the problem is convex (minimize max of convex
    # functions over a convex set), we can use bisection on t and check feasibility.
    # For a given t, check if there exists h >= 0, sum(h)=1, such that c[m] <= t for all m.
    # This is still a quadratic constraint, not linear.
    #
    # Given the time constraints and the fact that linprog is suggested, let me try a
    # simplified approach: assume the optimal h is uniform or has a simple form.
    #
    # Actually, let's just implement a gradient-based approach with a deadline guard,
    # since the LP formulation as described doesn't work directly for quadratic objectives.
    
    # Start with uniform distribution
    h = np.ones(N) / N
    best_h = h.copy()
    best_val = _compute_objective(h)
    
    # Use projected gradient descent
    lr = 0.01
    for step in range(5000):
        if time.time() > deadline:
            break
        
        # Compute autoconvolution
        c = np.convolve(h, h, mode='full')
        # c has length 2N-1
        S = np.sum(h)
        norm_c = c / (S * S)
        obj = np.max(norm_c)
        
        # Update best
        if obj < best_val:
            best_val = obj
            best_h = h.copy()
        
        # Gradient of objective with respect to h
        # d/dh_i [max_m c[m]/S^2]
        # This is tricky due to max. Use the subgradient at the argmax.
        m_star = np.argmax(norm_c)
        
        # c[m] = sum_j h[j] h[j+m]
        # dc[ms]/dh[i] = h[i+ms] + h[i-ms] (with appropriate bounds)
        # More carefully: c[m] = sum_{j} h[j] h[j+m] where j+m < N
        # dc[m]/dh[i] = h[i+m] (if i+m < N) + h[i-m] (if i-m >= 0)
        
        grad_c = np.zeros(N)
        for j in range(N):
            if j + m_star < N:
                grad_c[j] += h[j + m_star]
            if j - m_star >= 0:
                grad_c[j] += h[j - m_star]
        
        # d/dh [c[ms]/S^2] = grad_c / S^2 - 2*c[ms]*h / S^3
        # = (grad_c * S - 2*c[ms]*h) / S^3
        
        grad_obj = (grad_c * S - 2 * c[m_star] * h) / (S * S * S)
        
        # Projected gradient step: h_new = h - lr * grad_obj, then project to simplex
        h_new = h - lr * grad_obj
        # Project to {h >= 0, sum(h) = 1}
        h_new = _project_simplex(h_new)
        
        h = h_new
    
    return list(best_h)


def _compute_objective(h):
    c = np.convolve(h, h, mode='full')
    S = np.sum(h)
    if S == 0:
        return np.inf
    return np.max(c) / (S * S)


def _project_simplex(v):
    """Project v onto the probability simplex {x >= 0, sum(x) = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


# EVOLVE-BLOCK-END
