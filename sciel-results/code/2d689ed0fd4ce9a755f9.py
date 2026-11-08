"""
Strategy: Spectral Minimax via Fourier Coefficients with Smooth Max Objective
This approach fundamentally changes the representation from spatial step heights to
Fourier coefficients in the frequency domain.

Key Insights:
1. The overlap integral I(t) = ∫ h(x)(1-h(x-t)) dx depends on the autocorrelation of h.
   For h(x) = ∑ c_k e^{iπkx} (period 2), the autocorrelation A(t) = ∫ h(x)h(x-t)dx
   is a trigonometric polynomial with coefficients |c_k|^2.
   Specifically, A(t) = ∑_{k} |c_k|^2 e^{iπkt}.
   Since h is real, c_{-k} = c_k^*, so A(t) is real and even.
   A(t) = c_0^2 + 2 ∑_{k=1}^{K} Re(c_k^2 e^{iπkt}) ... wait.
   Let's be precise.
   h(x) = c_0 + ∑_{k=1}^K (a_k cos(πkx) + b_k sin(πkx)).
   A(t) = ∫ h(x)h(x-t)dx.
   Using orthogonality:
   ∫ cos(πkx)cos(πk(x-t)) dx = cos(πkt) ∫ cos^2(πkx) dx = 0.5 cos(πkt).
   ∫ sin(πkx)sin(πk(x-t)) dx = 0.5 cos(πkt).
   Cross terms vanish.
   So A(t) = c_0^2 + 0.5 ∑_{k=1}^K (a_k^2 + b_k^2) cos(πkt).
   
   The overlap is I(t) = 1 - A(t) (since ∫h = 1, ∫h^2 is part of A(0)? No.
   ∫ h(x)(1-h(x-t)) dx = ∫ h(x)dx - ∫ h(x)h(x-t)dx = 1 - A(t).
   Yes.
   
   So we want to minimize max_t (1 - A(t)), which is equivalent to maximizing min_t A(t).
   A(t) = c_0^2 + 0.5 ∑_{k=1}^K r_k^2 cos(πkt), where r_k^2 = a_k^2 + b_k^2.
   
   Constraints:
   1. ∫ h = 1. Since domain is [0,2], ∫ h = 2 * c_0. So 2 c_0 = 1 => c_0 = 0.5.
   2. 0 <= h(x) <= 1 for all x.
      h(x) = 0.5 + ∑_{k=1}^K (a_k cos(πkx) + b_k sin(πkx)).
      This is a non-convex constraint in (a_k, b_k).
      
   However, we can optimize over the coefficients (a_k, b_k) and use a penalty or
   projection to enforce 0 <= h <= 1.
   
   The objective is to maximize min_t A(t).
   A(t) is a continuous function of t. We can evaluate it on a dense grid.
   To make it smooth for optimization, we can use a soft-min or just minimize the negative
   of the minimum on a fine grid. Or better, use a log-sum-exp approximation of the max
   of the violations, but here we want max of A(t) to be large? No, we want min A(t) to be large.
   So we want to maximize min_t A(t).
   This is equivalent to minimizing -min_t A(t).
   The function f(t) = A(t) is smooth. The minimum is likely at points where f'(t)=0.
   
   Alternative: Since A(t) is a cosine series, its maximum and minimum are bounded by
   the sum of the coefficients.
   Specifically, A(t) is between c_0^2 - 0.5 ∑ r_k^2 and c_0^2 + 0.5 ∑ r_k^2.
   To maximize the minimum, we should try to make A(t) as flat as possible, or at least
   ensure the dips are not too deep.
   
   If we set all a_k, b_k = 0, A(t) = 0.25 constant. Min A = 0.25.
   Overlap = 1 - 0.25 = 0.75.
   
   Can we do better?
   If we add high frequency components, they oscillate faster.
   The minimum of A(t) will be at the points where the cosines are -1.
   If the frequencies are commensurate, the minimum is sharp.
   
   Let's try to optimize the coefficients a_k, b_k for k=1..K.
   We will use scipy.optimize.minimize with a smooth approximation of the minimum.
   Actually, minimizing the maximum overlap is the same as maximizing the minimum autocorrelation.
   
   Let's define the objective to be minimized:
   J = -min_{t in Grid} A(t) + Penalty(h < 0 or h > 1).
   
   To handle the box constraint 0 <= h <= 1, we can use a barrier or penalty.
   Since the optimal solution for such problems often lies on the boundary (h is 0 or 1 mostly),
   a simple penalty might struggle.
   
   However, note that for K=0, h=0.5 is the only solution.
   For K>=1, we can vary h.
   
   Let's use a different insight:
   The problem is equivalent to finding a set of points on the unit circle (for complex coefficients)
   that minimizes the peak sidelobe of the autocorrelation? No, it's the peak of the cross-correlation
   with the complement.
   
   Let's stick to the spatial domain but use a smarter optimizer:
   The previous program used piecewise linear interpolation and L-BFGS-B.
   The new strategy will use **Fourier Coefficients** as the optimization variables.
   This reduces the dimensionality if K is small, and the smoothness of the Fourier basis
   might allow better global optimization via random restarts in the coefficient space.
   
   We will optimize K=5 or K=10 Fourier modes.
   Variables: a_1..a_K, b_1..b_K.
   h(x) = 0.5 + ∑ (a_k cos(πkx) + b_k sin(πkx)).
   
   Objective: Maximize min_t A(t).
   A(t) = 0.25 + 0.5 ∑_{k=1}^K (a_k^2 + b_k^2) cos(πkt).
   
   Constraint: 0 <= h(x) <= 1.
   
   We can use scipy.optimize.minimize to minimize:
   f(params) = -min_{t_grid} A(t) + λ * ∫ (max(0, -h(x))^2 + max(0, h(x)-1)^2) dx
   
   This is a smooth objective (if we use a dense grid for min and integral).
   
   Let's implement this.
"""

import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_A_and_h(a, b, x_grid, t_grid):
    """
    Compute A(t) and h(x) from Fourier coefficients.
    a, b: arrays of length K.
    x_grid, t_grid: 1D arrays.
    """
    K = len(a)
    # h(x) = 0.5 + sum(a_k cos(pi k x) + b_k sin(pi k x))
    h = np.full_like(x_grid, 0.5)
    # A(t) = 0.25 + 0.5 * sum((a_k^2 + b_k^2) cos(pi k t))
    A = np.full_like(t_grid, 0.25)
    
    for k in range(1, K + 1):
        angle_x = np.pi * k * x_grid
        h += a[k-1] * np.cos(angle_x) + b[k-1] * np.sin(angle_x)
        
        angle_t = np.pi * k * t_grid
        A += 0.5 * (a[k-1]**2 + b[k-1]**2) * np.cos(angle_t)
        
    return h, A

def _objective(params, x_grid, t_grid, lambda_penalty=10.0):
    """
    Objective function to minimize.
    params: [a_1..a_K, b_1..b_K]
    """
    K = len(params) // 2
    a = params[:K]
    b = params[K:]
    
    h, A = _compute_A_and_h(a, b, x_grid, t_grid)
    
    # We want to maximize min(A), so minimize -min(A)
    min_A = np.min(A)
    obj = -min_A
    
    # Penalty for h out of [0,1]
    # Use squared hinge loss
    h_neg = np.clip(-h, 0, None)
    h_pos = np.clip(h - 1, 0, None)
    penalty = lambda_penalty * (np.sum(h_neg**2) + np.sum(h_pos**2)) / len(h_grid)
    
    return obj + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Resolution for evaluation
    N_x = 1024
    N_t = 1024
    L = 2.0
    x_grid = np.linspace(0, L, N_x, endpoint=False)
    t_grid = np.linspace(0, L, N_t, endpoint=False)
    
    # Number of Fourier modes
    # Start with small K, maybe increase if time permits
    K = 5
    
    # Initial guess: all zeros (h=0.5)
    best_params = np.zeros(2 * K)
    best_score = _objective(best_params, x_grid, t_grid)
    
    # Random restarts
    np.random.seed(42)
    
    # We need to ensure h stays in [0,1].
    # The max amplitude of the sum is sum |a_k| + |b_k|.
    # To be safe, keep coefficients small initially.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale by 0.1 to start near the feasible region
        scale = 0.1
        params_init = np.random.uniform(-scale, scale, 2 * K)
        
        # Optimize
        try:
            res = minimize(_objective, params_init, args=(x_grid, t_grid), 
                           method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Refine with higher K if time permits
    if time.time() < deadline - 20:
        K2 = 10
        # Interpolate coefficients? No, just start with zeros for new modes
        params_init = np.zeros(2 * K2)
        params_init[:2*K] = best_params
        
        try:
            res2 = minimize(_objective, params_init, args=(x_grid, t_grid), 
                           method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
                K = K2
        except Exception:
            pass
    
    # Convert optimized coefficients back to step heights
    N_out = 100
    x_centers = (np.arange(N_out) + 0.5) * (L / N_out)
    
    K_final = len(best_params) // 2
    a = best_params[:K_final]
    b = best_params[K_final:]
    
    h_out, _ = _compute_A_and_h(a, b, x_centers, np.array([0.0]))
    h_out = h_out.flatten()
    
    # Ensure 0 <= h <= 1 and sum = N_out / 2
    # The optimizer should have kept it in bounds, but let's project just in case.
    # Project to [0,1]^N with sum constraint.
    
    # Clip
    h_out = np.clip(h_out, 0, 1)
    
    # Normalize sum to N_out / 2
    target_sum = N_out / 2.0
    current_sum = np.sum(h_out)
    
    if current_sum > 0:
        h_out = h_out * (target_sum / current_sum)
    
    # Re-clip and re-normalize iteratively
    for _ in range(10):
        h_out = np.clip(h_out, 0, 1)
        s = np.sum(h_out)
        if s == 0:
            h_out = np.full(N_out, target_sum / N_out)
            break
        h_out = h_out * (target_sum / s)
        
    # Final exact sum adjustment
    diff = target_sum - np.sum(h_out)
    if abs(diff) > 1e-9:
        # Distribute difference to the element with the most room
        # Or just add to last element if possible
        if 0 <= h_out[-1] + diff <= 1:
            h_out[-1] += diff
        else:
            # Fallback: distribute evenly
            h_out = h_out + diff / N_out
            h_out = np.clip(h_out, 0, 1)
            # Re-normalize
            s = np.sum(h_out)
            if s > 0:
                h_out = h_out * (target_sum / s)

    return h_out

# EVOLVE-BLOCK-END
