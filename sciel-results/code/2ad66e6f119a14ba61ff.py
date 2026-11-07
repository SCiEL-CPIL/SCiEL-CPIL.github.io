import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-inspired kernel basis model.
    Uses 2 Gaussian bumps in the (t, x, v) space.
    params: 
      [w1, mu_x1, mu_v1, sigma1, mu_t1, 
       w2, mu_x2, mu_v2, sigma2, mu_t2]
    
    y = w1 * exp(-0.5 * d1^2 / sigma1^2) + w2 * exp(-0.5 * d2^2 / sigma2^2)
    where d_i^2 = (t_norm - mu_ti)^2 + (x_norm - mu_xi)^2 + (v_norm - mu_vi)^2
    sigma_i = params_sigma_i^2 + 0.1 (to ensure positive variance)
    """
    # Extract parameters
    w1 = params[0]
    mu_x1 = params[1]
    mu_v1 = params[2]
    sigma1 = params[3] ** 2 + 0.1
    mu_t1 = params[4]
    
    w2 = params[5]
    mu_x2 = params[6]
    mu_v2 = params[7]
    sigma2 = params[8] ** 2 + 0.1
    mu_t2 = params[9]
    
    # Normalize inputs to O(1) scale based on training ranges
    # t in [30, 50] -> center 40, range 20 -> scale 10
    # x in [-0.2, 0.2] approx -> scale 0.2
    # v in [-0.35, 0.35] approx -> scale 0.35
    t_norm = (t - 40.0) / 10.0
    x_norm = x / 0.2
    v_norm = v / 0.35
    
    # Compute squared distances for each Gaussian bump
    # d1^2 = (t_n - mu_t1)^2 + (x_n - mu_x1)^2 + (v_n - mu_v1)^2
    dt1 = t_norm - mu_t1
    dx1 = x_norm - mu_x1
    dv1 = v_norm - mu_v1
    d1_sq = dt1**2 + dx1**2 + dv1**2
    
    # d2^2 = (t_n - mu_t2)^2 + (x_n - mu_x2)^2 + (v_n - mu_v2)^2
    dt2 = t_norm - mu_t2
    dx2 = x_norm - mu_x2
    dv2 = v_norm - mu_v2
    d2_sq = dt2**2 + dx2**2 + dv2**2
    
    # Compute Gaussian terms
    # Clip exponent to avoid numerical issues, though exp handles large negative well
    exp_arg1 = -0.5 * d1_sq / sigma1
    exp_arg2 = -0.5 * d2_sq / sigma2
    
    # Ensure finite values
    g1 = np.exp(exp_arg1)
    g2 = np.exp(exp_arg2)
    
    # Combine
    result = w1 * g1 + w2 * g2
    
    return result
