import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Optimized softplus
    def softplus(x):
        # x is array or scalar
        # For x > 20, exp(x) overflows or becomes very large, log1p(exp(x)) ~ x
        # For x < -20, exp(x) ~ 0, log1p(exp(x)) ~ exp(x)
        # np.where is safe
        return np.where(x > 20, x, np.log1p(np.exp(x)))
    
    # Parameters
    K_s        = softplus(p[0]) + 1e-6
    T_opt      = p[1]
    T_slope_l  = softplus(p[2]) + 1e-6
    T_slope_h  = softplus(p[3]) + 1e-6
    pH_opt     = p[4]
    pH_width   = softplus(p[5]) + 1e-6
    mu_max     = softplus(p[6])
    b_slope    = p[7]
    b_shift    = p[8]
    mu_base    = p[9]
    
    # Normalizations
    bn = (b - 1.0) / 4.0
    
    # Monod Kinetics
    sfac = s / (K_s + s + 1e-9)
    
    # Skewed Temperature Response
    # dist = softplus(delta) - ln(2) ensures dist(0) = 0
    ln2 = np.log(2.0)
    d_low  = softplus(T_opt - temp) - ln2
    d_high = softplus(temp - T_opt) - ln2
    
    # Ensure non-negative distances for the exponential decay
    # softplus(x) - ln(2) is always >= 0?
    # f(x) = log(1+exp(x)) - log(2)
    # f(0) = 0.
    # f(-inf) = 0 - 0 = 0? No. exp(-inf)=0. log(1)=0. 0-ln2 = -ln2.
    # Wait.
    # If x is very negative, exp(x) is 0. log(1) = 0.
    # So softplus(x) ~ 0.
    # softplus(x) - ln(2) ~ -ln(2).
    # This is negative.
    # We want the distance to be 0 when T is far from T_opt? No, we want the penalty to be large.
    # The exponential term is exp(-slope * dist).
    # If dist is negative, exp(-slope * negative) = exp(positive) = Large.
    # This would INCREASE the growth rate far from optimum. That's wrong.
    
    # We need dist to be positive and increasing as we move away from T_opt.
    # softplus(x) is monotonic increasing.
    # softplus(x) - softplus(0) is not the right shift for "distance".
    # We want a function that is 0 at x=0 and positive elsewhere.
    # softplus(x) - softplus(0) is negative for x < 0.
    
    # Let's use:
    # dist_low = softplus(T_opt - temp) - softplus(0)  # This is negative for T > T_opt?
    # No, T_opt - temp is positive when T < T_opt.
    # If T < T_opt, delta > 0. softplus(delta) > softplus(0). dist > 0.
    # If T > T_opt, delta < 0. softplus(delta) < softplus(0). dist < 0.
    # But we only apply T_slope_l to d_low.
    # d_low is multiplied by T_slope_l.
    # If d_low is negative, exp(-slope * negative) > 1.
    # This means for T > T_opt, the "low" part contributes > 1.
    # But we also have d_high.
    # For T > T_opt, d_high > 0.
    # The sum is term_low + term_high.
    # If term_low becomes negative, it might cancel out term_high?
    
    # Better approach:
    # Use np.maximum(0, delta) smoothed by softplus?
    # Actually, softplus(x) is a smooth approximation of max(0, x).
    # So softplus(x) IS the distance measure we want!
    # It is 0 at x=0 (approx, actually ln2) and grows linearly for x>0.
    # For x<0, it is small but positive (exp(x)).
    # So softplus(x) is always positive.
    # We just need to subtract the baseline value at x=0 to center it?
    # No, if we subtract ln2, it becomes negative for x<0.
    # We want the penalty to be 0 at x=0.
    # So we should use:
    # dist = softplus(x) - softplus(0)  is bad for x<0.
    
    # Let's just use softplus(x) directly, but normalize the whole tfac so that tfac(T_opt)=1.
    # tfac_raw = exp(- (slope_l * softplus(T_opt - T) + slope_h * softplus(T - T_opt)))
    # At T=T_opt:
    # softplus(0) = ln2.
    # tfac_raw = exp(- (slope_l + slope_h) * ln2) = 2^(-(slope_l+slope_h)).
    # We want tfac(T_opt) = 1.
    # So tfac = tfac_raw / 2^(-(slope_l+slope_h)) = tfac_raw * 2^(slope_l+slope_h).
    # Or tfac = exp(- (slope_l * softplus(T_opt - T) + slope_h * softplus(T - T_opt)) + (slope_l + slope_h) * ln2)
    
    term_low  = T_slope_l * softplus(T_opt - temp)
    term_high = T_slope_h * softplus(temp - T_opt)
    
    # Normalization constant to make tfac(T_opt) = 1
    norm_const = (T_slope_l + T_slope_h) * np.log(2.0)
    
    tfac = np.exp(-(term_low + term_high) + norm_const)
    
    # pH Response: Gaussian
    pfac = np.exp(-0.5 * ((pH - pH_opt) / pH_width)**2)
    
    # Density Inhibition
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = 1.0 / (1.0 + np.exp(arg))
    
    return mu_max * sfac * tfac * pfac * dfac + mu_base
