import numpy as np

def equation(b, s, temp, pH, params):
    """
    Factorized Model for E. coli Growth Rate.
    
    Strategy:
    Instead of a single global polynomial, we model the growth rate as a product
    of four univariate response functions. This reflects biological reality where
    growth is limited by multiple independent factors (density, substrate, temp, pH).
    
    Model: mu = p[0] * f_b(b) * f_s(s) * f_t(temp) * f_p(pH)
    
    Each f_x is a smooth function controlled by 2.25 parameters (intercept, slope, center/scale).
    Since we have 10 parameters total:
    - p[0]: Global scale / intercept
    - p[1], p[2]: b response (linear-like or soft)
    - p[3], p[4]: s response (Monod-like saturation)
    - p[5], p[6]: temp response (Gaussian-like bell curve)
    - p[7], p[8]: pH response (Gaussian-like bell curve)
    - p[9]: Global offset or secondary interaction? 
             Let's use p[9] to modulate the temperature width or shift slightly, 
             or better, let's stick to 2 params per factor + 1 global = 9. 
             We have 10. Let's use p[9] as a global additive constant to handle baseline noise.
             
    Actually, to keep BFGS stable and avoid zero-division or extreme values:
    f_b(b) = p[1] * b + p[2]  (Linear, since b is density, likely positive effect)
    f_s(s) = 1 / (1 + exp(-p[3]*(s - p[4]))) (Logistic/Sigmoid for substrate saturation)
    f_t(temp) = exp(-0.5 * ((temp - p[5]) / p[6])^2) (Gaussian for optimal temp)
    f_p(pH) = exp(-0.5 * ((pH - p[7]) / p[8])^2) (Gaussian for optimal pH)
    
    Wait, BFGS starting from all ones might struggle with Gaussian centers if they are far from 1.
    The inputs are:
    b: [1, 5]
    s: [1, 5]
    temp: [1, 40]
    pH: [2, 12]
    
    If params start at 1:
    f_t(1) = exp(-0.5 * ((1-1)/1)^2) = 1. Good.
    f_t(40) = exp(-0.5 * (39)^2) ~ 0. Good.
    
    However, if the optimal temp is 37, p[5] needs to go from 1 to 37. That's a large step for BFGS.
    To help BFGS, we can normalize inputs inside the function.
    
    Let's normalize inputs to [0, 1] or [-1, 1] ranges.
    b_norm = (b - 1) / 4.0
    s_norm = (s - 1) / 4.0
    t_norm = (temp - 1) / 39.0
    p_norm = (pH - 2) / 10.0
    
    Now ranges are [0,1].
    Optimal temp ~ 37C -> t_norm ~ (37-1)/39 ~ 0.92.
    Optimal pH ~ 7 -> p_norm ~ (7-2)/10 ~ 0.5.
    
    Params start at 1.
    If we use Gaussians on normalized inputs:
    Center p[5] starts at 1. Range is [0,1]. So 1 is at the upper bound.
    This is risky.
    
    Alternative: Use a "Soft" polynomial or a neural-net-like single layer with fixed weights?
    No, we must use 10 fitted constants.
    
    Let's try a **Mixture of Experts** or **Gaussian Radial Basis Functions (RBF)** approach.
    With 10 params, we can't do many RBFs.
    
    Let's go back to the Factorized approach but with **Sigmoid** for temp and pH to allow monotonic or saturating behavior if the data is not strictly bell-shaped, or use a **Quadratic** for temp/pH but with normalized inputs.
    
    Revised Strategy: Normalized Quadratic Factorized Model.
    mu = p[0] * (p[1] + p[2]*b_n + p[3]*b_n^2) * (p[4] + p[5]*s_n + p[6]*s_n^2) * (p[7] + p[8]*t_n + p[9]*t_n^2) * ... wait, that's 10 params for 3 factors.
    
    Let's simplify:
    The current score is 29.4. The seed is likely simple.
    Inspiration 2 (28.9) and 3 (26.1) are close.
    The best observed child in the JSON is 1.39 (p67). That is a HUGE improvement.
    p67 used a "Generalized Additive Model (GAM)" with "piecewise-linear basis functions (hats)".
    
    Let's implement a **GAM with Hat Functions (B-splines of order 1)**.
    This is structurally different from global polynomials. It allows local flexibility.
    We have 10 params.
    We can allocate 2 "hat" functions per input?
    Inputs: b, s, temp, pH.
    
    If we use 2 hats per input, that's 8 hats. Plus 1 intercept + 1 global scale = 10.
    
    Hat function: h(x; c, w) = max(0, 1 - |x-c|/w).
    Smooth version: Use a Gaussian kernel? No, "hats" were mentioned as successful in p67.
    Let's use smooth bumps: exp(-((x-c)/w)^2).
    
    Allocation:
    1. Global Intercept/Scale: p[0]
    2. b: 2 Gaussians (centers c1, c2, widths w1, w2) -> 4 params? Too many.
    
    Let's stick to the **Factorized Model** but make it robust for BFGS.
    The key failure of BFGS with Gaussians is if the center is far from init.
    
    Let's try a **Logistic Product Model** (Monod-like for s, Logistic for temp/pH).
    
    mu = p[0] * 
         (1 / (1 + exp(-p[1]*(b - p[2])))) *   # Logistic for b (density)
         (1 / (1 + exp(-p[3]*(s - p[4])))) *   # Logistic for s (substrate)
         (1 / (1 + exp(-p[5]*(temp - p[6])))) * # Logistic for temp (optimal-ish)
         (1 / (1 + exp(-p[7]*(pH - p[8]))))     # Logistic for pH
    
    This uses 9 params. p[9] can be an additive constant.
    
    Init all ones:
    b ~ 1-5. Center 1. Slope 1.
    At b=1, logit(0) = 0.5.
    At b=5, logit(4) ~ 0.98.
    
    Temp ~ 1-40. Center 1. Slope 1.
    At t=1, 0.5.
    At t=40, logit(39) ~ 1.
    This forces a monotonic increase. If growth peaks at 37 and drops, this fails.
    
    To allow a peak, we need a "bell" shape.
    Product of two Logistcs? Or a Gaussian?
    
    Let's use **Gaussian Bumps** for Temp and pH, and **Logistic/Linear** for b and s.
    
    Params:
    p[0]: Global Scale
    p[1]: b Linear Coeff
    p[2]: s Logistic Slope
    p[3]: s Logistic Center
    p[4]: temp Gaussian Center
    p[5]: temp Gaussian Width
    p[6]: pH Gaussian Center
    p[7]: pH Gaussian Width
    p[8]: b Logistic Slope (to allow saturation)
    p[9]: b Logistic Center
    
    Total 10.
    
    Normalization is critical for BFGS.
    b_norm = (b - 1) / 4.0  # [0, 1]
    s_norm = (s - 1) / 4.0  # [0, 1]
    t_norm = (temp - 1) / 39.0 # [0, 1]
    p_norm = (pH - 2) / 10.0   # [0, 1]
    
    Model:
    f_b = 1 / (1 + exp(-p[8] * (b_norm - p[9])))
    f_s = 1 / (1 + exp(-p[2] * (s_norm - p[3])))
    f_t = exp(-0.5 * ((t_norm - p[4]) / p[5])^2)
    f_p = exp(-0.5 * ((p_norm - p[6]) / p[7])^2)
    
    mu = p[0] * f_b * f_s * f_t * f_p
    
    Init params = 1:
    p[0]=1, p[8]=1, p[9]=1.
    f_b(0) = sigmoid(-1) ~ 0.27.
    f_b(1) = sigmoid(0) = 0.5.
    
    f_s(0) = sigmoid(-1) ~ 0.27.
    f_s(1) = sigmoid(0) = 0.5.
    
    f_t(0) = exp(-0.5 * (-1)^2) = exp(-0.5) ~ 0.6.
    f_t(1) = exp(-0.5 * (0)^2) = 1.
    (Optimal at t_norm=1, i.e., 40C. Real optimal is ~37C, t_norm~0.92. Close enough for BFGS to move.)
    
    f_p(0) = exp(-0.5 * (-1)^2) ~ 0.6.
    f_p(1) = 1.
    (Optimal at pH_norm=1, i.e., 12. Real optimal ~7, p_norm=0.5. BFGS needs to move center from 1 to 0.5. Feasible.)
    
    This structure is convex-ish in the log domain? No, product of Gaussians is not convex.
    But BFGS is local. Starting from 1, the landscape should be navigable if we don't have extreme vanishing gradients.
    
    To ensure non-negative and smooth, this is good.
    
    One risk: If p[5] (width) becomes very small, the Gaussian becomes a spike, causing numerical issues.
    We can clip the width: w = max(p[5], 0.05).
    
    Let's refine the code.
    """
    p = params
    
    # Normalize inputs to [0, 1]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    # Clip normalized inputs to prevent extreme extrapolation issues in exponentials
    # Although inputs are in range, slight floating point errors or OOD might cause issues.
    # The task says "Inputs in training range", so clipping is a safety net.
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    t_n = np.clip(t_n, 0.0, 1.0)
    ph_n = np.clip(ph_n, 0.0, 1.0)
    
    # Factor 1: Density (b) - Logistic
    # p[8]: slope, p[9]: center
    slope_b = p[8]
    center_b = p[9]
    # Prevent overflow in exp
    z_b = -slope_b * (b_n - center_b)
    z_b = np.clip(z_b, -50, 50)
    f_b = 1.0 / (1.0 + np.exp(z_b))
    
    # Factor 2: Substrate (s) - Logistic
    # p[2]: slope, p[3]: center
    slope_s = p[2]
    center_s = p[3]
    z_s = -slope_s * (s_n - center_s)
    z_s = np.clip(z_s, -50, 50)
    f_s = 1.0 / (1.0 + np.exp(z_s))
    
    # Factor 3: Temperature (temp) - Gaussian
    # p[4]: center, p[5]: width
    center_t = p[4]
    width_t = np.abs(p[5]) + 0.01  # Ensure positive width, avoid division by zero
    diff_t = t_n - center_t
    f_t = np.exp(-0.5 * (diff_t / width_t) ** 2)
    
    # Factor 4: pH - Gaussian
    # p[6]: center, p[7]: width
    center_p = p[6]
    width_p = np.abs(p[7]) + 0.01
    diff_p = ph_n - center_p
    f_p = np.exp(-0.5 * (diff_p / width_p) ** 2)
    
    # Global Scale
    scale = p[0]
    
    return scale * f_b * f_s * f_t * f_p
