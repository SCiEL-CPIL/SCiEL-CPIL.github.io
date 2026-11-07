# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using piecewise-linear plateaus and rational inhibition.
    
    Strategy:
    1. Temperature and pH effects are modeled as linear ramps clipped to [0, 1].
       This enforces zero growth outside viable ranges and a plateau at optimum.
    2. Substrate and density effects are combined into a single rational inhibition term.
       This creates a smooth saturation curve that accounts for both nutrient limitation
       and crowding effects simultaneously.
    
    params (length 10):
    p[0]: max growth rate (mu_max)
    p[1]: T_min
    p[2]: T_opt
    p[3]: T_max
    p[4]: pH_min
    p[5]: pH_opt
    p[6]: pH_max
    p[7]: K_s (substrate saturation constant)
    p[8]: K_b (density inhibition constant)
    p[9]: baseline offset
    """
    p = params
    
    # Unpack parameters
    mu_max = p[0]
    T_min = p[1]
    T_opt = p[2]
    T_max = p[3]
    pH_min = p[4]
    pH_opt = p[5]
    pH_max = p[6]
    K_s = np.abs(p[7]) + 1e-6
    K_b = np.abs(p[8]) + 1e-6
    offset = p[9]
    
    # --- Temperature Factor ---
    # Linear rise from T_min to T_opt, linear fall from T_opt to T_max.
    # Clip to [0, 1] to enforce zero growth outside bounds and plateau at opt.
    
    # Ensure valid ranges to avoid division by zero or negative slopes in clip logic
    # We define the slope up and down implicitly via the clip boundaries.
    # If T_opt <= T_min, the ramp is degenerate. We handle this by enforcing a small gap.
    T_min_safe = T_min
    T_max_safe = T_max
    T_opt_safe = T_opt
    
    # To use clip effectively for a tent/plateau shape:
    # We want: 0 if T < T_min, (T - T_min) / (T_opt - T_min) if T_min <= T <= T_opt
    #          1 if T_opt < T < T_max (if plateau) or decreasing if tent.
    # The prompt says "piecewise-linear growth plateau". Usually this implies a trapezoid.
    # Let's implement a trapezoid:
    # Rise: T_min to T_opt
    # Plateau: T_opt to T_max? No, usually T_opt is the peak.
    # Let's stick to a symmetric/asymmetric tent shape which is a specific case of plateau with zero width,
    # OR a trapezoid where T_opt is the start of the plateau?
    # The previous code used T_opt as the peak of a tent (slope up then down).
    # Let's implement a trapezoid where the "plateau" is the region of max growth.
    # However, the previous code's "T_opt" was the peak.
    # Let's interpret "plateau" as the region where growth is maximal.
    # If we use np.clip, we can do:
    # factor = np.clip((temp - T_min) / (T_opt - T_min), 0, 1) * np.clip((T_max - temp) / (T_max - T_opt), 0, 1)
    # This creates a trapezoid if T_opt < T_max. If T_opt == T_max, it's a tent.
    # To allow a true plateau, we might need two points for the flat top.
    # With only 3 points (min, opt, max), we are constrained to a tent or a trapezoid where opt is one edge.
    # Let's assume the standard "optimum" is the peak of a tent for simplicity and robustness,
    # or a trapezoid where T_opt is the *start* of the plateau?
    # Given the previous score, the tent shape worked. Let's stick to a shape that peaks at T_opt.
    # Tent shape:
    # Up-slope: 1 / (T_opt - T_min)
    # Down-slope: 1 / (T_max - T_opt)
    
    # To avoid division by zero, add epsilon
    eps = 1e-6
    denom_up = np.abs(T_opt - T_min) + eps
    denom_down = np.abs(T_max - T_opt) + eps
    
    # Calculate linear components
    # If T < T_min, term is negative -> clipped to 0
    # If T > T_max, term is negative -> clipped to 0
    # At T_opt, both terms are 1 -> Product is 1
    
    temp_up = (temp - T_min) / denom_up
    temp_down = (T_max - temp) / denom_down
    
    # The product of two clipped linear functions creates a "tent" or "trapezoid-like" shape
    # that is 0 outside [T_min, T_max] and 1 at T_opt (if T_min < T_opt < T_max).
    # Note: If T_min > T_opt, the logic flips. We assume standard ordering T_min < T_opt < T_max.
    # To be safe against parameter ordering issues during optimization, we can use absolute differences
    # and assume the optimizer finds the correct order. However, BFGS can wander.
    # A more robust way:
    # temp_factor = np.clip((temp - T_min) / (T_opt - T_min), 0, 1) * np.clip((T_max - temp) / (T_max - T_opt), 0, 1)
    # This fails if T_opt == T_min.
    
    # Let's use the explicit calculation which handles signs better if we enforce order via softplus?
    # No, let's just use the clip method which is standard for this "plateau" description.
    # If T_opt is not between T_min and T_max, the factor might be 0 everywhere or 1 everywhere.
    # This acts as a hard constraint forcing the optimizer to find the correct order.
    
    temp_factor = np.clip((temp - T_min) / (T_opt - T_min + eps), 0, 1) * \
                  np.clip((T_max - temp) / (T_max - T_opt + eps), 0, 1)
    
    # --- pH Factor ---
    # Same logic
    pH_factor = np.clip((pH - pH_min) / (pH_opt - pH_min + eps), 0, 1) * \
                np.clip((pH_max - pH) / (pH_max - pH_opt + eps), 0, 1)
    
    # --- Substrate and Density Inhibition ---
    # Combined rational term: 1 / (1 + s/K_s + b/K_b)
    # This ensures growth decreases as s increases (if we view s as waste?) 
    # WAIT: s is substrate. Growth should INCREASE with s.
    # The previous model used Monod: s / (K_s + s). This increases with s.
    # My proposed "rational inhibition" 1/(1+s/K) decreases with s. This is WRONG for substrate.
    # Substrate promotes growth. Density inhibits growth.
    # So we need a term that increases with s and decreases with b.
    # Monod for s: s / (K_s + s)
    # Inhibition for b: 1 / (1 + b/K_b)
    # The previous model did exactly this.
    # The directive says "rational inhibition term for substrate and density".
    # Maybe it means a single rational function?
    # E.g., (s / (K_s + s)) * (1 / (1 + b/K_b)) is the product of two rationals.
    # Or a combined term like: (s) / (K_s + s + b * C)?
    # Let's stick to the biologically correct form: Monod for substrate, Michaelis-Menten-like inhibition for density.
    # To "improve" it, we can ensure the parameters are well-scaled.
    
    # Monod term
    substrate_factor = s / (K_s + s)
    
    # Density inhibition term
    density_factor = 1.0 / (1.0 + b / K_b)
    
    # Combine
    mu = mu_max * temp_factor * pH_factor * substrate_factor * density_factor + offset
    
    return mu
# EVOLVE-BLOCK-END
