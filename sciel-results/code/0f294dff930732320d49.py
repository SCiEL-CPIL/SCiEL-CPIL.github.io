import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a biologically-inspired Monod-type kinetic model.
    
    This model assumes:
    1. Growth is limited by nutrient availability (b, s) following Monod saturation kinetics.
    2. Temperature and pH affect growth via a Gaussian-like response centered at optimal values.
    3. The maximum growth rate is a learned parameter.
    
    Parameters:
    p[0]: Vmax (maximum growth rate scale)
    p[1]: Ks_b (half-saturation constant for b)
    p[2]: Ks_s (half-saturation constant for s)
    p[3]: T_opt (optimal temperature)
    p[4]: T_width (temperature sensitivity width)
    p[5]: pH_opt (optimal pH)
    p[6]: pH_width (pH sensitivity width)
    p[7]: b_s_coupling (interaction strength between b and s saturation)
    p[8]: T_pH_coupling (interaction strength between T and pH)
    p[9]: base_offset (baseline offset to handle non-zero minimal growth)
    """
    Vmax = params[0]
    Ks_b = np.abs(params[1]) + 1e-6
    Ks_s = np.abs(params[2]) + 1e-6
    T_opt = params[3]
    T_width = np.abs(params[4]) + 1e-6
    pH_opt = params[5]
    pH_width = np.abs(params[6]) + 1e-6
    b_s_coupling = params[7]
    T_pH_coupling = params[8]
    base_offset = params[9]

    # Monod saturation for b and s
    # Using (b / (Ks_b + b)) which is 0 at b=0 and 1 at b->inf
    sat_b = b / (Ks_b + b)
    sat_s = s / (Ks_s + s)
    
    # Combine saturation effects. 
    # Simple product: min(sat_b, sat_s) is biologically standard for co-limitation, 
    # but product is smoother and differentiable. 
    # We can also add a coupling term to allow for synergy or antagonism.
    # Let's use a weighted geometric mean or product with a coupling exponent.
    # For simplicity and differentiability, let's use:
    # nutrient_factor = sat_b * sat_s * (1 + b_s_coupling * (1 - sat_b * sat_s))
    # This allows the coupling to modify the shape.
    # Alternatively, a simpler multiplicative model with a shared constant.
    # Let's stick to a flexible multiplicative form:
    # nutrient_factor = (sat_b ** (1 + b_s_coupling)) * (sat_s ** (1 - b_s_coupling))
    # This is smooth and allows trade-offs.
    
    # Let's try a simpler, robust form:
    # nutrient_factor = (b / (Ks_b + b)) * (s / (Ks_s + s))
    # The coupling parameter can scale the interaction.
    
    # Let's use a more flexible form for nutrient limitation:
    # F_nutrient = (b * s) / (Ks_b * s + Ks_s * b + Ks_b * Ks_s)  # This is a common form for two substrates
    # But let's stick to standard Monod product for stability, and use coupling to adjust Vmax or shape.
    
    # Let's use:
    # nutrient_factor = sat_b * sat_s
    # And allow b_s_coupling to adjust the effective Ks or the shape.
    # Actually, let's just use the product and let the other params do the work.
    # To utilize b_s_coupling, let's modulate the saturation:
    # sat_b_eff = sat_b ** (1 + b_s_coupling)
    # sat_s_eff = sat_s ** (1 - b_s_coupling)
    # This is smooth and bounded.
    
    sat_b_eff = np.power(sat_b, 1.0 + b_s_coupling)
    sat_s_eff = np.power(sat_s, 1.0 - b_s_coupling)
    
    nutrient_factor = sat_b_eff * sat_s_eff

    # Temperature and pH response
    # Gaussian response: exp(-((x - opt) / width)^2)
    # This is smooth and positive.
    temp_factor = np.exp(-((temp - T_opt) / T_width) ** 2)
    ph_factor = np.exp(-((pH - pH_opt) / pH_width) ** 2)
    
    # Combine T and pH with a coupling term
    # The coupling can shift the optimal point or modify the width.
    # Let's allow the coupling to adjust the exponent or the center.
    # A simple way: effective_T_opt = T_opt + T_pH_coupling * (pH - pH_opt)
    # This allows the optimal temperature to shift with pH.
    eff_T_opt = T_opt + T_pH_coupling * (pH - pH_opt)
    temp_factor = np.exp(-((temp - eff_T_opt) / T_width) ** 2)
    
    # Final growth rate
    growth = Vmax * nutrient_factor * temp_factor * ph_factor + base_offset
    
    # Ensure non-negative growth? The problem says target range is [-0.00277, 1.553], so negative is possible.
    # But biologically growth rate is >= 0. The negative values in target might be due to measurement noise or net growth (growth - death).
    # Let's not clamp to 0 to allow fitting the negative tail if it exists.
    
    return growth
