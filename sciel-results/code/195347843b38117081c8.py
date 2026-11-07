import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # 10s safety margin before 150s kill

def equation(b, s, temp, pH, params):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    p = params
    
    # Normalization
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 20.0) / 20.0
    ph_n = (pH - 2.0) / 10.0
    
    # Structural change: Use a more robust, physically motivated form.
    # Instead of a generic quadratic density term, use a specific inhibition term
    # and ensure all rates/widths are positive via exp.
    
    # Monod term for substrate
    # p[1] is log(Ks). Exp ensures positive Ks.
    # p[9] is Hill coefficient. Use abs + small epsilon to ensure positive.
    ks = np.exp(p[1]) + 1e-8
    hill = np.abs(p[9]) + 0.1
    
    # Safe Monod calculation: s^hill / (ks^hill + s^hill)
    # To avoid overflow/underflow with high hill:
    # If s_n is 0, monod is 0.
    # We can compute 1 / (1 + (ks/s_n)^hill) for s_n > 0.
    # But s_n can be close to 0.
    # Let's stick to the direct form but clip s_n to avoid 0**0 issues if hill is small?
    # Actually, s_n is in [0.2, 1.0] based on [1, 5] range? No, [1, 4.999] -> s_n in [0.2, 1].
    # So s_n is never 0.
    # s_n**hill is safe.
    s_pow = s_n ** hill
    ks_pow = ks ** hill
    monod = s_pow / (ks_pow + s_pow)
    
    # Gaussian for temp
    # p[2] is optimal t_n
    # p[3] is log(width). Exp ensures positive width.
    t_opt = p[2]
    t_width = np.exp(p[3]) + 1e-8
    temp_factor = np.exp(-0.5 * ((t_n - t_opt) / t_width)**2)
    
    # Gaussian for pH
    # p[4] is optimal ph_n
    # p[5] is log(width). Exp ensures positive width.
    ph_opt = p[4]
    ph_width = np.exp(p[5]) + 1e-8
    ph_factor = np.exp(-0.5 * ((ph_n - ph_opt) / ph_width)**2)
    
    # Density/Biomass term
    # Structural change: Use a specific inhibition form instead of generic quadratic.
    # 1 / (1 + (b_n / Kb)^2)
    # p[6] is log(Kb). Exp ensures positive Kb.
    # This term is in [0, 1].
    # If we want to allow activation at low density, we might need a different form,
    # but inhibition is standard for growth rate saturation.
    # Let's try: 1 / (1 + exp(p[7]) * b_n) ? No, let's stick to the inhibition from inspiration 2.
    kb = np.exp(p[6]) + 1e-8
    dens_term = 1.0 / (1.0 + (b_n / kb)**2)
    
    # Global scale
    # p[8] is log(max_rate). Exp ensures positive.
    # p[9] was used for Hill. We have 10 params: p[0]..p[9].
    # Wait, I used p[9] for Hill.
    # Let's map:
    # p[0]: unused? Or bias?
    # Let's use p[0] as log(max_rate).
    # p[1]: log(Ks)
    # p[2]: t_opt
    # p[3]: log(t_width)
    # p[4]: ph_opt
    # p[5]: log(ph_width)
    # p[6]: log(Kb)
    # p[7]: unused?
    # p[8]: unused?
    # p[9]: Hill
    
    # Let's re-assign to use all 10 meaningfully if possible, or just use the ones needed.
    # Current mapping:
    # p[0]: log(max_rate)
    # p[1]: log(Ks)
    # p[2]: t_opt
    # p[3]: log(t_width)
    # p[4]: ph_opt
    # p[5]: log(ph_width)
    # p[6]: log(Kb)
    # p[7]: bias (additive)
    # p[8]: unused (set to 0)
    # p[9]: Hill
    
    max_rate = np.exp(p[0]) + 1e-8
    bias = p[7]
    
    result = max_rate * monod * temp_factor * ph_factor * dens_term + bias
    
    return result
