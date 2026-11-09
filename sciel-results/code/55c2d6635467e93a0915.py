"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Mixture of Experts (MoE) structure.
    
    Three linear experts are combined via soft gates driven by temperature and pH.
    Substrate follows Monod kinetics, and density provides logistic inhibition.
    
    Inputs:
        b: population density (range ~[1, 5])
        s: substrate concentration (range ~[1, 40])
        temp: temperature (range ~[1, 40])
        pH: pH level (range ~[1, 12])
    
    Params (10):
        p0: gate temperature center
        p1: gate temperature sharpness
        p2: gate pH center
        p3: gate pH sharpness
        p4: Monod half-saturation constant for substrate
        p5: expert 1 intercept (low-temp regime)
        p6: expert 2 intercept (mid-temp regime)
        p7: expert 3 intercept (high-temp/death regime)
        p8: density inhibition strength
        p9: expert 2 slope on substrate (relative weighting)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Clamp params to safe ranges to avoid numerical issues
    p0 = np.clip(p0, 0.0, 50.0)
    p1 = np.clip(p1, 0.1, 10.0)
    p2 = np.clip(p2, 0.0, 14.0)
    p3 = np.clip(p3, 0.1, 10.0)
    p4 = np.clip(p4, 0.1, 50.0)
    p8 = np.clip(p8, 0.0, 5.0)
    p9 = np.clip(p9, 0.0, 3.0)

    # Normalize inputs for stable computation
    t = temp.astype(np.float64)
    ph = pH.astype(np.float64)
    sub = s.astype(np.float64)
    dens = b.astype(np.float64)

    # --- Soft gates based on temperature and pH ---
    # Gate 1: active at low/moderate temps (below optimum)
    # Gate 2: active near optimal temp
    # Gate 3: active at high temps (death zone)
    
    # Temperature gating: two sigmoid-like gates
    # g1 = sigmoid((p0 - t) * p1)  -> high when t < p0
    # g3 = sigmoid((t - p0) * p1)  -> high when t > p0
    # g2 = 1 - g1 - g3 (clamped to be non-negative, then renormalize)
    
    # Use stable sigmoid
    def stable_sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    g1_raw = stable_sigmoid((p0 - t) * p1)
    g3_raw = stable_sigmoid((t - p0) * p1)
    g2_raw = 1.0 - g1_raw - g3_raw
    g2_raw = np.maximum(g2_raw, 0.0)

    # pH gating: multiplicative factor that peaks near optimal pH
    # Gaussian-like in pH: exp(-((ph - p2)^2) / (2 * sigma^2)) where sigma = 1/p3
    ph_sigma = 1.0 / p3
    ph_gate = np.exp(-0.5 * ((ph - p2) / ph_sigma) ** 2)
    # Normalize ph_gate to [0, 1] scale by dividing by max possible (which is 1 at ph=p2)
    # It's already in [0, 1]

    # Combine temp and pH gates
    # Each expert gets: temp_gate * ph_gate
    w1 = g1_raw * ph_gate
    w2 = g2_raw * ph_gate
    w3 = g3_raw * ph_gate

    # Renormalize weights to sum to 1 (convex combination)
    w_sum = w1 + w2 + w3
    w_sum = np.maximum(w_sum, 1e-12)
    w1 = w1 / w_sum
    w2 = w2 / w_sum
    w3 = w3 / w_sum

    # --- Expert functions ---
    # Each expert is a linear function of substrate (Monod-scaled) and density
    
    # Monod substrate response: s / (p4 + s)
    monod = sub / (p4 + sub)

    # Density inhibition: 1 / (1 + p8 * dens) or (1 - dens/b_max)
    # Use logistic inhibition: 1 / (1 + p8 * dens)
    dens_inh = 1.0 / (1.0 + p8 * dens)

    # Expert 1: low-temp, lower growth
    expert1 = p5 * monod * dens_inh
    # Expert 2: optimal temp, highest growth
    expert2 = p6 * monod * dens_inh + p9 * monod
    # Expert 3: high-temp, reduced/growth death
    expert3 = p7 * monod * dens_inh

    # --- Combine experts ---
    result = w1 * expert1 + w2 * expert2 + w3 * expert3

    return result
# EVOLVE-BLOCK-END
