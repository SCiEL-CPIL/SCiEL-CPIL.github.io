import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Mixture of Experts (MoE) structure.
    
    The model uses a soft-gated convex combination of three linear expert functions
    to capture complex, non-uniform response surfaces. The gates are driven by
    temperature and pH using sigmoid activations to allow sharp transitions.
    Substrate follows Monod kinetics, and density inhibition is multiplicative.
    
    Parameters (10 total):
    p0: global scale
    p1: Monod Ks
    p2: gate temperature center
    p3: gate temperature width
    p4: gate pH center
    p5: gate pH width
    p6: expert 1 weight (low temp)
    p7: expert 2 weight (mid temp)
    p8: expert 3 weight (high temp)
    p9: baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive width parameters for sigmoid gates
    width_temp = np.abs(p3) + 1e-6
    width_ph = np.abs(p5) + 1e-6
    
    # Monod substrate term
    monod = s / (s + np.abs(p1) + 1e-6)
    
    # Density inhibition
    density = 1.0 / (1.0 + b / (np.abs(p9) + 1.0))
    
    # Gate functions: sigmoid-based
    # Gate 1: active at low temperature
    gate1 = 1.0 / (1.0 + np.exp((temp - p2) / width_temp))
    # Gate 2: active at mid temperature (bell-shaped via difference of sigmoids)
    gate2 = np.exp(-((temp - p2) / width_temp)**2)
    # Gate 3: active at high temperature
    gate3 = 1.0 / (1.0 + np.exp(-(temp - p2) / width_temp))
    
    # Normalize gates to sum to 1
    gate_sum = gate1 + gate2 + gate3 + 1e-10
    gate1 = gate1 / gate_sum
    gate2 = gate2 / gate_sum
    gate3 = gate3 / gate_sum
    
    # Expert functions: linear in pH and density, with different slopes
    # Expert 1: low temp expert
    expert1 = p6 * (1.0 + (pH - p4) / width_ph)
    # Expert 2: mid temp expert
    expert2 = p7 * (1.0 + 0.5 * (pH - p4) / width_ph)
    # Expert 3: high temp expert (thermal death)
    expert3 = p8 * (1.0 - 0.5 * (pH - p4) / width_ph)
    
    # Combine experts with gates
    combined = gate1 * expert1 + gate2 * expert2 + gate3 * expert3
    
    # Final output
    result = p0 * monod * density * combined + p9
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
