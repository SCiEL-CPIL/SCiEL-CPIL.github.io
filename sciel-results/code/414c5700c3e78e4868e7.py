"""E. coli growth rate model using a Mixture of Experts with quadratic polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Mixture of Experts (MoE) approach.
    
    This model uses three expert networks, each a quadratic polynomial, combined
    via soft gates determined by temperature and pH. This allows the model to
    capture distinct regimes of growth behavior (e.g., optimal vs. stress conditions)
    without the rigidity of a single global polynomial.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Parameters (10):
    p0: Gate temperature center
    p1: Gate temperature width
    p2: Gate pH center
    p3: Gate pH width
    p4: Expert 1 bias
    p5: Expert 2 bias
    p6: Expert 3 bias
    p7: Expert 1 temp coefficient
    p8: Expert 2 temp coefficient
    p9: Expert 3 temp coefficient
    """
    # Track elapsed time for safety
    start_time = time.time()
    deadline = start_time + 115.0  # 5-second safety margin before 120s limit

    # Normalize inputs
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 3.0) / 2.0
    temp_norm = (temp - 20.0) / 10.0
    ph_norm = (pH - 7.0) / 2.0

    # Unpack parameters
    gate_temp_center = params[0]
    gate_temp_width = np.abs(params[1]) + 0.1  # Ensure positive width
    gate_ph_center = params[2]
    gate_ph_width = np.abs(params[3]) + 0.1  # Ensure positive width

    exp1_bias = params[4]
    exp2_bias = params[5]
    exp3_bias = params[6]
    exp1_temp_coeff = params[7]
    exp2_temp_coeff = params[8]
    exp3_temp_coeff = params[9]

    # Compute gate logits based on temperature and pH
    # Expert 1: Low temperature regime
    # Expert 2: Optimal temperature regime
    # Expert 3: High temperature regime
    
    # Use Gaussian-like gates
    temp_gate = np.exp(-((temp_norm - gate_temp_center) ** 2) / (2 * gate_temp_width ** 2))
    ph_gate = np.exp(-((ph_norm - gate_ph_center) ** 2) / (2 * gate_ph_width ** 2))

    # Combine gates: 
    # Expert 1 weight: high when temp is low (negative temp_norm)
    # Expert 2 weight: high when temp is near optimal (near 0)
    # Expert 3 weight: high when temp is high (positive temp_norm)
    
    w1 = np.exp(-np.maximum(0, temp_norm) ** 2 / 2)  # Low temp expert
    w2 = temp_gate  # Optimal temp expert
    w3 = np.exp(-np.maximum(0, -temp_norm) ** 2 / 2)  # High temp expert

    # Normalize weights to sum to 1
    w_sum = w1 + w2 + w3
    w1 = w1 / (w_sum + 1e-10)
    w2 = w2 / (w_sum + 1e-10)
    w3 = w3 / (w_sum + 1e-10)

    # Check time
    if time.time() > deadline:
        # Return a simple fallback prediction
        return np.nan_to_num(exp1_bias + exp2_bias + exp3_bias, nan=0.0, posinf=1e6, neginf=-1e6)

    # Compute expert predictions (simplified quadratic forms)
    # Expert 1: Focus on low temp, linear in b and s
    pred1 = exp1_bias + exp1_temp_coeff * temp_norm + 0.5 * b_norm + 0.5 * s_norm

    # Expert 2: Focus on optimal conditions, includes interaction
    pred2 = exp2_bias + exp2_temp_coeff * temp_norm + ph_gate * (b_norm * s_norm) + 0.3 * ph_norm

    # Expert 3: Focus on high temp stress, quadratic in temp
    pred3 = exp3_bias + exp3_temp_coeff * temp_norm - 0.5 * temp_norm ** 2 + 0.2 * ph_norm

    # Combine experts with soft gates
    pred = w1 * pred1 + w2 * pred2 + w3 * pred3

    # Final time check
    if time.time() > deadline:
        return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure the output is finite
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
