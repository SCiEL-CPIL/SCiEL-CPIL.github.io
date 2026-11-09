import numpy as np
import time

_DEADLINE = time.time() + 135.0

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator: a = f(x, v, t)
    
    Uses a modular multiplicative structure:
      a = F_stiffness(x) * x + F_damping(v) * v + F_forcing(t)
    
    where each F is a bounded, smooth function of its argument.
    This mirrors the physics: restoring force proportional to displacement,
    drag force proportional to velocity, and external periodic drive.
    
    Parameters (10 total):
    p0: bias
    p1: linear stiffness coefficient
    p2: cubic stiffness coefficient (Duffing)
    p3: stiffness saturation scale
    p4: linear damping coefficient
    p5: quadratic damping coefficient
    p6: damping saturation scale
    p7: forcing amplitude (sin component)
    p8: forcing amplitude (cos component)
    p9: forcing frequency multiplier
    """
    # Normalize inputs for better conditioning
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Stiffness force: bounded restoring force
    # F_stiff(x) = p1 + p2 * tanh(x_n / p3) * x_n^2 / max(|p3|, 0.1)
    p3_safe = np.where(np.abs(p3) < 0.1, 0.1 * np.sign(p3), p3)
    p3_safe = np.where(p3_safe == 0, 0.1, p3_safe)
    x2 = x_n * x_n
    stiff_sat = np.tanh(x_n / np.abs(p3_safe))
    F_stiff = p1 + p2 * stiff_sat * x2 / np.abs(p3_safe)
    
    # Damping force: bounded drag
    # F_damp(v) = p4 + p5 * tanh(v_n / p6) * |v_n| / max(|p6|, 0.1)
    p6_safe = np.where(np.abs(p6) < 0.1, 0.1 * np.sign(p6), p6)
    p6_safe = np.where(p6_safe == 0, 0.1, p6_safe)
    v_abs = np.abs(v_n)
    damp_sat = np.tanh(v_n / np.abs(p6_safe))
    F_damp = p4 + p5 * damp_sat * v_abs / np.abs(p6_safe)
    
    # Forcing: periodic drive
    freq = np.where(np.abs(p9) < 0.1, 0.1 * np.sign(p9), p9)
    freq = np.where(freq == 0, 0.1, freq)
    F_force = p7 * np.sin(freq * t_n) + p8 * np.cos(freq * t_n)
    
    # Combine: a = F_stiff * x_n + F_damp * v_n + F_force + p0
    res = p0 + F_stiff * x_n + F_damp * v_n + F_force
    
    return res
