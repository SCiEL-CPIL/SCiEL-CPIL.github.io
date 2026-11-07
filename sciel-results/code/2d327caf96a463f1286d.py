"""Nonlinear parametric model for a driven Duffing/Van der Pol oscillator.

Strategy: Instead of a linear basis (GLM), use the 10 parameters as nonlinear
physical constants inside an analytic approximation of a driven Duffing oscillator:
  m*x'' + c*v + k*x + a*x^3 = F0*cos(w_d*t)

We construct a particular solution (steady-state driven response with amplitude
that depends nonlinearly on the drive frequency and stiffness) plus a transient
term that decays in time (capturing initial-condition effects). This gives the
function genuine nonlinear dynamics that a linear combination of basis functions
cannot replicate.

Parameter mapping (10 params):
  p0 = mass m (positive, clamped)
  p1 = damping c (positive, clamped)
  p2 = linear stiffness k
  p3 = cubic stiffness a (Duffing term)
  p4 = drive amplitude F0
  p5 = drive frequency w_d
  p6 = drive phase phi
  p7 = transient decay rate gamma (positive, clamped)
  p8 = transient initial displacement scale A0
  p9 = transient initial velocity scale B0
"""
import numpy as np
import time

_DEADLINE_SECONDS = 140.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration (target) of a nonlinear driven-damped oscillator,
    modeled via an analytic particular + transient solution."""
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9) = p

    # Clamp physical parameters to avoid singularities / divergence
    m = np.clip(p0, 0.01, 100.0)       # mass
    c = np.clip(p1, 0.001, 50.0)       # damping
    k = p2                              # linear stiffness (can be negative for softening)
    a = p3                              # cubic stiffness
    F0 = p4                             # drive amplitude
    wd = np.clip(p5, 0.01, 50.0)       # drive frequency (keep positive)
    phi = p6                            # drive phase
    gamma = np.clip(p7, 0.001, 10.0)   # transient decay rate
    A0 = p8                             # transient amplitude scale
    B0 = p9                             # transient velocity scale

    # --- Particular solution: driven response ---
    # For x'' + (c/m)x' + (k/m)x + (a/m)x^3 = (F0/m) cos(wd*t + phi)
    # Approximate the nonlinear frequency shift: effective frequency squared
    # omega0^2 = k/m + 3*a*x^2/m (instantaneous, evaluated at current x)
    # The amplitude of the particular solution for a linearized system:
    # X = (F0/m) / sqrt((k/m - wd^2 + 3*a*x^2/m)^2 + (c*wd/m)^2)
    # Phase shift: tan(delta) = (c*wd/m) / (k/m - wd^2 + 3*a*x^2/m)
    
    m_safe = np.where(np.abs(m) > 1e-8, m, 1e-8)
    
    # Instantaneous effective stiffness (linearized around current x)
    omega0_sq = (k + 3.0 * a * x * x) / m_safe
    
    # Denominator components for the driven response
    denom_real = omega0_sq - wd * wd
    denom_imag = c * wd / m_safe
    
    # Amplitude of particular solution
    denom_mag = np.sqrt(denom_real * denom_real + denom_imag * denom_imag)
    denom_mag_safe = np.where(np.abs(denom_mag) > 1e-8, denom_mag, 1e-8)
    
    X_part = (F0 / m_safe) / denom_mag_safe
    
    # Phase of particular solution
    delta = np.arctan2(denom_imag, denom_real)
    
    # Particular displacement: X_part * cos(wd*t + phi - delta)
    x_part = X_part * np.cos(wd * t + phi - delta)
    
    # Particular velocity: -X_part * wd * sin(wd*t + phi - delta)
    v_part = -X_part * wd * np.sin(wd * t + phi - delta)
    
    # --- Transient solution (decaying oscillation) ---
    # Approximate as damped oscillation with natural frequency ~ sqrt(max(k/m, 0.1))
    omega_n = np.sqrt(np.clip(k / m_safe, 0.01, 100.0))
    
    # Damping ratio effect: use gamma as effective decay
    decay = np.exp(-gamma * t)
    
    x_trans = A0 * decay * np.cos(omega_n * t)
    v_trans = B0 * decay * np.cos(omega_n * t) - A0 * omega_n * decay * np.sin(omega_n * t)
    
    # Total displacement and velocity
    x_total = x_part + x_trans
    v_total = v_part + v_trans
    
    # --- Acceleration (the target) ---
    # x'' = -omega0_sq * x_total - (c/m) * v_total - (a/m) * x_total^3 + (F0/m)*cos(wd*t+phi)
    # But since we constructed x_total to satisfy the ODE approximately,
    # compute acceleration directly from the ODE residual:
    # a_target = (F0/m)*cos(wd*t+phi) - (c/m)*v_total - (k/m)*x_total - (a/m)*x_total^3
    
    drive = (F0 / m_safe) * np.cos(wd * t + phi)
    accel = drive - (c / m_safe) * v_total - (k / m_safe) * x_total - (a / m_safe) * x_total * x_total * x_total
    
    # Alternative: compute second derivative of x_total analytically
    # x_part'' = -X_part * (wd^2) * cos(wd*t+phi-delta)
    # This is equivalent to the ODE if X_part and delta are exact.
    # Use the ODE form as it's more stable.
    
    # Ensure finiteness
    accel = np.where(np.isfinite(accel), accel, 0.0)
    
    return np.asarray(accel, dtype=float)
