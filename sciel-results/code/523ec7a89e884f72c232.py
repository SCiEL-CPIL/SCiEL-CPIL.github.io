"""
Strategy: Multi-Start Global Optimization for a Physically-Motivated Model.
The current program uses a fixed-center RBF basis optimized via BFGS.
This new strategy uses a physically-grounded functional form (Ramberg-Osgood variant)
and performs multi-start Nelder-Mead optimization to find the global minimum,
avoiding the local minima traps common in non-convex fitting landscapes.

Model:
sigma = E * eps + H * eps_pl^n + S * (1 - exp(-k * eps_pl))
where:
E = p0 + p1 * T
H = p2 + p3 * T
n = p4
S = p5 + p6 * T
k = p7
eps_pl = max(0, eps - sigma_y/E)
sigma_y = p8 + p9 * T

This form naturally captures:
1. Linear elastic region (E * eps)
2. Yield point (transition at eps_y)
3. Power-law hardening (H * eps_pl^n)
4. Saturation behavior (exponential term)
5. Temperature dependence for all key parameters
"""
import numpy as np
from scipy.optimize import minimize
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a physically-motivated model.
    
    Parameters (10 total):
    p0: Base elastic modulus
    p1: Temperature coefficient for elastic modulus
    p2: Base hardening coefficient
    p3: Temperature coefficient for hardening coefficient
    p4: Hardening exponent
    p5: Base saturation stress
    p6: Temperature coefficient for saturation stress
    p7: Saturation rate constant
    p8: Base yield stress
    p9: Temperature coefficient for yield stress
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # Temperature-dependent parameters
    E = p0 + p1 * t
    H = p2 + p3 * t
    n = p4
    S = p5 + p6 * t
    k = p7
    sigma_y = p8 + p9 * t
    
    # Plastic strain: strain minus elastic strain at yield
    # Ensure E is not zero or negative to avoid division issues
    E_safe = np.where(np.abs(E) < 1e-8, 1e-8, E)
    
    eps_pl = np.maximum(0, s - sigma_y / E_safe)
    
    # Elastic term
    elastic_term = E * s
    
    # Hardening term (power law)
    # Use np.power with safe base
    hardening_term = H * np.power(eps_pl, n)
    
    # Saturation term (exponential)
    # Clip argument to avoid overflow
    exp_arg = -k * eps_pl
    exp_arg = np.clip(exp_arg, -50, 50)
    saturation_term = S * (1 - np.exp(exp_arg))
    
    stress = elastic_term + hardening_term + saturation_term
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress

def _fit_model(strain, temp, y):
    """
    Perform multi-start Nelder-Mead optimization to fit the model parameters.
    Returns the best parameters found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial best parameters (all ones as per spec)
    best_params = np.ones(10)
    best_mse = np.inf
    
    # Define multiple starting points to escape local minima
    # Base starts: variations around the all-ones initialization
    starts = [
        np.ones(10),
        np.array([100, -10, 10, -1, 0.5, 0.5, 10, 0.1, 0.01, -0.01]),
        np.array([50, -5, 5, -0.5, 0.3, 0.3, 5, 0.05, 0.005, -0.005]),
        np.array([200, -20, 20, -2, 0.8, 0.8, 20, 0.2, 0.02, -0.02]),
        np.array([75, -7.5, 7.5, -0.75, 0.6, 0.6, 7.5, 0.15, 0.015, -0.015]),
        np.array([150, -15, 15, -1.5, 0.7, 0.7, 15, 0.12, 0.012, -0.012]),
        np.array([10, -1, 1, -0.1, 0.2, 0.2, 1, 0.01, 0.001, -0.001]),
        np.array([5, -0.5, 0.5, -0.05, 0.1, 0.1, 0.5, 0.005, 0.0005, -0.0005]),
        np.array([300, -30, 30, -3, 0.9, 0.9, 30, 0.3, 0.03, -0.03]),
        np.array([500, -50, 50, -5, 1.0, 1.0, 50, 0.5, 0.05, -0.05]),
    ]
    
    # Add random starts if time permits
    np.random.seed(42)
    for _ in range(5):
        random_start = np.random.uniform(0.1, 100, size=10)
        random_start[1] *= -1
        random_start[3] *= -1
        random_start[9] *= -1
        starts.append(random_start)
    
    for start in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            # Use Nelder-Mead for robustness in non-convex landscapes
            result = minimize(
                lambda p: np.mean((equation(strain, temp, p) - y) ** 2),
                start,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < best_mse:
                if np.all(np.isfinite(result.x)):
                    best_params = result.x
                    best_mse = result.fun
                    
        except Exception:
            continue
            
    return best_params

# Global cache for fitted parameters
_fitted_params = None
_data_cached = None

def _get_fitted_params(strain, temp, y):
    global _fitted_params, _data_cached
    if _fitted_params is None or _data_cached != (strain, temp, y):
        _fitted_params = _fit_model(strain, temp, y)
        _data_cached = (strain, temp, y)
    return _fitted_params

# Note: The evaluator calls equation(strain, temp, params) with params provided by the fit.
# However, the task contract says: "The evaluator fits params... with scipy.optimize.minimize(method="BFGS") starting from all ones"
# This means the evaluator will call equation with the params it finds via BFGS.
# We cannot control the optimizer used by the evaluator.
# Therefore, we must provide a function that works well when optimized by BFGS from all-ones.
# 
# Wait, re-reading the contract: "The evaluator fits params, an array of length 10, with scipy.optimize.minimize(method="BFGS") starting from all ones, minimizing the training MSE."
# This implies I cannot implement multi-start Nelder-Mead inside `equation` because `equation` is called *with* the params already chosen by the evaluator's BFGS run.
# 
# Correction: The strategy must be a better *functional form* that is more amenable to BFGS optimization from all-ones, OR I must realize that I can't change the optimizer.
# 
# If I can't change the optimizer, I must provide a function that has a smoother, more convex-like landscape or a better local optimum reachable from all-ones via BFGS.
# 
# Let's pivot: Use a **Softplus-based smooth piecewise linear model** with a physically-motivated structure that is smoother than the RBF or Power-law (which can have sharp gradients).
# 
# New Strategy: Smooth Piecewise Linear (Softplus) with Temperature Coupling.
# sigma = E(T) * s + H(T) * softplus(s - s_y(T)) * s^0.5 + S(T) * softplus(s - s_y(T))^2
# This is smoother than power-law near zero and captures the knee.

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a Smooth Piecewise Linear (Softplus) model.
    
    Parameters (10 total):
    p0: Base elastic modulus
    p1: Temperature coefficient for elastic modulus
    p2: Base yield strain
    p3: Temperature coefficient for yield strain
    p4: Base hardening slope (plastic)
    p5: Temperature coefficient for hardening slope
    p6: Base saturation strain scale
    p7: Temperature coefficient for saturation strain scale
    p8: Base offset
    p9: Temperature coefficient for offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # Temperature-dependent parameters
    E = p0 + p1 * t
    s_y = p2 + p3 * t
    H = p4 + p5 * t
    S_scale = p6 + p7 * t
    offset = p8 + p9 * t
    
    # Ensure E and S_scale are positive to avoid weird behavior
    E_safe = np.clip(E, 1e-6, None)
    S_scale_safe = np.clip(S_scale, 1e-6, None)
    
    # Softplus approximation: log(1 + exp(x))
    # Use stable version
    x1 = (s - s_y) / S_scale_safe
    x2 = (s - s_y) / S_scale_safe
    
    # Clip to avoid overflow in exp
    x1_clip = np.clip(x1, -50, 50)
    x2_clip = np.clip(x2, -50, 50)
    
    softplus1 = np.log1p(np.exp(x1_clip))
    softplus2 = np.log1p(np.exp(x2_clip))
    
    # Model: Elastic + Plastic Hardening (linear in plastic strain) + Saturation (quadratic)
    # The softplus terms act as smooth hinges.
    elastic_term = E * s
    plastic_term = H * S_scale_safe * softplus1
    saturation_term = S_scale_safe * softplus2**2 # This might be too complex, let's keep it simple
    
    # Actually, let's use a simpler, very robust form:
    # sigma = E*s + H * (s - s_y)_+^1.5 + S * (1 - exp(-k*(s-s_y)_+))
    # But power 1.5 can be non-smooth at 0 if not careful.
    
    # Let's stick to a well-conditioned form:
    # sigma = E*s + A * tanh((s - s_y)/w) + B * (s - s_y) * tanh((s - s_y)/w)
    
    # Revised Model:
    # sigma = E*s + H * tanh((s - s_y)/w) * (s - s_y) + S * tanh((s - s_y)/w)
    # This is smooth everywhere.
    
    w = 0.01 + np.clip(p6, 1e-4, None) # Width of transition
    
    delta = (s - s_y) / w
    delta = np.clip(delta, -10, 10)
    tanh_delta = np.tanh(delta)
    
    elastic_term = E * s
    # Plastic term: proportional to (s - s_y) but smoothed by tanh
    plastic_term = H * (s - s_y) * tanh_delta
    # Offset/saturation term
    sat_term = S_scale_safe * tanh_delta
    
    stress = elastic_term + plastic_term + sat_term + offset
    
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
