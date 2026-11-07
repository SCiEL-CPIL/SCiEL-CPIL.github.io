import numpy as np
import time

_solution_cache = {}

def _get_policy_for_params(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _solution_cache:
        return _solution_cache[key]
        
    dt = T / steps
    # Base depth from liquidity/intensity parameters
    # Heuristic: d_base ~ 1 / (k * A * dt)
    # But capped to reasonable values
    d_base = 1.0 / (k * A * dt)
    
    # Inventory risk term coefficient
    # In continuous AS, spread shift is gamma * sigma^2 * q * tau
    # Here we use the same structure
    risk_coeff = gamma * sigma**2
    
    _solution_cache[key] = (d_base, risk_coeff, dt, T, Q)
    return _solution_cache[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    d_base, risk_coeff, dt, T_full, Q_max = _get_policy_for_params(params)
    
    # Time remaining
    tau = T_full - t
    # Ensure tau is positive and not too large
    tau = np.clip(tau, 0.0, T_full)
    
    # Inventory array
    q = np.asarray(inventory)
    
    # Calculate depths
    # bid_depth = d_base - risk_coeff * q * tau
    # ask_depth = d_base + risk_coeff * q * tau
    
    # The AS model uses (1 - exp(-gamma sigma^2 tau)) / (gamma sigma^2) as the base spread.
    # Let's incorporate that time-decay factor into d_base to be more accurate.
    # If gamma sigma^2 is small, 1/(gamma sigma^2). If large, tau.
    # Let's stick to the simpler linear approximation for speed and stability, 
    # but scale d_base by the time-decay factor if possible.
    
    # Actually, the term 1/(k A dt) is constant.
    # The risk term is linear in q and tau.
    
    bid_depths = d_base - risk_coeff * q * tau
    ask_depths = d_base + risk_coeff * q * tau
    
    # Constraints
    # Depths must be >= 0. If negative, set to a very small positive or 0?
    # The problem says "distances below and above mid".
    # "np.inf means no quote".
    # "Values below -3/k ... make the whole result invalid".
    # So we must ensure bid_depths >= 0.
    # If the optimal is to not quote, we should return np.inf.
    # However, returning 0 might be interpreted as quoting at mid, which is risky.
    # The heuristic suggests that if the risk term dominates, we don't want to quote.
    # Let's set negative depths to np.inf.
    
    # But wait, if d_base is small and risk is high, we might get negative.
    # Let's clip to a small epsilon? Or inf?
    # "np.inf means no quote".
    # If we quote at 0, we get filled instantly? p(0) = 1 - exp(-A dt).
    # If the expected value of that fill is negative (due to inventory), we should not quote.
    # So np.inf is the correct signal for "do not quote".
    
    bid_depths[bid_depths < 0] = np.inf
    ask_depths[ask_depths < 0] = np.inf
    
    # Boundary conditions:
    # No bid fill at q = Q. So if q == Q, bid_depth = np.inf.
    # No ask fill at q = -Q. So if q == -Q, ask_depth = np.inf.
    
    bid_depths[q == Q_max] = np.inf
    ask_depths[q == -Q_max] = np.inf
    
    # Ensure no NaNs or Infs that are invalid (only +inf is allowed for no quote)
    # The problem states: "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # So we must ensure no -inf or NaN.
    
    # Check for NaNs
    bid_depths[np.isnan(bid_depths)] = np.inf
    ask_depths[np.isnan(ask_depths)] = np.inf
    
    # Check for -inf
    bid_depths[np.isneginf(bid_depths)] = np.inf
    ask_depths[np.isneginf(ask_depths)] = np.inf
    
    return bid_depths, ask_depths
