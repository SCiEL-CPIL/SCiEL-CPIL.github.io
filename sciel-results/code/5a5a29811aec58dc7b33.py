# EVOLVE-BLOCK-START
import numpy as np

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
        
    # Time remaining
    tau = T - t
    
    # Analytical approximation for optimal depths
    # Based on Avellaneda-Stoikov continuous-time limit
    # Bid depth b = 1/k + q * gamma * sigma^2 * tau
    # Ask depth a = 1/k - q * gamma * sigma^2 * tau
    # Note: In AS, the spread is 2/k. The reservation price shifts by -q*gamma*sigma^2*tau.
    # Mid is the fair price.
    # Bid = Mid + Shift - 1/k
    # Ask = Mid + Shift + 1/k
    # Depth below mid (bid_depth) = Mid - Bid = Mid - (Mid + Shift - 1/k) = -Shift + 1/k
    # Depth above mid (ask_depth) = Ask - Mid = Mid + Shift + 1/k - Mid = Shift + 1/k
    # Shift = -q * gamma * sigma^2 * tau
    # So:
    # bid_depth = 1/k + q * gamma * sigma^2 * tau
    # ask_depth = 1/k - q * gamma * sigma^2 * tau
    
    inv = inventory.astype(float)
    
    # Calculate depths
    # Note: If q is positive, we want to sell, so ask depth should be small (tight) and bid depth large (wide).
    # Formula:
    # bid_depth = 1/k + q * gamma * sigma^2 * tau
    # ask_depth = 1/k - q * gamma * sigma^2 * tau
    
    # Let's verify signs.
    # If q > 0 (long inventory), we want to reduce it.
    # We want ask fills (sell) to be more likely -> ask_depth should be small.
    # We want bid fills (buy) to be less likely -> bid_depth should be large.
    # Formula:
    # ask_depth = 1/k - q * ... (decreases with q) -> Correct.
    # bid_depth = 1/k + q * ... (increases with q) -> Correct.
    
    factor = gamma * sigma * sigma * tau
    base_depth = 1.0 / k
    
    bid_depths = base_depth + inv * factor
    ask_depths = base_depth - inv * factor
    
    # Ensure depths are non-negative. If depth <= 0, it implies we should quote at or beyond mid.
    # In this model, depths are distances. Negative depth doesn't make sense as a "distance below/above".
    # If the optimal depth is negative, it means the reservation price has crossed the mid.
    # We should clip to 0? Or a small epsilon?
    # The task says "distances below and above mid".
    # If we clip to 0, we are quoting at mid.
    # If we set to inf, we don't quote.
    # Usually, if the reservation price is very aggressive, you might cross the spread.
    # But here we are quoting relative to mid.
    # Let's clip to 0.
    
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Handle boundaries:
    # "no bid fill at q = Q" -> If inventory is Q, we cannot buy more. So bid_depth should be inf.
    # "no ask fill at q = -Q" -> If inventory is -Q, we cannot sell more. So ask_depth should be inf.
    
    # Create masks
    at_max_inv = (inventory == Q)
    at_min_inv = (inventory == -Q)
    
    bid_depths[at_max_inv] = np.inf
    ask_depths[at_min_inv] = np.inf
    
    # Also, if the computed depth is very small, it's fine.
    # Check for any NaNs or infs that shouldn't be there?
    # np.inf is allowed for "no quote".
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
