"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed values
_cache = {}

def _get_cache_key(params, t, inventory):
    """Create a cache key from parameters and time."""
    return (params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q'], t)

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy with improved skew calibration.
    
    Uses a non-linear skew function that better balances spread capture
    against inventory risk. The skew magnitude is calibrated to the
    expected value of reducing position versus holding cost.
    """
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Time remaining
    t_rem = T - t
    
    # Base depth: risk-neutral quote depth
    base_depth = 1.0 / k
    
    # Normalize inventory to [-1, 1]
    inv_norm = inventory.astype(np.float64) / Q
    
    # Compute skew using a smooth non-linear function
    # The optimal skew balances:
    # - Expected earnings from a fill: depth * P(fill)
    # - Risk cost of inventory change: gamma * sigma^2 * q * dt
    # 
    # For a given inventory q, the marginal value of reducing position
    # is approximately gamma * sigma^2 * q * t_rem (integrated over time)
    # The marginal cost of tighter quotes is the reduced spread.
    #
    # We use a tanh-based smooth function to avoid extreme quotes
    # while maintaining strong skew at high inventory levels.
    
    # Scale factor calibrated to balance spread and risk
    # Higher A means more frequent fills, so we can afford wider quotes
    # Higher k means depth matters more (exponential sensitivity)
    # t_rem scales the urgency of reducing position
    
    # Effective skew parameter: combine risk and fill dynamics
    risk_factor = gamma * sigma * sigma * t_rem
    fill_factor = A * dt  # Expected fill probability per unit depth
    
    # The optimal skew should be proportional to risk / (risk + fill benefit)
    # This creates a smooth saturation effect
    # When risk is high relative to fill benefit, skew is large
    # When fill benefit is high, skew is moderate (capture more spread)
    
    # Smooth skew using a rational function
    # skew = base_skew * inv_norm / (1 + |inv_norm| * damping)
    # This ensures |skew| < 1 and is smooth at zero
    
    damping = 1.0 / (1.0 + risk_factor * 0.5)
    
    # Additional time-dependent urgency: more skew as we approach maturity
    time_urgency = 1.0 + 0.5 * (1.0 - t_rem / T)
    
    # Combined skew magnitude
    skew_scale = risk_factor * time_urgency * damping * 0.15
    
    # Apply smooth non-linear transform to inventory
    # Use tanh to bound the effect and smooth the response
    inv_tanh = np.tanh(inv_norm * 2.0)  # Scale to make tanh effective
    
    # Compute bid and ask depth adjustments
    # bid_depth decreases when long (encourage bid fills)
    # ask_depth increases when long (discourage ask fills)
    
    bid_adj = base_depth * (1.0 + skew_scale * inv_tanh)
    ask_adj = base_depth * (1.0 - skew_scale * inv_tanh)
    
    # Clip to valid range: must be > -3/k and finite
    min_depth = -3.0 / k + 0.01  # Small margin above -3/k
    max_depth = 10.0 / k  # Reasonable upper bound
    
    bid_depths = np.clip(bid_adj, min_depth, max_depth)
    ask_depths = np.clip(ask_adj, min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    # Set bid to inf where we're at max inventory (can't take more bids)
    bid_depths[at_max_inventory] = np.inf
    
    # Set ask to inf where we're at min inventory (can't take more asks)
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
