import numpy as np

def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) for the given state.
    inventory: (N,) array of ints in [-Q, Q]
    mid: (N,) array of floats
    params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    """
    N = len(inventory)
    Q = params['Q']
    k = params['k']
    
    # Heuristic parameters
    # Base depth around 0.5
    d_base = 0.5
    # Sensitivity to inventory
    alpha = 0.05
    # Min and max depth bounds
    d_min = 0.05
    d_max = 5.0
    
    bid_depths = np.empty(N, dtype=float)
    ask_depths = np.empty(N, dtype=float)
    
    # Vectorized operations
    q = np.array(inventory)
    
    # Calculate base depths adjusted by inventory
    # For q > 0: Ask tight, Bid wide
    # For q < 0: Bid tight, Ask wide
    # For q = 0: Both base
    
    abs_q = np.abs(q)
    
    # Tightening factor: reduce depth for the side that reduces inventory
    # Widening factor: increase depth for the side that increases inventory
    
    # Ask depth (sells, reduces q)
    # If q > 0, we want to sell, so make ask depth small.
    # If q < 0, we want to buy, so make ask depth large.
    # d_ask = d_base - alpha * q? 
    # If q=10, d_ask = 0.5 - 0.5 = 0.
    # If q=-10, d_ask = 0.5 + 0.5 = 1.0.
    # This linear relationship works.
    
    d_ask_raw = d_base - alpha * q
    d_bid_raw = d_base + alpha * q
    
    # Clip to valid range
    d_ask = np.clip(d_ask_raw, d_min, d_max)
    d_bid = np.clip(d_bid_raw, d_min, d_max)
    
    # Handle boundaries
    # If q == Q, no bid fill -> d_bid = inf
    # If q == -Q, no ask fill -> d_ask = inf
    bid_depths = np.where(q == Q, np.inf, d_bid)
    ask_depths = np.where(q == -Q, np.inf, d_ask)
    
    # Ensure no negative or NaN
    # The problem states values below -3/k are invalid. 
    # Our d_min is 0.05 which is positive.
    # If k is very large, -3/k is close to 0. 
    # Let's ensure d >= 0.01 just in case.
    # Also check if inf is handled correctly by the evaluator (it says np.inf means no quote).
    
    return bid_depths, ask_depths
