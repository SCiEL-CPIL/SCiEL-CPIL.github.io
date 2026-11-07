"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290  # Safety margin before 330s kill


def quote(t, inventory, mid, params):
    """
    Inventory-aware quoting with adaptive depth adjustment.
    
    Strategy:
    - Base depth from 1/k (risk-neutral)
    - Adjust based on inventory: widen in direction that increases |q|,
      tighten in direction that reduces |q|
    - Use a nonlinear (tanh-like) adjustment to handle large inventories
    - Scale adjustment with remaining time and volatility
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    dt = T / steps
    remaining_time = T - t
    time_fraction = remaining_time / T  # 1 at start, 0 at end
    
    q = inventory.astype(np.float64)
    
    # Base depth: risk-neutral level
    base_depth = 1.0 / k
    
    # Inventory adjustment: widen quotes in direction that increases |q|,
    # tighten in direction that reduces |q|
    # When q > 0 (long): bid wider (less likely to fill), ask tighter (more likely to fill)
    # When q < 0 (short): bid tighter, ask wider
    
    # Nonlinear adjustment with saturation
    # beta controls the strength of inventory response
    # Scale with volatility and remaining time for better risk management
    beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
    
    # Use tanh for smooth nonlinear response
    q_scaled = q / Q  # Normalize inventory to [-1, 1]
    q_adjust = np.tanh(1.5 * q_scaled)  # Smooth saturation
    
    bid_depth = base_depth + beta * q_adjust
    ask_depth = base_depth - beta * q_adjust
    
    # Clamp to minimum positive depth to avoid numerical issues
    min_depth = 0.01
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # At boundaries, no quote on the side that would push inventory further
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
