    # Theoretical optimal skew for LQ control
    # Depth slope beta = gamma * sigma^2 * t_rem / k
    # We apply this to normalized inventory q/Q.
    # d(q) = d0 - beta * (q/Q) * Q ? No.
    # d(q) = d0 - beta * q.
    # In code: d = base_depth * (1 + skew_scale * inv_norm)
    # d = base_depth + base_depth * skew_scale * (q/Q)
    # We want base_depth * skew_scale / Q = beta = gamma * sigma^2 * t_rem / k
    # So skew_scale = beta * Q / base_depth = (gamma * sigma^2 * t_rem / k) * Q / (1/k)
    # skew_scale = gamma * sigma^2 * t_rem * Q
    
    # Let's check magnitude.
    # Regime 1: gamma=0.05, sigma=2, t_rem=1, Q=30.
    # skew_scale = 0.05 * 4 * 1 * 30 = 6.0
    # Current code: risk_factor = 0.05 * 4 * 1 = 0.2.
    # skew_scale = 0.2 * 1.5 * (1/(1+0.1)) * 0.15 ≈ 0.2 * 1.5 * 0.9 * 0.15 ≈ 0.04.
    # Huge difference! 6.0 vs 0.04.
    # The current skew is extremely small.
    # If skew is too small, we don't manage inventory well.
    # If skew is too large, we quote weird depths (negative bid depth?).
    
    # If skew_scale = 6.0, and inv_norm = 1 (q=30),
    # bid_adj = base_depth * (1 - 6.0) = -5 * base_depth.
    # base_depth = 1/1.5 = 0.66.
    # bid_adj = -3.3.
    # min_depth = -3/k = -2.0.
    # So it clips to -2.0.
    # This means for high inventory, we quote at the minimum valid depth (closest to mid).
    # This is good! We want to dump inventory.
    
    # So the theoretical scaling suggests a much larger skew.
