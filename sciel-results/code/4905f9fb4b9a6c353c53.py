"""Analytic separable DP for optimal market making with inventory risk."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 310.0
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve via analytic separable Bellman decomposition.
    
    Key insight: The Bellman equation separates into independent 1-D problems
    for bid and ask depths once the next-step value differentials are known.
    
    For state q at time t:
      V_b = V[t+1, q+1] (value if bid fills)
      V_a = V[t+1, q-1] (value if ask fills)  
      V_s = V[t+1, q]   (value if no fill or both fill)
      
    The optimal bid depth d_b maximizes:
      (V_b - V_s) * Pb + d_b * Pb + [terms independent of d_b]
    where Pb = 1 - exp(-A*exp(-k*d_b)*dt)
    
    Taking derivative and setting to zero:
      (V_b - V_s) * Pb' + Pb + d_b * Pb' = 0
      where Pb' = dPb/dd_b = A*exp(-k*d_b)*dt*k*exp(-k*d_b) = A*k*dt*exp(-2*k*d_b)
      
    Let u = exp(-k*d_b), so Pb = 1 - exp(-c*u) where c = A*dt
    Then d_b = -ln(u)/k, Pb' = c*k*u^2
    
    The equation becomes:
      (V_b - V_s) * c*k*u^2 + (1 - exp(-c*u)) + (-ln(u)/k) * c*k*u^2 = 0
      c*k*u^2 * [(V_b - V_s) - ln(u)] + 1 - exp(-c*u) = 0
      
    This is a 1-D equation in u in (0, 1], solvable by Newton's method.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    V = np.zeros((N + 1, n_q))
    bid_table = np.zeros((N, n_q))
    ask_table = np.zeros((N, n_q))
    
    c = A * dt  # characteristic probability scale
    cost_per_q = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    def _solve_depth(delta_v, c, k, dt):
        """
        Solve for optimal depth given value differential delta_v = V_fill - V_same.
        
        If delta_v >= 0: we want to fill (depth helps), solve:
          c*k*u^2 * (delta_v - ln(u)) + 1 - exp(-c*u) = 0
        
        If delta_v < 0: we want to avoid fill, optimal depth is large (-> infinity)
        but we cap at a reasonable max. Actually if delta_v < 0, the optimal is
        to set Pb as small as possible, i.e., depth -> infinity. But since the
        objective has +Pb*d_b, and d_b grows, we need to be careful.
        
        Let's re-derive: maximize f(d) = delta_v * P(d) + d * P(d)
        where P(d) = 1 - exp(-c * exp(-k*d))
        
        f'(d) = delta_v * P'(d) + P(d) + d * P'(d)
              = P'(d) * (delta_v + d) + P(d)
        
        P'(d) = c * k * exp(-2*k*d) > 0 always
        
        f'(d) = 0 => c*k*exp(-2kd) * (delta_v + d) + 1 - exp(-c*exp(-kd)) = 0
        
        Let u = exp(-kd) in (0,1]:
          c*k*u^2 * (delta_v - ln(u)/k) + 1 - exp(-c*u) = 0
          c*k*u^2 * delta_v - c*u^2*ln(u) + 1 - exp(-c*u) = 0
        
        Define g(u) = c*k*u^2*delta_v - c*u^2*ln(u) + 1 - exp(-c*u)
        g'(u) = 2*c*k*u*delta_v - 2*c*u*ln(u) - c*u + c*exp(-c*u)
               = c*u * [2*k*delta_v - 2*ln(u) - 1 + exp(-c*u)]
        
        For delta_v > 0: g(0+) = 1 > 0, g(1) = c*k*delta_v - 0 + 1 - exp(-c) 
                          = c*k*delta_v + 1 - exp(-c) > 0
        Hmm, both positive. Let me check the sign convention.
        
        Actually wait: we're MAXIMIZING f, so we need f'(d) = 0 with f''(d) < 0.
        Let me just do a robust 1-D search with Newton + bisection fallback.
        """
        if delta_v >= 0:
            # Want to encourage fills. Optimal d is in [0, ~10/k]
            # Use Newton on g(u) = 0 where u = exp(-k*d), u in (0, 1]
            # g(u) = c*k*u^2*delta_v - c*u^2*ln(u) + 1 - exp(-c*u)
            # At u=1: g(1) = c*k*delta_v + 1 - exp(-c)
            # At u->0: g(u) -> 1 (since u^2*ln(u)->0, exp(-c*u)->1)
            # 
            # Actually for large delta_v, the optimal u is small (large depth).
            # For small delta_v, optimal u is near 1 (small depth).
            #
            # Let's just do a simple scalar optimization via golden section or 
            # Newton on f(d) directly.
            
            def f(d):
                u = np.exp(-k * d)
                P = 1.0 - np.exp(-c * u)
                return delta_v * P + d * P
            
            def fp(d):
                u = np.exp(-k * d)
                P = 1.0 - np.exp(-c * u)
                Pp = c * k * u * u
                return Pp * (delta_v + d) + P
            
            # Newton's method starting from d0 = 1/k
            d = 1.0 / k
            for _ in range(30):
                try:
                    fd = fp(d)
                    if abs(fd) < 1e-12:
                        break
                    # Second derivative for safety
                    # f''(d) = P''(d)*(delta_v+d) + 2*P'(d)
                    # P'(d) = c*k*exp(-2kd)
                    # P''(d) = -2*k*c*k*exp(-2kd) = -2*k*P'(d)
                    Pp = c * k * np.exp(-2 * k * d)
                    Ppp = -2 * k * Pp
                    fpp = Ppp * (delta_v + d) + 2 * Pp
                    if fpp == 0:
                        break
                    d_new = d - fd / fpp
                    # Damping
                    if d_new < 0:
                        d_new = d / 2.0
                    if d_new > 20.0 / k:
                        d_new = 20.0 / k
                    if abs(d_new - d) < 1e-10:
                        d = d_new
                        break
                    d = d_new
                except (FloatingPointError, OverflowError):
                    break
            
            return max(0.0, min(d, 10.0 / k))
        else:
            # delta_v < 0: we don't want fills. f(d) = delta_v*P + d*P
            # As d -> inf, P -> 0, d*P -> 0 (exponentially). f -> 0.
            # At d=0: f(0) = delta_v*(1-exp(-c)) < 0
            # So f increases from negative to 0. Maximum at d -> inf? 
            # But f'(d) = P'(d)*(delta_v+d) + P(d)
            # For large d: P'(d) ~ c*k*exp(-2kd), P(d) ~ c*exp(-kd)
            # f'(d) ~ c*k*exp(-2kd)*(delta_v+d) + c*exp(-kd)
            # For very large d, the second term dominates and is positive.
            # So f is increasing for large d, meaning we want d as large as possible.
            # But in practice, we cap depth. The marginal benefit of going deeper
            # diminishes. Let's find where f'(d) = 0 if it exists in reasonable range.
            
            def fp(d):
                u = np.exp(-k * d)
                P = 1.0 - np.exp(-c * u)
                Pp = c * k * u * u
                return Pp * (delta_v + d) + P
            
            # f'(0) = c*k*(delta_v) + (1-exp(-c))
            # If f'(0) > 0, then f is increasing at 0, so optimum is at boundary or further
            # If f'(0) < 0, there's a local max somewhere
            
            d = 1.0 / k
            for _ in range(30):
                try:
                    fd = fp(d)
                    if abs(fd) < 1e-12:
                        break
                    Pp = c * k * np.exp(-2 * k * d)
                    Ppp = -2 * k * Pp
                    fpp = Ppp * (delta_v + d) + 2 * Pp
                    if fpp == 0:
                        break
                    d_new = d - fd / fpp
                    if d_new < 0:
                        d_new = d / 2.0
                    if d_new > 20.0 / k:
                        d_new = 20.0 / k
                    if abs(d_new - d) < 1e-10:
                        d = d_new
                        break
                    d = d_new
                except (FloatingPointError, OverflowError):
                    break
            
            return max(0.0, min(d, 10.0 / k))
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: simple heuristic
            for tt in range(t, -1, -1):
                base_d = 1.0 / k
                bid_table[tt] = np.clip(base_d + 0.01 * q_vals, 0.0, None)
                ask_table[tt] = np.clip(base_d - 0.01 * q_vals, 0.0, None)
                V[tt] = V[tt + 1] if tt + 1 <= N else 0.0
            break
        
        V_next = V[t + 1, :]
        
        # Value differentials for each inventory state
        # Bid fill: q -> q+1, so delta_v_bid[i] = V_next[i+1] - V_next[i]
        # Ask fill: q -> q-1, so delta_v_ask[i] = V_next[i-1] - V_next[i]
        
        delta_bid = np.zeros(n_q)
        delta_bid[:n_q - 1] = V_next[1:n_q] - V_next[:n_q - 1]
        delta_bid[n_q - 1] = 0.0  # q=Q, no bid fill possible
        
        delta_ask = np.zeros(n_q)
        delta_ask[1:] = V_next[:n_q - 1] - V_next[1:]
        delta_ask[0] = 0.0  # q=-Q, no ask fill possible
        
        # Solve for optimal depths
        # Vectorize Newton iteration over all i
        bid_depths = np.ones(n_q) / k
        ask_depths = np.ones(n_q) / k
        
        # Mask for valid bid/ask
        bid_valid = np.ones(n_q, dtype=bool)
        bid_valid[n_q - 1] = False
        ask_valid = np.ones(n_q, dtype=bool)
        ask_valid[0] = False
        
        # Newton iterations
        for _ in range(40):
            if time.time() > _DEADLINE:
                break
            
            # For bid depths
            d = bid_depths
            u = np.exp(-k * d)
            P = 1.0 - np.exp(-c * u)
            Pp = c * k * u * u
            Ppp = -2.0 * k * Pp
            
            # f'(d) = Pp * (delta_bid + d) + P
            fp = Pp * (delta_bid + d) + P
            fpp = Ppp * (delta_bid + d) + 2.0 * Pp
            
            # Newton update
            valid = bid_valid & (np.abs(fpp) > 1e-15) & np.isfinite(fp) & np.isfinite(fpp)
            d_new = d.copy()
            d_new[valid] = d[valid] - fp[valid] / fpp[valid]
            d_new = np.clip(d_new, 0.0, 10.0 / k)
            
            if np.max(np.abs(d_new - d)) < 1e-10:
                bid_depths = d_new
                break
            bid_depths = d_new
            
            # For ask depths
            d = ask_depths
            u = np.exp(-k * d)
            P = 1.0 - np.exp(-c * u)
            Pp = c * k * u * u
            Ppp = -2.0 * k * Pp
            
            fp = Pp * (delta_ask + d) + P
            fpp = Ppp * (delta_ask + d) + 2.0 * Pp
            
            valid = ask_valid & (np.abs(fpp) > 1e-15) & np.isfinite(fp) & np.isfinite(fpp)
            d_new = d.copy()
            d_new[valid] = d[valid] - fp[valid] / fpp[valid]
            d_new = np.clip(d_new, 0.0, 10.0 / k)
            
            if np.max(np.abs(d_new - d)) < 1e-10:
                ask_depths = d_new
                break
            ask_depths = d_new
        
        bid_table[t, :] = bid_depths
        ask_table[t, :] = ask_depths
        
        # Compute V[t, i] using the optimal depths
        # V[t,i] = Pb*(1-Pa)*V_b + Pa*(1-Pb)*V_a + (Pb*Pa + (1-Pb)*(1-Pa))*V_s + Pb*db + Pa*da - cost
        # where V_b = V_next[i+1] (or V_s if at boundary), etc.
        
        db = bid_depths
        da = ask_depths
        u_b = np.exp(-k * db)
        u_a = np.exp(-k * da)
        Pb = 1.0 - np.exp(-c * u_b)
        Pa = 1.0 - np.exp(-c * u_a)
        
        # Apply boundary masks
        Pb[n_q - 1] = 0.0  # q=Q, no bid fill
        Pa[0] = 0.0        # q=-Q, no ask fill
        
        # Next state values
        V_b = np.zeros(n_q)
        V_b[:n_q - 1] = V_next[1:n_q]
        V_b[n_q - 1] = V_next[n_q - 1]
        
        V_a = np.zeros(n_q)
        V_a[1:] = V_next[:n_q - 1]
        V_a[0] = V_next[0]
        
        V_s = V_next.copy()
        
        V[t, :] = (Pb * (1 - Pa) * V_b +
                   Pa * (1 - Pb) * V_a +
                   (Pb * Pa + (1 - Pb) * (1 - Pa)) * V_s +
                   Pb * db +
                   Pa * da -
                   cost_per_q)
    
    _CACHE[key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    bid_table, ask_table = _solve_regime(
        params["sigma"], params["A"], params["k"],
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    Q = params["Q"]
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    bid_depths = np.clip(bid_depths, 0.0, None)
    ask_depths = np.clip(ask_depths, 0.0, None)
    
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=1e6, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=1e6, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
