"""Market-making with inventory risk: Analytical LQ Control via HJB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 290.0

# Cache for regime-specific precomputed constants
_CONST_CACHE = {}

def _get_constants(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _CONST_CACHE:
        dt = params['T'] / params['steps']
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        
        _CONST_CACHE[key] = {
            'dt': dt,
            'sigma': sigma,
            'A': A,
            'k': k,
            'gamma': gamma,
            'Q': Q
        }
    return _CONST_CACHE[key]

def _solve_depth_equation(x, A, k, dt, cost_term):
    """
    Solve A * exp(-k*x) * x = cost_term / dt using Newton's method.
    x is the depth.
    cost_term = (gamma/2) * sigma^2 * q^2
    """
    # f(x) = A * exp(-k*x) * x - (cost_term / dt) = 0
    # f'(x) = A * exp(-k*x) * (1 - k*x)
    
    # Initial guess:
    # If cost_term is 0, x -> 0 (but we want max profit, so x -> 1/k? 
    # Wait, if cost_term is 0, we want to maximize A*exp(-kx)*x. 
    # Max is at x=1/k.
    # If cost_term is large, x is small.
    # Let's guess x0 = min(1/k, cost_term / (A*dt) + 1e-6) 
    # Actually, if cost_term is very small, x is near 1/k.
    # If cost_term is very large, x is near cost_term / (A*dt).
    
    target = cost_term / dt
    if target < 1e-12:
        return 1.0 / k

    # Rough estimate: A * x * exp(-kx) = target
    # If x is small, exp(-kx) ~ 1, so x ~ target / A
    # If x is large, exp(-kx) is small.
    
    # Let's use a few fixed points or a simple iterative method.
    # Newton: x_{n+1} = x_n - f/f'
    
    # Safe initial guess
    # If target is small, x is large.
    # If target is large, x is small.
    # Let's try x0 = np.sqrt(target / A) as a middle ground? No.
    
    # Let's use the fact that max of A*exp(-kx)*x is A/(e*k).
    # If target > A/(e*k), there is no solution? 
    # No, the equation is for the *marginal* condition.
    # The objective is max A*exp(-kx)*x * dt - cost_term * dt.
    # Derivative: A*exp(-kx)*(1-kx)*dt - 2*cost_term*x*dt = 0?
    # Wait, the cost in the objective is (gamma/2) sigma^2 q^2 dt.
    # The HJB derivative w.r.t depth involves the change in value function.
    # V_q is the marginal value of inventory.
    # The condition is: Marginal Profit = Marginal Cost.
    # Marginal Profit of increasing depth x by dx:
    # d/dx [ (1-exp(-A e^{-kx} dt)) * x ] ≈ A e^{-kx} dt (for small dt)
    # Actually, let's look at the reward: R = p_bid * x.
    # p_bid = 1 - exp(-lambda dt) ≈ lambda dt.
    # R ≈ lambda x dt = A e^{-kx} x dt.
    # dR/dx = A e^{-kx} (1-kx) dt.
    
    # Marginal Cost:
    # The inventory risk penalty is (gamma/2) sigma^2 q^2 dt.
    # Changing depth x changes the expected inventory change.
    # E[dq] = p_ask - p_bid.
    # For bid depth x: p_bid increases with x. So E[dq] decreases (becomes more negative).
    # If q > 0 (long inventory), we want to reduce q, so we want to increase p_bid (sell more).
    # So we want to increase x (bid depth) to make it more likely to fill?
    # Wait, if I quote deeper (larger x), p_bid increases.
    # If I am long (q>0), I want to sell. Selling increases q? No, "bid hit ... moves inventory one unit".
    # Usually: Bid hit -> Buy? Or Sell?
    # "bid hit with probability ... A fill earns its depth and moves inventory one unit".
    # Standard MM: You quote bid (buy) and ask (sell).
    # If bid is hit, you BUY. Inventory increases (q -> q+1).
    # If ask is hit, you SELL. Inventory decreases (q -> q-1).
    # Let's re-read carefully: "bid hit ... moves inventory one unit".
    # "ask likewise".
    # Usually, inventory q is number of shares held.
    # Buy: q+1. Sell: q-1.
    # If q is positive (long), you want to sell.
    # To sell, you need the ASK to be hit.
    # Ask depth y. p_ask = 1 - exp(-A e^{-ky} dt).
    # To increase p_ask, increase y.
    # So if q > 0, we should increase ask depth y and decrease bid depth x.
    
    # Let's check the sign of the cost gradient.
    # Value function V(q).
    # HJB: max_x,y { p_x x + p_y y - C(q) dt + E[V(q + d_q)] }
    # d_q = I(bid hit) - I(ask hit). (Assuming bid=buy, ask=sell)
    # E[d_q] = p_x - p_y.
    # V(q + E[d_q]) ≈ V(q) + V_q E[d_q].
    # So term is V_q (p_x - p_y).
    # If q > 0, V_q < 0 (marginal value of extra inventory is negative due to risk).
    # We want to maximize V_q p_x - V_q p_y = -|V_q| p_x + |V_q| p_y.
    # To maximize this, we want to MINIMIZE p_x and MAXIMIZE p_y.
    # Minimize p_x -> minimize x.
    # Maximize p_y -> maximize y.
    
    # So for q > 0:
    # Bid depth x should be small.
    # Ask depth y should be large.
    
    # Let's derive the optimal x (bid depth) for q > 0.
    # Maximize: p_x x - |V_q| p_x. (Ignoring y for a moment, assuming independent optimization? No, they are coupled via V_q but x and y appear separately in the immediate reward and the linear approximation of the future value).
    # Actually, the immediate reward for bid is p_x x.
    # The future value impact is V_q p_x.
    # Total term for x: p_x (x + V_q).
    # Since V_q is negative, let V_q = -C (where C>0).
    # Term: p_x (x - C).
    # If x < C, this is negative. We should set x=0?
    # If x > C, this is positive.
    # Let's maximize f(x) = (1-exp(-A e^{-kx} dt)) (x - C).
    # For small dt, f(x) ≈ A e^{-kx} dt (x - C).
    # Maximize g(x) = e^{-kx} (x - C).
    # g'(x) = -k e^{-kx} (x - C) + e^{-kx} (1) = e^{-kx} (1 - k(x - C)) = e^{-kx} (1 - kx + kC).
    # Set to 0: 1 - kx + kC = 0 => x = 1/k + C.
    
    # So optimal bid depth x* = 1/k + C, where C = -V_q.
    # Since C > 0 for q > 0, x* > 1/k.
    # Wait, if x* > 1/k, p_x is smaller than at 1/k.
    # But the term (x-C) becomes larger.
    # Let's check second derivative or bounds.
    # If x < C, g(x) < 0. We prefer 0 (no quote) if g(x*) < 0.
    # g(x*) = e^{-k(1/k+C)} (1/k + C - C) = e^{-1-kC} (1/k).
    # This is always positive. So we always quote?
    # Unless x* is so large that p_x is negligible?
    # The formula assumes x is a valid depth.
    
    # Now for Ask depth y (for q > 0).
    # Term: p_y (y - V_q) = p_y (y + C).
    # Maximize h(y) = (1-exp(-A e^{-ky} dt)) (y + C) ≈ A e^{-ky} dt (y + C).
    # Maximize m(y) = e^{-ky} (y + C).
    # m'(y) = -k e^{-ky} (y + C) + e^{-ky} (1) = e^{-ky} (1 - k(y + C)).
    # Set to 0: 1 - ky - kC = 0 => y = 1/k - C.
    # If C > 1/k, y < 0. We should set y=0 (no quote) or y=0?
    # If y < 0, m(y) is decreasing for y > 1/k - C?
    # If C > 1/k, the max is at y=0?
    # m(0) = C.
    # m(y) for y>0: if 1/k - C < 0, the peak is to the left of 0. So m(y) is decreasing for y>0.
    # So optimal y = 0 if C >= 1/k.
    # If C < 1/k, optimal y = 1/k - C.
    
    # Summary for q > 0 (Long):
    # C = -V_q (positive marginal risk cost).
    # Bid depth x = 1/k + C.
    # Ask depth y = max(0, 1/k - C).
    
    # For q < 0 (Short):
    # V_q > 0. Let C = V_q (positive).
    # Term for bid x: p_x (x + V_q) = p_x (x + C).
    # Maximize e^{-kx} (x + C) => x = 1/k - C.
    # If C >= 1/k, x = 0.
    # Term for ask y: p_y (y - V_q) = p_y (y - C).
    # Maximize e^{-ky} (y - C) => y = 1/k + C.
    
    # So the policy is symmetric:
    # C = |V_q|
    # If q > 0:
    #   bid = 1/k + C
    #   ask = max(0, 1/k - C)
    # If q < 0:
    #   bid = max(0, 1/k - C)
    #   ask = 1/k + C
    # If q = 0:
    #   C = 0
    #   bid = 1/k
    #   ask = 1/k
    
    # Now, what is C = |V_q|?
    # In LQ theory, V(q) ≈ - (gamma/2) sigma^2 q^2 * (some time factor)?
    # Actually, the cost term in the objective is (gamma/2) sigma^2 q^2 dt.
    # The HJB equation for the value function V(t, q):
    # -V_t = max_{x,y} { E[reward] + E[V(t+dt, q+dq)] - V(t,q) }
    # For small dt, V(t+dt, q+dq) ≈ V(t,q) + V_t dt + V_q dq.
    # So -V_t = max { E[reward] + V_q E[dq] } / dt (roughly).
    # The term E[reward] is O(dt). E[dq] is O(dt).
    # So V_t and V_q are related.
    
    # A common approximation for this type of problem (Avellaneda-Stoikov or similar) is that the inventory cost acts like a risk aversion.
    # The "optimal spread" or "tilt" is proportional to gamma * sigma^2 * q.
    # Specifically, the deviation from the neutral depth 1/k is proportional to the risk cost.
    # Let's assume C = alpha * q, where alpha is a constant depending on params.
    # From the derivation, C = -V_q.
    # If V(q) ≈ - (gamma/2) sigma^2 q^2 * K, then V_q = - gamma sigma^2 q K.
    # So C = gamma sigma^2 q K.
    # What is K?
    # In the steady state or short horizon, the risk penalty accumulates.
    # A simple heuristic: C = gamma * sigma^2 * q * dt? No, that's too small.
    # Let's look at the "tilt" in the seed program. It used a grid search for m.
    # The optimal tilt m should be such that the change in depth is proportional to q.
    # bid = 1/k + m*q.
    # ask = 1/k - m*q.
    # Comparing to our derived:
    # If q > 0: bid = 1/k + C, ask = 1/k - C.
    # So C = m * q.
    # Thus m = C / q = gamma * sigma^2 * K.
    
    # How to estimate K?
    # The cost is (gamma/2) sigma^2 q^2 dt.
    # Over the remaining time T-t, the expected cost is roughly (gamma/2) sigma^2 E[q^2] (T-t).
    # If we assume q follows a random walk with variance proportional to dt, E[q^2] ~ q^2.
    # The marginal cost of holding one more unit is gamma sigma^2 q (T-t).
    # So C ≈ gamma sigma^2 q (T-t).
    # Let's use C = gamma * sigma^2 * q * (T - t).
    # Wait, the cost in the objective is summed over steps.
    # Sum (gamma/2) sigma^2 q_n^2 dt.
    # If q stays constant, total cost is (gamma/2) sigma^2 q^2 T.
    # Marginal cost of increasing q by 1 is gamma sigma^2 q T.
    # So C = gamma sigma^2 q (T-t).
    
    # Let's test this hypothesis.
    # m = gamma * sigma^2 * (T-t).
    # bid = 1/k + m*q
    # ask = 1/k - m*q
    
    # We need to clip depths to [0, inf].
    # Also, if q=Q, no bid fill. So bid depth can be anything, but effectively 0 reward.
    # We can just set bid to 1/k + m*Q.
    
    # Let's refine the constant.
    # In the Avellaneda-Stoikov model, the optimal ask/bid is:
    # s_a = 1/(2 gamma sigma^2) + ...
    # Here the fill probability is different (Poisson with rate depending on depth).
    # The "1/k" term comes from the max of x e^{-kx}.
    # The "C" term comes from the inventory risk.
    
    # Let's use C = gamma * sigma^2 * q * (T - t).
    
    return _CONST_CACHE[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    c = _get_constants(params)
    dt = c['dt']
    sigma = c['sigma']
    A = c['A']
    k = c['k']
    gamma = c['gamma']
    Q = c['Q']
    T = params['T']
    
    # Neutral depth
    d0 = 1.0 / k
    
    # Risk cost coefficient
    # C = gamma * sigma^2 * q * (T - t)
    # T - t is the remaining time.
    remaining_time = T - t
    
    # Calculate C for each path
    C = gamma * sigma**2 * inventory * remaining_time
    
    # Calculate depths
    # bid = d0 + C
    # ask = d0 - C
    # But we must ensure non-negative.
    # And handle the specific cases:
    # If q > 0:
    #   bid = d0 + C
    #   ask = max(0, d0 - C)
    # If q < 0:
    #   bid = max(0, d0 - C)
    #   ask = d0 + C
    
    # Note: C has the same sign as q.
    # If q > 0, C > 0.
    #   bid = d0 + C (always > d0)
    #   ask = d0 - C (could be < 0)
    # If q < 0, C < 0.
    #   bid = d0 + C (could be < 0)
    #   ask = d0 - C (always > d0)
    
    # So:
    # bid = d0 + C
    # ask = d0 - C
    
    # Clip to 0
    bid_depths = np.clip(d0 + C, 0.0, np.inf)
    ask_depths = np.clip(d0 - C, 0.0, np.inf)
    
    # Boundary conditions:
    # No bid fill at q = Q.
    # No ask fill at q = -Q.
    # This is handled by the model's probability calculation (if q=Q, p_bid=0).
    # We don't need to set depth to inf, but we can if we want to be explicit.
    # However, the prompt says "np.inf means no quote".
    # If we quote at depth d, and it doesn't fill, we get 0 reward.
    # If we quote at inf, we get 0 reward.
    # So it's equivalent.
    # But to be safe and follow "no quote" if it's useless:
    # If inventory == Q, set bid_depths to inf?
    # If inventory == -Q, set ask_depths to inf?
    # Let's do that.
    
    bid_depths[inventory == Q] = np.inf
    ask_depths[inventory == -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
