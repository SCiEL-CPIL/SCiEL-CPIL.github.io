import numpy as np
import time
from scipy.optimize import brentq

_START = time.time()
_DEADLINE = _START + 290.0
_cache = {}

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # V[t, q_index]
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    # Precompute costs
    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Helper to solve for x given shift S and p_other
    # Equation: e^{-A x dt} [ A x ln(x) - A k x (S) + 1 ] = 1
    # Let F(x) = e^{-A x dt} [ A x ln(x) - A k x S + 1 ] - 1
    # We want F(x) = 0.
    # x in (0, 1].
    # As x -> 0, ln(x) -> -inf, x ln(x) -> 0. Term in bracket -> 1. e^0 -> 1. F(0) = 0?
    # Limit x->0: A x ln x -> 0. Bracket -> 1. e^0 -> 1. F(0) = 1*1 - 1 = 0.
    # So x=0 is a root?
    # Let's check the derivative or behavior.
    # If x is very small, p_b is small.
    # If S is large positive (want to buy), we want large p_b, so small x? No, small x means large b, small p_b.
    # Wait, x = exp(-k b). Large b -> small x.
    # If we want to buy, we want high p_b? No, we want to maximize p_b (b + shift).
    # If shift is positive, we want to increase p_b.
    # Let's check the sign of F(x).
    # At x=1: e^{-A dt} [ 0 - A k S + 1 ] - 1.
    # If S is large, this is negative.
    # If x is small, F(x) is close to 0.
    # Is there a root in (0, 1)?
    # Let's test numerically for a specific case.
    
    def find_x(S, p_other, D_cross):
        # S = delta_v + p_other * D_cross
        # Actually the shift in the derivative condition is (b + delta_v + p_other * D_cross)
        # The term in the bracket was derived as:
        # A x e^{-A x dt} (ln x - k (delta_v + p_other D_cross)) + 1 - e^{-A x dt} = 0
        # Let Shift = delta_v + p_other * D_cross
        # F(x) = e^{-A x dt} ( A x ln x - A k x Shift + 1 ) - 1
        
        if S == 0:
            # If Shift is 0, we maximize p_b * b.
            # Optimal x is where derivative of (1-e^{-ax})(-1/k ln x) is 0.
            # This is a standard case.
            pass
        
        def F(x):
            if x <= 0: return 0.0
            if x > 1: return -1.0 # Should not happen
            term = A * x * np.log(x) - A * k * x * S + 1.0
            return np.exp(-A * x * dt) * term - 1.0
            
        # We know x=0 is a limit point.
        # Let's search in [1e-12, 1.0]
        try:
            # Check if max is at boundary x=1 (b=0)
            val_1 = F(1.0)
            val_0 = F(1e-12)
            
            # If F(1) > 0 and F(0) < 0, there is a root.
            # Actually, let's just find the root.
            # If no root, the max is at boundary.
            # The function p(x)(b(x)+S) is unimodal.
            # We can just use brentq on the derivative of the objective?
            # No, F(x) IS the derivative condition (set to 0).
            
            # If F(1) and F(1e-12) have same sign, no root in interior.
            # Max is at boundary.
            # If F(1) > 0, slope is positive at 1?
            # Let's check slope of Objective J(x) = (1-e^{-ax})(-1/k ln x + S)
            # J'(x) proportional to F(x) * (positive factor)?
            # Let's verify the sign.
            # J'(x) = a e^{-ax} (-1/k ln x + S) + (1-e^{-ax}) (-1/kx)
            # = e^{-ax} [ a(-1/k ln x + S) - 1/kx (1-e^{-ax}) ]
            # = e^{-ax} [ -a/k ln x + aS - 1/kx + 1/kx e^{-ax} ]
            # Multiply by kx:
            # kx J'(x) e^{ax} = -a x ln x + a k x S - 1 + e^{-ax}
            # = -( a x ln x - a k x S + 1 - e^{-ax} )
            # = -( a x ln x - a k x S + 1 ) + e^{-ax}
            # Set to 0:
            # a x ln x - a k x S + 1 = e^{ax}
            # (a x ln x - a k x S + 1) e^{-ax} = 1
            # This matches F(x) = 0.
            
            # So if F(1) > 0, then J'(1) < 0?
            # kx J' e^{ax} = - (F(x) + 1 - e^{-ax} + e^{-ax})? No.
            # J'(x) = 0 iff F(x) = 0.
            # If F(1) > 0, then (term) > 1, so e^{-ax}(term) > 1? No.
            # F(1) = e^{-A dt} ( -A k S + 1 ) - 1.
            # If F(1) > 0, then e^{-A dt} (1 - A k S) > 1.
            # This implies 1 - A k S is large positive.
            # This happens if S is very negative (we want to sell, so b should be small? No, if S is negative, we want to avoid bid fills? Or we want bid fills to be cheap?)
            # If S is negative, we want small b (small depth) or no bid?
            # If S < 0, the term (b+S) is negative for small b.
            # We want to maximize p(b)(b+S).
            # If S is very negative, we might want b=0? Or b=inf?
            # If b=0, value is 0.
            # If b=inf, value is 0.
            # If there is a max in between, it must be positive.
            
            # Let's just use brentq if signs differ.
            if val_1 * val_0 < 0:
                x_opt = brentq(F, 1e-12, 1.0, xtol=1e-12)
            else:
                # Check boundaries
                # Calculate J at x=1 and x=1e-12
                def J(x):
                    if x <= 0: return 0.0
                    p = 1.0 - np.exp(-A * x * dt)
                    b = -np.log(x) / k
                    return p * (b + S)
                
                J_1 = J(1.0)
                J_0 = J(1e-12)
                if J_1 > J_0:
                    x_opt = 1.0
                else:
                    x_opt = 1e-12
            return x_opt
        except Exception:
            return 1.0 # Fallback

    # Iteration parameters
    n_iter = 10
    
    for t in range(N - 1, -1, -1):
        V_next = V[t + 1]
        
        # Initialize
        p_b_curr = np.zeros(n_q)
        p_a_curr = np.zeros(n_q)
        b_curr = np.full(n_q, np.inf)
        a_curr = np.full(n_q, np.inf)
        
        dV_b = np.zeros(n_q)
        dV_a = np.zeros(n_q)
        D_cross = np.zeros(n_q)
        
        if n_q > 1:
            dV_b[:-1] = V_next[1:] - V_next[:-1]
            dV_a[1:] = V_next[:-1] - V_next[1:]
        
        if n_q > 2:
            D_cross[1:-1] = V_next[2:] + V_next[:-2] - 2 * V_next[1:-1]
            
        for _ in range(n_iter):
            # Solve for b (x_b)
            # Shift_b = dV_b + p_a_curr * D_cross
            shift_b = dV_b + p_a_curr * D_cross
            # Solve for a (x_a)
            # Shift_a = dV_a + p_b_curr * D_cross
            shift_a = dV_a + p_b_curr * D_cross
            
            x_b_arr = np.zeros(n_q)
            x_a_arr = np.zeros(n_q)
            
            for i in range(n_q):
                if i == 2 * Q: # q=Q, no bid
                    x_b_arr[i] = 0.0
                else:
                    x_b_arr[i] = find_x(shift_b[i], p_a_curr[i], D_cross[i])
                    
                if i == 0: # q=-Q, no ask
                    x_a_arr[i] = 0.0
                else:
                    x_a_arr[i] = find_x(shift_a[i], p_b_curr[i], D_cross[i])
            
            p_b_curr = 1.0 - np.exp(-A * dt * x_b_arr)
            p_a_curr = 1.0 - np.exp(-A * dt * x_a_arr)
            
            b_curr = -np.log(x_b_arr) / k
            b_curr[x_b_arr == 0.0] = np.inf
            
            a_curr = -np.log(x_a_arr) / k
            a_curr[x_a_arr == 0.0] = np.inf
            
            # Force boundaries
            b_curr[2 * Q] = np.inf
            p_b_curr[2 * Q] = 0.0
            a_curr[0] = np.inf
            p_a_curr[0] = 0.0
            
        # Store actions
        opt_b[t, :] = b_curr
        opt_a[t, :] = a_curr
        
        # Update V
        # J = p_b (b + dV_b + p_a D_cross) + p_a (a + dV_a)
        # Note: The term p_b p_a D_cross is included in the b-part?
        # J = p_b b + p_a a + p_b dV_b + p_a dV_a + p_b p_a D_cross
        #   = p_b (b + dV_b + p_a D_cross) + p_a (a + dV_a)
        # Yes.
        
        contrib_b = p_b_curr * (b_curr + dV_b + p_a_curr * D_cross)
        contrib_b = np.where(np.isinf(b_curr), 0.0, contrib_b)
        
        contrib_a = p_a_curr * (a_curr + dV_a)
        contrib_a = np.where(np.isinf(a_curr), 0.0, contrib_a)
        
        V[t, :] = V_next + cost + contrib_b + contrib_a

    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        return np.full(4000, np.inf), np.full(4000, np.inf)
        
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs]
    ask_depths = opt_a[t_idx, idxs]
    
    # Ensure validity
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
