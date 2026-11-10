"""Combined program: MDP with Newton-Raphson depth optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill

_POLICY_CACHE = {}

def _solve_mdp(params):
    """
    Solves the finite-horizon MDP for the given parameters.
    Returns arrays D_b and D_a of shape (steps, 2*Q + 1) containing optimal depths.
    Uses Newton-Raphson to find the optimal depth for each state, avoiding grid search.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    c = A * dt  # Constant in the exponent for fill probability
    
    n_inv = 2 * Q + 1
    
    # Value function V[n, q_index]
    # q_index = q + Q
    V = np.zeros((steps + 1, n_inv))
    
    # Optimal depths D_b[n, q_index], D_a[n, q_index]
    D_b = np.full((steps, n_inv), np.inf)
    D_a = np.full((steps, n_inv), np.inf)
    
    inv_cost_coeff = (gamma / 2.0) * sigma**2 * dt
    
    # Newton-Raphson solver for optimal depth d given marginal value M
    # Maximize f(d) = (1 - exp(-c * exp(-k*d))) * (d + M)
    # f'(d) = k * c * exp(-k*d) * exp(-c * exp(-k*d)) * (d + M) + 1 - exp(-c * exp(-k*d))
    # We solve f'(d) = 0.
    
    def find_optimal_depth(M, d_init=0.1):
        """
        Finds the optimal depth d >= 0 maximizing f(d) given M.
        Returns d_opt and val_opt.
        """
        # If M is very large positive, we want d -> 0 (high prob, low depth cost relative to gain?)
        # Actually if M is large positive, d+M is large. We want high p. d->0 gives max p.
        # If M is very large negative, we want d -> inf (low prob, avoid loss).
        
        # Check boundaries
        # f(0) = (1 - exp(-c)) * M
        # f(inf) -> 0
        
        # If M <= 0, the optimal d might be 0 if f(0) > 0? 
        # If M < 0, d+M can be negative. 
        # f(d) = p(d)(d+M). 
        # If d < -M, f(d) < 0. Since f(0) = p(0)M < 0, and f(inf)=0, the max is 0 (at d=inf or where d+M=0 if p>0).
        # But we can choose d=inf (no quote). Value is 0.
        # So if M < 0, we should check if any d gives positive value.
        # p(d) > 0 always. d+M > 0 requires d > -M.
        # If we set d slightly above -M, value is small positive.
        # However, usually in these problems M is positive near the end and can be negative.
        # If M < 0, the best strategy is often to not quote (d=inf, val=0) or quote very far.
        # Let's assume the optimal d is where f'(d)=0.
        
        # Derivative:
        # h(d) = f'(d) = k * c * exp(-k*d) * exp(-c * exp(-k*d)) * (d + M) + 1 - exp(-c * exp(-k*d))
        
        # Newton's method
        d = d_init
        # If M is negative, start further out?
        if M < 0:
            d = max(0.0, -M + 1.0/k) # Start where d+M > 0
            
        for _ in range(10):
            exp_kd = np.exp(-k * d)
            inner_exp = np.exp(-c * exp_kd)
            
            # f'(d)
            # term1 = k * c * exp_kd * inner_exp * (d + M)
            # term2 = 1 - inner_exp
            # h = term1 + term2
            
            # h'(d)
            # Let u = exp(-kd), v = exp(-c*u)
            # h = k c u v (d+M) + 1 - v
            # dh/du = k c (v + u * dv/du) (d+M) - dv/du
            # dv/du = -c * exp(-c*u) * (-1) = c * v
            # So dv/du = c v
            # dh/du = k c (v + u c v) (d+M) - c v = k c v (1 + c u) (d+M) - c v
            # du/dd = -k u
            # h'(d) = dh/du * du/dd = -k u [ k c v (1 + c u) (d+M) - c v ]
            # h'(d) = -k u c v [ k (1 + c u) (d+M) - 1 ]
            
            v = inner_exp
            u = exp_kd
            
            h = k * c * u * v * (d + M) + 1.0 - v
            h_prime = -k * u * c * v * (k * (1.0 + c * u) * (d + M) - 1.0)
            
            if abs(h_prime) < 1e-12:
                break
                
            step = h / h_prime
            d_new = d - step
            
            # Ensure d >= 0
            if d_new < 0:
                d_new = 0.0
                
            if abs(d_new - d) < 1e-8:
                d = d_new
                break
            d = d_new
            
        # Calculate final value
        exp_kd = np.exp(-k * d)
        p = 1.0 - np.exp(-c * exp_kd)
        val = p * (d + M)
        
        # Ensure we don't return a negative value if 0 is better (no quote)
        if val < 0:
             # If the max found is negative, it's better to not quote (val=0, d=inf)
             # But wait, if M is very negative, d+M is negative. 
             # If we can't make it positive, we should return inf, 0.
             # However, the MDP will add this to V. If val < 0, it's better to set d=inf, val=0.
             # So we check if val < 0.
             return np.inf, 0.0
             
        return d, val

    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
            # Fallback: fill remaining with a heuristic depth
            # Use a simple depth like 1/k
            for q_idx in range(n_inv):
                D_b[n, q_idx] = 1.0 / k
                D_a[n, q_idx] = 1.0 / k
            break
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            inv_cost = -inv_cost_coeff * q**2
            
            # Bid side
            if q < Q:
                M_b = V[n + 1, q_idx + 1]
                d_b_opt, val_b = find_optimal_depth(M_b)
            else:
                d_b_opt = np.inf
                val_b = 0.0
                
            # Ask side
            if q > -Q:
                M_a = V[n + 1, q_idx - 1]
                d_a_opt, val_a = find_optimal_depth(M_a)
            else:
                d_a_opt = np.inf
                val_a = 0.0
                
            D_b[n, q_idx] = d_b_opt
            D_a[n, q_idx] = d_a_opt
            
            V[n, q_idx] = val_b + val_a + inv_cost
            
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_mdp(params)
        
    D_b, D_a = _POLICY_CACHE[key]
    
    dt = T / steps
    n = int(round(t / dt))
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    inv = np.asarray(inventory, dtype=np.int64)
    q_indices = inv + Q
    
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    # Boundary constraints
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    # Ensure validity
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.0, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
