"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed optimal policies
_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 280.0  # Safety margin before 330s kill

def _refine_depth(c, k, M, d_init, n_iter=30):
    """
    Refine the optimal depth using golden-section search around d_init.
    Maximize f(d) = (1 - exp(-c * exp(-k*d))) * (d + M) for d >= 0.
    """
    if M == 0 and c == 0:
        return 0.0
    
    def f(d):
        if d < 0:
            d = 0.0
        p = 1.0 - np.exp(-c * np.exp(-k * d))
        return p * (d + M)
    
    # Golden-section search in a window around d_init
    # The function is unimodal for d >= 0
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
    
    # Set initial bracket
    # f(0) = (1-exp(-c)) * M
    # f(inf) -> 0 (since exp(-kd) -> 0, so p -> 1, but... wait)
    # Actually as d -> inf, exp(-kd) -> 0, so p -> 1-exp(0) = 1, and (d+M) -> inf
    # So f(d) -> inf? That can't be right for a bounded problem.
    
    # Wait, let me recheck. P(d) = 1 - exp(-c * exp(-k*d))
    # As d -> inf, exp(-k*d) -> 0, so c*exp(-k*d) -> 0, so exp(-c*exp(-k*d)) -> 1, so P(d) -> 0.
    # As d -> 0, exp(-k*d) -> 1, so P(0) = 1 - exp(-c).
    # So P(d) decreases from 1-exp(-c) at d=0 to 0 as d->inf.
    # f(d) = P(d)*(d+M). At d=0: (1-exp(-c))*M. As d->inf: 0*inf = indeterminate.
    # Let's check the limit: P(d) ~ c*exp(-k*d) for large d, so f(d) ~ c*exp(-k*d)*(d+M) -> 0.
    # So f(d) is 0 at both ends (d=0 gives (1-exp(-c))*M which may be > 0, and d->inf gives 0).
    # Actually at d=0, f(0) = (1-exp(-c))*M. If M > 0, this is positive.
    # The function rises then falls. It's unimodal.
    
    # Use golden section search. We need a bracket [a, b] where the maximum is inside.
    # d=0 is a valid point. Let's find an upper bound.
    # f(d) > 0 requires P(d) > 0 and d+M > 0. Since d >= 0 and M >= 0 (typically), d+M >= 0.
    # P(d) > 0 for all finite d. But f(d) -> 0 as d -> inf.
    # Let's just search in [0, d_max] where d_max is large enough.
    
    # Actually, for the MDP, M = V[n+1, q±1] which is the expected future value.
    # M can be positive or negative. If M is negative, the optimal d might be at the boundary.
    
    # Let's do a simple approach: search in [0, d_max] with golden section.
    # d_max: where P(d) becomes very small. P(d) = 1-exp(-c*exp(-kd)).
    # For P(d) < 1e-10, we need c*exp(-kd) < 1e-10, so exp(-kd) < 1e-10/c, so kd > ln(c/1e-10).
    # d_max = (ln(c) + 23.0) / k (approx, since ln(1e10) ≈ 23)
    
    if c <= 0:
        return 0.0
    
    d_max = max(5.0/k, (np.log(max(c, 1e-15)) + 30.0) / k)
    d_max = min(d_max, 100.0)  # Cap to avoid numerical issues
    
    a = 0.0
    b = d_max
    
    # Golden section search
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1 = f(c1)
    f2 = f(c2)
    
    for _ in range(n_iter):
        if f1 < f2:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + gr * (b - a)
            f2 = f(c2)
        else:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - gr * (b - a)
            f1 = f(c1)
    
    return (a + b) / 2.0

def _solve_mdp(params):
    """
    Solves the finite-horizon MDP for the given parameters.
    Uses a coarse grid + local refinement for optimal depths.
    Returns arrays D_b and D_a of shape (steps, 2*Q + 1) containing optimal depths.
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
    
    # Value function V[n, q_index]
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depths D_b[n, q_index], D_a[n, q_index]
    D_b = np.full((steps, 2 * Q + 1), np.inf)
    D_a = np.full((steps, 2 * Q + 1), np.inf)
    
    # Coarse grid for initial search
    d_max_coarse = 5.0 / k
    n_coarse = 500
    d_grid = np.linspace(0, d_max_coarse, n_coarse)
    
    # Precompute probabilities for the grid
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-c * exp_term)
    
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fill remaining with 0 depths (conservative)
            for n2 in range(n, -1, -1):
                for q2 in range(-Q, Q + 1):
                    q_idx2 = q2 + Q
                    D_b[n2, q_idx2] = 0.0
                    D_a[n2, q_idx2] = 0.0
            break
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Cost
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Bid side
            if q < Q:
                M_b = V[n + 1, q + 1 + Q]
                # Coarse search
                obj_b = P_grid * (d_grid + M_b)
                idx_b = np.argmax(obj_b)
                d_b_coarse = d_grid[idx_b]
                # Refine
                d_b_opt = _refine_depth(c, k, M_b, d_b_coarse)
                # Ensure d_b_opt is valid
                if not np.isfinite(d_b_opt) or d_b_opt < 0:
                    d_b_opt = d_b_coarse
                val_b = (1.0 - np.exp(-c * np.exp(-k * d_b_opt))) * (d_b_opt + M_b)
            else:
                d_b_opt = np.inf
                val_b = 0.0
                
            # Ask side
            if q > -Q:
                M_a = V[n + 1, q - 1 + Q]
                # Coarse search
                obj_a = P_grid * (d_grid + M_a)
                idx_a = np.argmax(obj_a)
                d_a_coarse = d_grid[idx_a]
                # Refine
                d_a_opt = _refine_depth(c, k, M_a, d_a_coarse)
                # Ensure d_a_opt is valid
                if not np.isfinite(d_a_opt) or d_a_opt < 0:
                    d_a_opt = d_a_coarse
                val_a = (1.0 - np.exp(-c * np.exp(-k * d_a_opt))) * (d_a_opt + M_a)
            else:
                d_a_opt = np.inf
                val_a = 0.0
                
            D_b[n, q_idx] = d_b_opt
            D_a[n, q_idx] = d_a_opt
            
            V[n, q_idx] = val_b + val_a - cost
            
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a key for the cache
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Check if we have already solved this MDP
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_mdp(params)
    
    D_b, D_a = _POLICY_CACHE[key]
    
    # Determine current step n
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    # Process inventory
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Map inventory to index: q + Q
    inv_clipped = np.clip(inv, -Q, Q)
    q_indices = inv_clipped + Q
    
    # Retrieve depths
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    # Apply boundary constraints explicitly to be safe
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
