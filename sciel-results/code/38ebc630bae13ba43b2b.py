"""Optimal market making with inventory risk - Analytic HJB continuous-time solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0  # safety margin before 330s kill
_policy_cache = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the continuous-time HJB with quadratic value ansatz V(t,q) = alpha(t)*q^2 + beta(t)*q + gamma(t).
    
    The HJB equation:
      -V_t = (gamma/2)*sigma^2*q^2 + max_{b,a} [
          p_b*(1-p_a)*(b + V(q-1)) + p_a*(1-p_b)*(a + V(q+1)) + p_b*p_a*(b+a+V(q)) + (1-p_b)*(1-p_a)*V(q)
        ]
    
    With V(q) = alpha*q^2 + beta*q + gamma:
      V(q-1) = alpha*(q-1)^2 + beta*(q-1) + gamma = alpha*q^2 - 2*alpha*q + alpha + beta*q - beta + gamma
      V(q+1) = alpha*(q+1)^2 + beta*(q+1) + gamma = alpha*q^2 + 2*alpha*q + alpha + beta*q + beta + gamma
    
    The max over (b,a) of:
      p_b*(1-p_a)*b + p_a*(1-p_b)*a + p_b*p_a*(b+a)
      + p_b*(1-p_a)*V(q-1) + p_a*(1-p_b)*V(q+1) + (p_b*p_a + (1-p_b)*(1-p_a))*V(q)
    
    Simplify the depth earnings: b*p_b + a*p_a (as derived in the parent)
    
    The V terms:
      V(q-1) = alpha*q^2 + (beta-2*alpha)*q + (alpha - beta + gamma)
      V(q+1) = alpha*q^2 + (beta+2*alpha)*q + (alpha + beta + gamma)
      V(q)   = alpha*q^2 + beta*q + gamma
    
    Let's define:
      Vm1 = alpha*q^2 + (beta-2*alpha)*q + (alpha - beta + gamma)
      Vp1 = alpha*q^2 + (beta+2*alpha)*q + (alpha + beta + gamma)
      V0  = alpha*q^2 + beta*q + gamma
    
    The objective to maximize over (p_b, p_a) [where p_b = 1-exp(-A*exp(-k*b)*dt), etc.]:
      F = b*p_b + a*p_a 
        + p_b*(1-p_a)*Vm1 + p_a*(1-p_b)*Vp1 + (1-p_b-p_a+2*p_b*p_a)*V0
    
    For the continuous-time limit, dt is small. Let's work with the per-step formulation.
    
    Key insight: For a fixed p_b, the optimal p_a (and hence a) maximizes:
      p_a * [a + (1-p_b)*Vp1 - p_b*Vm1 + 2*p_b*V0 - V0]  -- this is getting complex.
    
    Let me use a different approach. Since the problem is separable in a useful way,
    and we have the quadratic ansatz, let's compute the optimal depths numerically 
    but efficiently using the structure.
    
    Actually, the cleanest approach for the quadratic ansatz:
    
    The value function at step n is V_n(q) = alpha_n * q^2 + beta_n * q + gamma_n.
    We solve backwards. At each step, given V_{n+1} (quadratic), we find the optimal
    (b,a) for each q, compute V_n(q), and fit a quadratic to it.
    
    But since V_n(q) will be quadratic in q if V_{n+1} is quadratic (by the structure),
    we can derive alpha_n, beta_n, gamma_n analytically or semi-analytically.
    
    Let's use the following approach:
    1. Start with V_{steps}(q) = 0 (alpha=beta=gamma=0)
    2. For each step backwards, for a range of q values, compute the optimal (b,a)
       using the FOC, evaluate V_n(q), and fit a quadratic.
    
    For the FOC: Given p_b, the optimal a maximizes:
      g(a) = p_a(a) * [a + C]  where C depends on p_b and V values.
    
    Let me derive this properly.
    
    E(b,a) = c + b*p_b + a*p_a + p_b*(1-p_a)*Vm1 + p_a*(1-p_b)*Vp1 + (1-p_b-p_a+2*p_b*p_a)*V0
    
    where c = (gamma/2)*sigma^2*q^2*dt
    
    Expand:
    = c + b*p_b + a*p_a + p_b*Vm1 - p_b*p_a*Vm1 + p_a*Vp1 - p_b*p_a*Vp1 + V0 - p_b*V0 - p_a*V0 + 2*p_b*p_a*V0
    
    = c + b*p_b + p_b*Vm1 + (1-p_b)*Vp1 + (1-p_b)*V0 - p_b*V0  [constants in p_a]
      + p_a * [a + Vp1 - V0 - p_b*Vm1 - p_b*Vp1 + 2*p_b*V0]
    
    Wait, let me redo this carefully:
    
    E = c + b*p_b + a*p_a 
      + p_b*Vm1 - p_b*p_a*Vm1 
      + p_a*Vp1 - p_b*p_a*Vp1 
      + V0 - p_b*V0 - p_a*V0 + 2*p_b*p_a*V0
    
    Terms not involving p_a:
      c + b*p_b + p_b*Vm1 + V0 - p_b*V0
    
    Terms linear in p_a:
      a*p_a + p_a*Vp1 - p_a*V0
    
    Terms with p_b*p_a:
      -p_b*p_a*Vm1 - p_b*p_a*Vp1 + 2*p_b*p_a*V0
      = p_b*p_a * (-Vm1 - Vp1 + 2*V0)
    
    So: E = [c + b*p_b + p_b*Vm1 + V0 - p_b*V0] 
           + p_a * [a + Vp1 - V0 + p_b*(-Vm1 - Vp1 + 2*V0)]
    
    For fixed b (fixed p_b), maximize over a:
      maximize p_a(a) * [a + D]  where D = Vp1 - V0 + p_b*(-Vm1 - Vp1 + 2*V0)
    
    Similarly, for fixed a (fixed p_a), maximize over b:
      maximize p_b(b) * [b + D']  where D' = Vm1 - V0 + p_a*(-Vm1 - Vp1 + 2*V0)
    
    The function h(x) = p(x) * (x + D) where p(x) = 1 - exp(-A*exp(-k*x)*dt)
    
    FOC: h'(x) = p'(x)*(x+D) + p(x) = 0
    
    p'(x) = -exp(-A*exp(-k*x)*dt) * A * exp(-k*x) * (-k) * dt
          = k * A * dt * exp(-k*x) * exp(-A*exp(-k*x)*dt)
    
    Let u = A*exp(-k*x)*dt, so p = 1-exp(-u), p' = k*exp(-u)*u/dt * dt = k*exp(-u)*u... 
    
    Actually: p(x) = 1 - exp(-A*exp(-k*x)*dt)
    dp/dx = -exp(-A*exp(-k*x)*dt) * A * exp(-k*x) * (-k) * dt
          = k * A * dt * exp(-k*x) * exp(-A*exp(-k*x)*dt)
    
    Let me define: for a given D, we need to find x >= 0 such that:
      k*A*dt*exp(-k*x)*exp(-A*exp(-k*x)*dt) * (x + D) + (1-exp(-A*exp(-k*x)*dt)) = 0
    
    This is a 1D equation in x. We can solve it with bisection.
    
    Given time constraints, let me use a practical hybrid:
    - Use the quadratic ansatz to get alpha, beta, gamma at each step
    - For each step, compute optimal depths for a few representative q values
    - Use these to determine the policy
    
    Actually, given the complexity, let me use a more direct approach:
    Solve the DP but with a much finer and smarter depth grid, fully vectorized.
    
    The key structural change: use the quadratic ansatz to precompute V(q) values
    analytically, then for each step solve the 2D optimization vectorized over all q.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # We'll solve backwards, maintaining the value function as a quadratic.
    # alpha, beta, gamma for V(q) = alpha*q^2 + beta*q + gamma
    alpha = 0.0
    beta = 0.0
    gamma_val = 0.0
    
    # Policy: for each step and each inventory level, store (bid_depth, ask_depth)
    # We'll compute this on the fly during the backward pass and store.
    policy = np.zeros((steps, n_inv, 2))
    
    # Depth grid for the 2D optimization
    max_depth = 8.0 / k
    n_depth = 60
    depth_grid = np.linspace(0, max_depth, n_depth)
    exp_term = np.exp(-k * depth_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Inventory values
    inv_vals = np.arange(-Q, Q + 1)  # shape (n_inv,)
    
    # Precompute for the quadratic:
    # Vm1(q) = alpha*(q-1)^2 + beta*(q-1) + gamma = alpha*q^2 + (beta-2*alpha)*q + (alpha-beta+gamma)
    # Vp1(q) = alpha*(q+1)^2 + beta*(q+1) + gamma = alpha*q^2 + (beta+2*alpha)*q + (alpha+beta+gamma)
    # V0(q)  = alpha*q^2 + beta*q + gamma
    
    # We need to handle the fact that at boundaries (q=Q, q=-Q), fills are restricted.
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 20:
            # Timeout: fill remaining with a simple heuristic
            base = 1.0 / k
            for nn in range(n, -1, -1):
                for ii in range(n_inv):
                    q = inv_vals[ii]
                    policy[nn, ii, 0] = base * (1 + 0.1 * q)
                    policy[nn, ii, 1] = base * (1 - 0.1 * q)
            return policy
        
        # Current value function coefficients
        # V(q) = alpha*q^2 + beta*q + gamma_val
        
        # Compute V0, Vm1, Vp1 for all q
        V0 = alpha * inv_vals**2 + beta * inv_vals + gamma_val
        Vm1 = alpha * (inv_vals - 1)**2 + beta * (inv_vals - 1) + gamma_val
        Vp1 = alpha * (inv_vals + 1)**2 + beta * (inv_vals + 1) + gamma_val
        
        # Boundary adjustments: at q=Q, Vm1 is not used (no bid fill)
        # At q=-Q, Vp1 is not used (no ask fill)
        # We'll handle this in the optimization
        
        cost = 0.5 * gamma * sigma**2 * dt
        c_vals = cost * inv_vals**2  # shape (n_inv,)
        
        # For each q, solve the 2D optimization
        # Vectorize over q: for each q, we have V0_q, Vm1_q, Vp1_q
        # E[b_idx, a_idx, q_idx] = c_q + depth[b]*p_b[b] + depth[a]*p_a[a]
        #   + p_b[b]*(1-p_a[a])*Vm1_q + p_a[a]*(1-p_b[b])*Vp1_q
        #   + (1-p_b[b]-p_a[a]+2*p_b[b]*p_a[a])*V0_q
        
        # This is a 3D array of size (n_depth, n_depth, n_inv) = 60*60*61 = 219,600
        # That's very manageable!
        
        b_col = depth_grid[:, None, None]  # (n_depth, 1, 1)
        p_b_col = p_grid[:, None, None]
        a_row = depth_grid[None, :, None]  # (1, n_depth, 1)
        p_a_row = p_grid[None, :, None]
        
        V0_3d = V0[None, None, :]  # (1, 1, n_inv)
        Vm1_3d = Vm1[None, None, :]
        Vp1_3d = Vp1[None, None, :]
        c_3d = c_vals[None, None, :]
        
        # Compute E for all combinations
        E = (c_3d 
             + b_col * p_b_col + a_row * p_a_row
             + p_b_col * (1 - p_a_row) * Vm1_3d
             + p_a_row * (1 - p_b_col) * Vp1_3d
             + (1 - p_b_col - p_a_row + 2 * p_b_col * p_a_row) * V0_3d)
        
        # Boundary handling:
        # At q = Q (last index), no bid fill: set p_b = 0 for all b
        # At q = -Q (first index), no ask fill: set p_a = 0 for all a
        
        # For q = Q: only ask can fill
        # E = c + a*p_a + p_a*Vp1 + (1-p_a)*V0 = c + V0 + p_a*(a + Vp1 - V0)
        # Maximize over a: p_a*(a + Vp1 - V0)
        # Set b = 0 (or any value, doesn't matter since p_b effectively 0)
        
        # For q = -Q: only bid can fill
        # E = c + b*p_b + p_b*Vm1 + (1-p_b)*V0 = c + V0 + p_b*(b + Vm1 - V0)
        # Maximize over b: p_b*(b + Vm1 - V0)
        
        # Let's handle boundaries separately
        # Interior: all q from 1 to n_inv-2 (i.e., -Q+1 to Q-1)
        
        # For interior states, find argmax over (b_idx, a_idx)
        # E shape: (n_depth, n_depth, n_inv)
        
        # For interior (indices 1 to n_inv-2):
        if n_inv > 2:
            E_int = E[:, :, 1:-1]  # (n_depth, n_depth, n_inv-2)
            flat_idx = np.argmax(E_int, axis=(0, 1))  # (n_inv-2,)
            best_b_int = flat_idx // n_depth
            best_a_int = flat_idx % n_depth
            policy[n, 1:-1, 0] = depth_grid[best_b_int]
            policy[n, 1:-1, 1] = depth_grid[best_a_int]
        
        # For q = Q (last index, no bid):
        q_idx_last = n_inv - 1
        V0_l = V0[q_idx_last]
        Vp1_l = Vp1[q_idx_last]
        c_l = c_vals[q_idx_last]
        # Maximize p_a*(a + Vp1_l - V0_l)
        D_l = Vp1_l - V0_l
        vals_a = p_grid * (depth_grid + D_l)
        best_a_l = np.argmax(vals_a)
        policy[n, q_idx_last, 0] = 0.0  # no bid
        policy[n, q_idx_last, 1] = depth_grid[best_a_l]
        
        # For q = -Q (first index, no ask):
        q_idx_first = 0
        V0_f = V0[q_idx_first]
        Vm1_f = Vm1[q_idx_first]
        c_f = c_vals[q_idx_first]
        # Maximize p_b*(b + Vm1_f - V0_f)
        D_f = Vm1_f - V0_f
        vals_b = p_grid * (depth_grid + D_f)
        best_b_f = np.argmax(vals_b)
        policy[n, q_idx_first, 0] = depth_grid[best_b_f]
        policy[n, q_idx_first, 1] = 0.0  # no ask
        
        # Now update the value function: V_n(q) = E at the optimal (b,a)
        # For interior:
        if n_inv > 2:
            V_curr_int = E_int[best_b_int, best_a_int, np.arange(n_inv - 2)]
        else:
            V_curr_int = np.array([])
        
        # For boundaries:
        V_curr_first = c_f + V0_f + vals_b[best_b_f]
        V_curr_last = c_l + V0_l + vals_a[best_a_l]
        
        # Assemble V_curr
        V_curr = np.empty(n_inv)
        V_curr[0] = V_curr_first
        V_curr[-1] = V_curr_last
        if n_inv > 2:
            V_curr[1:-1] = V_curr_int
        
        # Fit quadratic to V_curr: V(q) = alpha_new*q^2 + beta_new*q + gamma_new
        # Use least squares
        q_mat = np.column_stack([inv_vals**2, inv_vals, np.ones(n_inv)])
        coeffs, _, _, _ = np.linalg.lstsq(q_mat, V_curr, rcond=None)
        alpha = coeffs[0]
        beta = coeffs[1]
        gamma_val = coeffs[2]
    
    return policy


def quote(t, inventory, mid, params):
    """Quote bid and ask depths based on solved policy."""
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps

    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    cache_key = (sigma, A, k, gamma, T, steps, Q)

    if cache_key not in _policy_cache:
        if time.time() > _DEADLINE - 30:
            # Fallback heuristic
            base_depth = 1.0 / k
            alpha = 0.05 * gamma * sigma**2 / (A * k)
            bid_depths = np.clip(base_depth * (1 + alpha * inventory), 0.0, 8.0 / k)
            ask_depths = np.clip(base_depth * (1 - alpha * inventory), 0.0, 8.0 / k)
            return bid_depths, ask_depths

        policy = _solve_regime(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[cache_key] = policy

    policy = _policy_cache[cache_key]

    inv_clipped = np.clip(inventory, -Q, Q)
    inv_idx = inv_clipped + Q

    bid_depths = policy[n, inv_idx, 0].copy()
    ask_depths = policy[n, inv_idx, 1].copy()

    # Handle boundary: no bid at q=Q, no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf

    # Ensure valid values
    bid_depths = np.clip(bid_depths, 0.0, 8.0 / k)
    ask_depths = np.clip(ask_depths, 0.0, 8.0 / k)
    # Re-apply boundary (clip might have changed inf)
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
