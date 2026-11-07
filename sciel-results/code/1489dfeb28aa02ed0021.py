"""
Strategy: Pre-compute the exact optimal policy using Dynamic Programming (Bellman Equation).
We solve for the optimal bid/ask depths for each (t, q) state.
Key improvements:
1. Analytic/iterative refinement for optimal depths to avoid coarse grid artifacts.
2. Vectorized DP computation over all inventory states simultaneously where possible.
3. Efficient deadline guard.
4. Handling of boundary conditions (q=Q, q=-Q) correctly.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
_policy_cache = {}

# Deadline guard
_start_time = time.time()
DEADLINE_MARGIN = 10.0  # seconds

def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all (t, q) states using DP.
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    # V[t, q] is the maximum expected cumulative reward from time t onwards.
    # We compute V backwards from T to 0.
    # V[steps, q] = 0 for all q (no terminal charge)
    
    # We need to store the optimal actions for each (t, q)
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # Current value function (starts at terminal)
    V_next = np.zeros(n_states)
    
    # Search space for depths.
    # p(d) = 1 - exp(-A exp(-k d) dt)
    # We want to maximize E[b, a].
    # The function is concave in d? Let's assume unimodal and use a refined search.
    # Initial grid: [0, 3.0] with step 0.01? 
    # 3.0 is large enough for k=0.75 (exp(-2.25) is small).
    # Let's use a coarse grid to find the neighborhood, then refine.
    
    # Coarse grid
    max_depth = 3.0
    n_coarse = 301  # step 0.01
    coarse_grid = np.linspace(0, max_depth, n_coarse)
    
    # Precompute probabilities for coarse grid
    # p = 1 - exp(-A exp(-k d) dt)
    exp_term_coarse = np.exp(-k * coarse_grid)
    p_coarse = 1.0 - np.exp(-A * exp_term_coarse * dt)
    
    # V_next is shape (n_states,)
    # We will iterate t from steps-1 down to 0.
    
    for t in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > 330.0 - DEADLINE_MARGIN:
            # Fill remaining with default values
            opt_bid[t:, :] = 1.0 / k
            opt_ask[t:, :] = 1.0 / k
            break
        
        V_current = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_values[q_idx]
            
            allow_bid = (q < Q)
            allow_ask = (q > -Q)
            
            v_q = V_next[q_idx]
            
            if allow_bid:
                v_qp = V_next[q_idx + 1]
            else:
                v_qp = 0.0
            
            if allow_ask:
                v_qm = V_next[q_idx - 1]
            else:
                v_qm = 0.0
            
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # We want to maximize E(b, a) - cost
            # E(b, a) = (1-pb)(1-pa)v_q + pb(1-pa)(b+v_qp) + (1-pb)pa(a+v_qm) + pb pa (b+a+v_q)
            
            # Simplify E:
            # Let's try to decouple b and a if possible?
            # E = v_q - pb v_q - pa v_q + pb pa v_q
            #   + b pb - b pb pa + v_qp pb - v_qp pb pa
            #   + a pa - a pb pa + v_qm pa - v_qm pb pa
            #   + b pb pa + a pb pa + v_q pb pa
            #
            # Group by b:
            # b [ pb - pb pa + pb pa ] = b pb
            # Group by a:
            # a [ pa - pb pa + pb pa ] = a pa
            #
            # Constant terms (w.r.t b, a):
            # v_q (1 - pb - pa + pb pa)
            # v_qp (pb - pb pa) = pb (1-pa) v_qp
            # v_qm (pa - pb pa) = pa (1-pb) v_qm
            # v_q (pb pa - pb pa) = 0?
            # Wait, let's re-expand carefully.
            #
            # Term 1: (1-pb)(1-pa)v_q = (1 - pb - pa + pb pa) v_q
            # Term 2: pb(1-pa)(b + v_qp) = (pb - pb pa) b + (pb - pb pa) v_qp
            # Term 3: (1-pb)pa(a + v_qm) = (pa - pb pa) a + (pa - pb pa) v_qm
            # Term 4: pb pa (b + a + v_q) = pb pa b + pb pa a + pb pa v_q
            #
            # Sum:
            # b terms: b(pb - pb pa) + b(pb pa) = b pb
            # a terms: a(pa - pb pa) + a(pb pa) = a pa
            #
            # v_q terms:
            # v_q (1 - pb - pa + pb pa) + v_q (pb pa) ? No.
            # From T1: v_q (1 - pb - pa + pb pa)
            # From T4: v_q (pb pa)
            # Total v_q: v_q (1 - pb - pa + 2 pb pa)
            
            # v_qp terms:
            # From T2: v_qp (pb - pb pa)
            
            # v_qm terms:
            # From T3: v_qm (pa - pb pa)
            
            # So E(b, a) = b pb(b) + a pa(a) + v_q (1 - pb(b) - pa(a) + 2 pb(b) pa(a)) 
            #             + v_qp (pb(b) - pb(b) pa(a)) 
            #             + v_qm (pa(a) - pb(b) pa(a))
            
            # This is NOT separable because of the cross terms pb pa.
            # However, we can iterate.
            # Fix a, optimize b. Fix b, optimize a.
            
            # Let's use the coarse grid to find initial b*, a*.
            # Then refine.
            
            # Vectorized over coarse grid
            pb_2d = p_coarse[:, None]
            pa_2d = p_coarse[None, :]
            b_2d = coarse_grid[:, None]
            a_2d = coarse_grid[None, :]
            
            # Masking
            if not allow_bid:
                pb_2d.fill(0.0)
            if not allow_ask:
                pa_2d.fill(0.0)
            
            # Compute E
            # E = b pb + a pa + v_q (1 - pb - pa + 2 pb pa) + v_qp (pb - pb pa) + v_qm (pa - pb pa)
            
            t1 = b_2d * pb_2d
            t2 = a_2d * pa_2d
            t3 = v_q * (1.0 - pb_2d - pa_2d + 2.0 * pb_2d * pa_2d)
            t4 = v_qp * (pb_2d - pb_2d * pa_2d)
            t5 = v_qm * (pa_2d - pb_2d * pa_2d)
            
            total = t1 + t2 + t3 + t4 + t5 - cost
            
            # Find max
            max_idx = np.argmax(total)
            i, j = np.unravel_index(max_idx, total.shape)
            
            b_init = coarse_grid[i]
            a_init = coarse_grid[j]
            
            # Refinement: Local search around b_init, a_init
            # Use a fine grid in [b_init - 0.05, b_init + 0.05] and [a_init - 0.05, a_init + 0.05]
            # Clip to [0, max_depth]
            
            b_low = max(0.0, b_init - 0.05)
            b_high = min(max_depth, b_init + 0.05)
            a_low = max(0.0, a_init - 0.05)
            a_high = min(max_depth, a_init + 0.05)
            
            n_fine = 51
            fine_grid_b = np.linspace(b_low, b_high, n_fine)
            fine_grid_a = np.linspace(a_low, a_high, n_fine)
            
            exp_term_b = np.exp(-k * fine_grid_b)
            p_fine_b = 1.0 - np.exp(-A * exp_term_b * dt)
            
            exp_term_a = np.exp(-k * fine_grid_a)
            p_fine_a = 1.0 - np.exp(-A * exp_term_a * dt)
            
            pb_fine_2d = p_fine_b[:, None]
            pa_fine_2d = p_fine_a[None, :]
            b_fine_2d = fine_grid_b[:, None]
            a_fine_2d = fine_grid_a[None, :]
            
            if not allow_bid:
                pb_fine_2d.fill(0.0)
            if not allow_ask:
                pa_fine_2d.fill(0.0)
            
            t1f = b_fine_2d * pb_fine_2d
            t2f = a_fine_2d * pa_fine_2d
            t3f = v_q * (1.0 - pb_fine_2d - pa_fine_2d + 2.0 * pb_fine_2d * pa_fine_2d)
            t4f = v_qp * (pb_fine_2d - pb_fine_2d * pa_fine_2d)
            t5f = v_qm * (pa_fine_2d - pb_fine_2d * pa_fine_2d)
            
            total_fine = t1f + t2f + t3f + t4f + t5f - cost
            
            max_idx_fine = np.argmax(total_fine)
            i_f, j_f = np.unravel_index(max_idx_fine, total_fine.shape)
            
            opt_bid[t, q_idx] = fine_grid_b[i_f]
            opt_ask[t, q_idx] = fine_grid_a[j_f]
            V_current[q_idx] = total_fine[i_f, j_f]
            
        V_next = V_current
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
    
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
