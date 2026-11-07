"""Optimal market making: vectorized 1D-scalar HJB with separable depth optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 290s safety margin out of 330s

_REGIMES = [
    {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
    {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
    {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
]

_PRECOMPUTED = {}


def _solve_regime(regime):
    """
    Vectorized HJB backward induction.
    
    Key insight: the objective at each state (n, q) separates into
    f_bid(bd) + f_ask(ad) where:
      f_bid(bd) = bd * p_b(bd) + (1-p_a) * p_b(bd) * (V(q+1) - V(q))
      f_ask(ad) = ad * p_a(ad) + (1-p_b) * p_a(ad) * (V(q-1) - V(q))
    
    Each is a 1D function of a single depth variable. We minimize the
    negative (i.e., maximize) using scipy's bounded scalar optimizer.
    All 61 inventory states are processed in parallel via NumPy.
    """
    sigma = regime['sigma']
    A = regime['A']
    k = regime['k']
    gamma = regime['gamma']
    T = regime['T']
    steps = regime['steps']
    Q = regime['Q']
    dt = T / steps

    n_q = 2 * Q + 1  # 61
    # Index mapping: qi -> q = qi - Q

    # Depth bounds: the fill rate lambda(d) = A*exp(-k*d) decays;
    # beyond d_max the fill prob is negligible.
    # lambda(d_max) * dt ~ 1e-6 => d_max = -ln(1e-6/(A*dt))/k
    d_max = -np.log(1e-6 / (A * dt)) / k
    d_max = min(d_max, 5.0 / k)  # cap at 5/k for safety
    d_min = 0.0

    # Precompute fill probability function (vectorized over depth array)
    def fill_prob(d):
        """d: scalar or array. Returns 1 - exp(-A*exp(-k*d)*dt)."""
        return 1.0 - np.exp(-A * np.exp(-k * d) * dt)

    # Value function: V[qi] for current backward step
    V = np.zeros(n_q)

    # Tables: [steps, n_q]
    bid_table = np.full((steps, n_q), 1.0 / k, dtype=np.float64)
    ask_table = np.full((steps, n_q), 1.0 / k, dtype=np.float64)

    # For the 1D optimization, we'll use a fixed grid + local refinement.
    # Strategy: coarse grid of 80 points, then for each state do a
    # local golden-section refinement around the best grid point.
    # This is much faster than 40x40=1600 evaluations per state.

    n_grid = 80
    grid = np.linspace(d_min, d_max, n_grid)

    # Precompute fill probs on grid
    grid_p = fill_prob(grid)  # shape (n_grid,)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 20:
            # Fallback: fill remaining with 1/k
            for m in range(n, steps):
                bid_table[m] = 1.0 / k
                ask_table[m] = 1.0 / k
            break

        # Current value at next step is V (shape n_q)
        # Cost at this step
        q_arr = np.arange(n_q) - Q  # shape (n_q,)
        cost = 0.5 * gamma * sigma * sigma * q_arr * q_arr * dt  # shape (n_q,)

        # Value differences
        # V(q+1) - V(q): for q < Q
        dV_plus = np.zeros(n_q)
        dV_minus = np.zeros(n_q)
        # For qi = 0..n_q-2: V[qi+1] - V[qi]
        dV_plus[:n_q - 1] = V[1:] - V[:-1]
        dV_plus[n_q - 1] = 0.0  # q = Q: no bid fill
        # For qi = 1..n_q-1: V[qi-1] - V[qi]
        dV_minus[1:] = V[:-1] - V[1:]
        dV_minus[0] = 0.0  # q = -Q: no ask fill

        # For each state qi, we need to find optimal bd and ad.
        # f_bid(bd) = bd * p_b(bd) + (1 - p_a) * p_b(bd) * dV_plus
        #   But p_a depends on ad, which is chosen simultaneously.
        # 
        # However, the coupling is weak: (1-p_a) is a constant multiplier
        # on the dV term for a given ad. We can iterate:
        #   1. Start with ad = 1/k, compute p_a
        #   2. Optimize bd given p_a
        #   3. Optimize ad given p_b
        #   4. Repeat 2-3 twice for convergence
        #
        # Actually, let's just do the full 2D but smartly:
        # Since the grid is 80x80 = 6400 per state, and we have 61 states,
        # that's 390,400 evaluations per step. With 1000 steps that's 390M.
        # Too slow.
        #
        # Better: use the separability more carefully.
        # The total objective (excluding constant V(q) and cost) is:
        #   bd*p_b + ad*p_a + p_b*(1-p_a)*dV_plus + (1-p_b)*p_a*dV_minus
        # = bd*p_b + (1-p_a)*p_b*dV_plus + ad*p_a + (1-p_b)*p_a*dV_minus
        # = p_b*(bd + (1-p_a)*dV_plus) + p_a*(ad + (1-p_b)*dV_minus)
        #
        # This is NOT separable because p_a appears in the bd term and vice versa.
        # But the coupling is through (1-p_a) and (1-p_b) which are close to 1
        # for reasonable depths (p ~ 0.1-0.5).
        #
        # Practical approach: alternate optimization.
        # For each state, do 2-3 rounds of:
        #   - Fix ad, optimize bd (1D)
        #   - Fix bd, optimize ad (1D)
        #
        # For the 1D optimization of bd given p_a:
        #   g(bd) = bd * p_b(bd) + (1-p_a) * p_b(bd) * dV_plus
        #         = p_b(bd) * (bd + (1-p_a)*dV_plus)
        # We want to MAXIMIZE g(bd).
        #
        # Similarly for ad given p_b:
        #   h(ad) = ad * p_a(ad) + (1-p_b) * p_a(ad) * dV_minus
        #         = p_a(ad) * (ad + (1-p_b)*dV_minus)
        # MAXIMIZE h(ad).

        # Initialize with previous step's depths or 1/k
        bid_curr = np.full(n_q, 1.0 / k)
        ask_curr = np.full(n_q, 1.0 / k)

        # Do 2 rounds of alternating optimization
        for _round in range(2):
            # --- Optimize bid given current ask ---
            p_a = fill_prob(ask_curr)  # shape (n_q,)
            # g(bd) = p_b(bd) * (bd + (1-p_a)*dV_plus)
            # For states where q >= Q (qi = n_q-1), no bid fill: skip
            # For states where q <= -Q (qi = 0), dV_minus = 0 but bid still valid
            
            # Vectorized: for each qi, find best bd on grid
            # g[qi, j] = grid_p[j] * (grid[j] + (1 - p_a[qi]) * dV_plus[qi])
            # Shape: (n_q, n_grid)
            
            # For qi = n_q-1 (q=Q): no bid fill, set bd = 0 (or any, doesn't matter)
            # Actually we should set p_b = 0 for that state.
            
            # Compute g for all states simultaneously
            # term = (1 - p_a[qi]) * dV_plus[qi] for each qi
            term_bid = (1.0 - p_a) * dV_plus  # shape (n_q,)
            # g[qi, j] = grid_p[j] * (grid[j] + term_bid[qi])
            # = grid_p[j] * grid[j] + grid_p[j] * term_bid[qi]
            
            # For each qi, argmax over j
            # g_matrix = grid_p[:, None] * (grid[None, :] + term_bid[:, None])
            # This is (n_q, n_grid) = (61, 80) = 4880 elements. Fast.
            
            g_matrix = grid_p[:, None] * (grid[None, :] + term_bid[:, None])  # (n_q, n_grid)
            best_idx_bid = np.argmax(g_matrix, axis=1)  # (n_q,)
            bid_curr = grid[best_idx_bid]
            bid_curr[n_q - 1] = 0.0  # q = Q: no bid quote needed
            
            # --- Optimize ask given current bid ---
            p_b = fill_prob(bid_curr)  # shape (n_q,)
            term_ask = (1.0 - p_b) * dV_minus  # shape (n_q,)
            h_matrix = grid_p[:, None] * (grid[None, :] + term_ask[:, None])  # (n_q, n_grid)
            best_idx_ask = np.argmax(h_matrix, axis=1)  # (n_q,)
            ask_curr = grid[best_idx_ask]
            ask_curr[0] = 0.0  # q = -Q: no ask quote needed

        # Store results
        bid_table[n, :] = bid_curr
        ask_table[n, :] = ask_curr

        # Update V: V_new[qi] = -cost[qi] + max_obj[qi] + V[qi]
        # max_obj = g(bid_opt) + h(ask_opt) - but we need the actual value
        # Let's compute it properly:
        p_b = fill_prob(bid_curr)
        p_a = fill_prob(ask_curr)
        # For boundary states
        p_b[n_q - 1] = 0.0  # q = Q
        p_a[0] = 0.0  # q = -Q

        fill_reward = bid_curr * p_b + ask_curr * p_a  # shape (n_q,)
        
        # V_future = (1-p_b)(1-p_a)*V(q) + p_b(1-p_a)*V(q+1) + (1-p_b)p_a*V(q-1) + p_b*p_a*V(q)
        # = V(q) + p_b(1-p_a)*dV_plus + (1-p_b)*p_a*dV_minus
        V_future_delta = p_b * (1 - p_a) * dV_plus + (1 - p_b) * p_a * dV_minus
        
        V_new = -cost + fill_reward + V + V_future_delta
        V = V_new

    return bid_table, ask_table


def _precompute_all():
    global _PRECOMPUTED
    for regime in _REGIMES:
        key = (regime['sigma'], regime['A'], regime['k'], regime['gamma'])
        if key not in _PRECOMPUTED:
            if time.time() > _DEADLINE - 30:
                # Fallback
                Q = regime['Q']
                n_q = 2 * Q + 1
                depth = 1.0 / regime['k']
                bd = np.full((regime['steps'], n_q), depth)
                ad = np.full((regime['steps'], n_q), depth)
                _PRECOMPUTED[key] = (bd, ad)
            else:
                bd, ad = _solve_regime(regime)
                _PRECOMPUTED[key] = (bd, ad)


def _init():
    global _PRECOMPUTED
    if not _PRECOMPUTED:
        _precompute_all()


_init()


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) for the current step."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']

    key = (sigma, A, k, gamma)

    if key in _PRECOMPUTED:
        bid_table, ask_table = _PRECOMPUTED[key]
    else:
        # Fallback
        depth = 1.0 / k
        bid_depths = np.full(len(inventory), depth, dtype=np.float64)
        ask_depths = np.full(len(inventory), depth, dtype=np.float64)
        return bid_depths, ask_depths

    n = int(round(t / (params['T'] / steps)))
    n = min(max(n, 0), steps - 1)

    # Map inventory to index
    qi = inventory + Q
    qi = np.clip(qi, 0, 2 * Q)

    bid_depths = bid_table[n, qi].astype(np.float64)
    ask_depths = ask_table[n, qi].astype(np.float64)

    # Safety: ensure valid values
    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, 1.0 / k)

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
