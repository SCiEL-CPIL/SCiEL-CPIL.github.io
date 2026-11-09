"""Optimal market making with inventory risk - DP-based quote policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for solved policies per regime
_policy_cache = {}
_start_time = time.time()
_DEADLINE = 320  # seconds, leave margin before 330s kill

def _solve_dp(params):
    """
    Solve the discrete DP for the given regime.
    Returns a policy table: policy[step, inventory] = (bid_depth, ask_depth)
    or None if timeout.
    """
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps

    # Time budget check
    if time.time() - _start_time > _DEADLINE - 30:
        return None

    # Inventory range: -Q to Q
    inv_range = np.arange(-Q, Q + 1)
    n_inv = len(inv_range)
    inv_idx = {v: i for i, v in enumerate(inv_range)}

    # We need to solve backwards from step = steps-1 down to 0.
    # Value function V_n(q) = max expected future reward from step n onwards given inventory q.
    # At each step, we choose bid_depth b >= 0 and ask_depth a >= 0.
    # 
    # The reward structure:
    # - If bid fills: earn b (the depth), inventory decreases by 1 (q -> q-1)
    # - If ask fills: earn a, inventory increases by 1 (q -> q+1)
    # - Both can fill independently
    # - Cost: (gamma/2) * sigma^2 * q^2 * dt (inventory penalty, applied to q before fills)
    #
    # Probabilities:
    # p_bid = 1 - exp(-A * exp(-k * b) * dt)  [if q < Q, else 0]
    # p_ask = 1 - exp(-A * exp(-k * a) * dt)  [if q > -Q, else 0]
    #
    # Expected value for given (b, a) at state (n, q):
    # E = (gamma/2)*sigma^2*q^2*dt  [cost]
    #   + p_bid*(1-p_ask)*(b + V_{n+1}(q-1))
    #   + p_ask*(1-p_bid)*(a + V_{n+1}(q+1))
    #   + p_bid*p_ask*(b + a + V_{n+1}(q))
    #   + (1-p_bid)*(1-p_ask)*V_{n+1}(q)
    #
    # Simplify:
    # E = cost + p_bid*(1-p_ask)*b + p_ask*(1-p_bid)*a + p_bid*p_ask*(b+a)
    #   + p_bid*(1-p_ask)*V_{n+1}(q-1) + p_ask*(1-p_bid)*V_{n+1}(q+1)
    #   + p_bid*p_ask*V_{n+1}(q) + (1-p_bid)*(1-p_ask)*V_{n+1}(q)
    #
    # The V terms:
    # = p_bid*(1-p_ask)*V(q-1) + p_ask*(1-p_bid)*V(q+1) + [p_bid*p_ask + (1-p_bid)*(1-p_ask)]*V(q)
    # = p_bid*(1-p_ask)*V(q-1) + p_ask*(1-p_bid)*V(q+1) + [1 - p_bid - p_ask + 2*p_bid*p_ask]*V(q)
    #
    # The depth terms:
    # = p_bid*(1-p_ask)*b + p_ask*(1-p_bid)*a + p_bid*p_ask*(b+a)
    # = b*[p_bid*(1-p_ask) + p_bid*p_ask] + a*[p_ask*(1-p_bid) + p_bid*p_ask]
    # = b*p_bid + a*p_ask
    #
    # So: E = cost + b*p_bid + a*p_ask + p_bid*(1-p_ask)*V(q-1) + p_ask*(1-p_bid)*V(q+1) + [1-p_bid-p_ask+2*p_bid*p_ask]*V(q)

    # For the optimal policy, we need to choose b and a to maximize E.
    # Note that b and a appear linearly in the objective with coefficients p_bid and p_ask.
    # But p_bid and p_ask depend on b and a nonlinearly.
    #
    # For a fixed p_bid (i.e., fixed b), the optimal a maximizes:
    #   a * p_ask + p_ask*(1-p_bid)*V(q+1) + [1-p_bid-p_ask+2*p_bid*p_ask]*V(q)
    #   = a * p_ask + p_ask * [(1-p_bid)*V(q+1) + (1-p_bid)*V(q) - p_ask*V(q) + 2*p_bid*p_ask*V(q)/p_ask ... ]
    #
    # This is getting complex. Let me think differently.
    #
    # Actually, since b and a are continuous, and the objective is:
    # f(b, a) = b*p_bid(b) + a*p_ask(a) + g(p_bid(b), p_ask(a))
    # where g depends on V values.
    #
    # The key insight: for a given regime, the optimal depths might be found by
    # considering that the marginal benefit of increasing depth equals the marginal cost
    # in terms of inventory risk.
    #
    # Let's use a simpler approach: since the problem is 1D in each depth, and the
    # coupling is only through the probability terms, we can try a grid search over
    # reasonable depth values.
    #
    # But first, let's note that the seed uses 1/k as depth. The optimal depth should
    # depend on inventory: when q is large positive, we want to widen the bid (or not
    # quote bid) to avoid more inventory, and tighten the ask.
    #
    # Given time constraints, let's use an approximate approach:
    # The optimal bid depth b* satisfies roughly:
    #   A * exp(-k*b) * dt * [V(q) - V(q-1) + b] ≈ 0  (marginal condition)
    # This is approximate. A better approach for the discrete model:
    #
    # Let's just do the DP with a finite grid of depth values.

    # Depth grid: from 0 to some max_depth
    # max_depth: at depth d, p = 1-exp(-A*exp(-k*d)*dt)
    # For very large d, exp(-k*d) -> 0, so p -> 0.
    # Let's use max_depth = 10/k (where exp(-k*10/k) = exp(-10) ≈ 4.5e-5)
    max_depth = 10.0 / k
    n_depth = 200
    depth_grid = np.linspace(0, max_depth, n_depth)

    # Precompute p for each depth
    # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depth_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)

    # Terminal value (step = steps): V_steps(q) = 0 for all q
    V_next = np.zeros(n_inv)

    # Policy table
    policy = np.zeros((steps, n_inv, 2))

    # Solve backwards
    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > _DEADLINE - 10:
            return None

        V_curr = np.zeros(n_inv)
        cost = 0.5 * gamma * sigma * sigma * dt

        for i_q, q in enumerate(inv_range):
            # Cost term
            c = cost * q * q

            best_val = -np.inf
            best_b_idx = 0
            best_a_idx = 0

            # For boundary conditions
            can_bid = (q < Q)
            can_ask = (q > -Q)

            # V(q-1), V(q), V(q+1)
            V_q = V_next[i_q]
            V_qm1 = V_next[i_q - 1] if i_q > 0 else 0.0
            V_qp1 = V_next[i_q + 1] if i_q < n_inv - 1 else 0.0

            # Iterate over depth grid for b and a
            # To save time, we can note that the objective separates somewhat.
            # For each b_idx, find best a_idx, then find best b_idx.
            # But the coupling makes it 2D. Let's do a 2D grid but with early termination.

            # Actually, let's try a smarter approach:
            # For fixed b, the optimal a can be found analytically or by 1D search.
            # f(a) = a*p_ask(a) + p_ask(a)*(1-p_bid)*V_qp1 + [1-p_bid-p_ask(a)+2*p_bid*p_ask(a)]*V_q
            #      = a*p_ask(a) + p_ask(a)*[(1-p_bid)*V_qp1 + (1-p_bid)*V_q] - p_ask(a)*V_q + 2*p_bid*p_ask(a)*V_q
            #      = a*p_ask(a) + p_ask(a)*[(1-p_bid)*(V_qp1+V_q) - V_q + 2*p_bid*V_q]
            #      = a*p_ask(a) + p_ask(a)*[(1-p_bid)*V_qp1 + (1-p_bid)*V_q - V_q + 2*p_bid*V_q]
            #      = a*p_ask(a) + p_ask(a)*[(1-p_bid)*V_qp1 + V_q*(1-p_bid-1+2*p_bid)]
            #      = a*p_ask(a) + p_ask(a)*[(1-p_bid)*V_qp1 + V_q*(p_bid-1)]
            #      = a*p_ask(a) + p_ask(a)*[(1-p_bid)*V_qp1 - (1-p_bid)*V_q]
            #      = a*p_ask(a) + p_ask(a)*(1-p_bid)*(V_qp1 - V_q)
            
            # So for fixed b (fixed p_bid), maximize over a:
            #   g(a) = a*p_ask(a) + p_ask(a)*(1-p_bid)*(V_qp1 - V_q)
            #        = p_ask(a) * [a + (1-p_bid)*(V_qp1 - V_q)]
            
            # Similarly, the full expression for fixed b:
            # E = c + b*p_bid + p_bid*(1-p_ask)*V_qm1 + p_ask*(1-p_bid)*V_qp1 + [1-p_bid-p_ask+2*p_bid*p_ask]*V_q
            #   + a*p_ask  [from the depth term]
            
            # Let me re-derive more carefully.
            # E = c + b*p_bid + a*p_ask 
            #   + p_bid*(1-p_ask)*V_qm1 + p_ask*(1-p_bid)*V_qp1 
            #   + (1 - p_bid - p_ask + 2*p_bid*p_ask)*V_q
            
            # Group terms by p_ask:
            # = c + b*p_bid 
            #   + p_ask*[a + (1-p_bid)*V_qp1 - p_bid*V_qm1 + 2*p_bid*V_q]
            #   + (1-p_bid)*V_qm1 + (1-p_bid)*V_qp1 + (1-p_bid)*V_q - p_ask*V_q
            # 
            # Hmm, this is getting messy. Let me just do the 2D grid but optimize.

            # For efficiency, precompute for each b_idx the terms that don't depend on a
            # and for each a_idx the terms that depend on a.
            
            # Actually, given n_depth=200, a 200x200 grid per state per step is 40000 ops.
            # With 61 states and 1000 steps, that's 2.44 billion ops. Too slow.
            
            # Let's use a smaller grid or a smarter method.
            # 
            # Key insight: the optimal depth is likely near 1/k (the seed). 
            # Let's use a coarser grid: n_depth = 50
            # And we can vectorize over the depth grid.
            
            # For each b_idx (0 to n_depth-1):
            #   p_b = p_grid[b_idx] if can_bid else 0
            #   For each a_idx:
            #     p_a = p_grid[a_idx] if can_ask else 0
            #     E = c + b*p_b + a*p_a + p_b*(1-p_a)*V_qm1 + p_a*(1-p_b)*V_qp1 + (1-p_b-p_a+2*p_b*p_a)*V_q
            
            # Vectorize over a_idx for each b_idx:
            if can_bid and can_ask:
                p_b = p_grid  # shape (n_depth,)
                b_vals = depth_grid  # shape (n_depth,)
                # For each b_idx, compute over all a_idx
                # E[b_idx, a_idx] = c + b_vals[b_idx]*p_b[b_idx] + depth_grid[a_idx]*p_grid[a_idx]
                #   + p_b[b_idx]*(1-p_grid[a_idx])*V_qm1 + p_grid[a_idx]*(1-p_b[b_idx])*V_qp1
                #   + (1-p_b[b_idx]-p_grid[a_idx]+2*p_b[b_idx]*p_grid[a_idx])*V_q
                
                # This is still O(n_depth^2) per state. With n_depth=50, that's 2500 per state.
                # 61 states * 1000 steps * 2500 = 152.5 million. Might be OK in numpy.
                
                # Let's vectorize:
                # For a given b_idx:
                # term1 = depth_grid * p_grid  # (n_depth,)
                # term2 = p_b[b_idx] * (1 - p_grid) * V_qm1  # scalar
                # term3 = p_grid * (1 - p_b[b_idx]) * V_qp1  # (n_depth,)
                # term4 = (1 - p_b[b_idx] - p_grid + 2 * p_b[b_idx] * p_grid) * V_q  # (n_depth,)
                # E_a = term1 + term2 + term3 + term4 + c + b_vals[b_idx]*p_b[b_idx]
                # best_a = argmax(E_a)
                
                # To fully vectorize over b_idx too:
                # p_b[:, None] * (1 - p_grid[None, :]) * V_qm1  -> (n_depth, n_depth)
                # This creates a 50x50 = 2500 array per state. That's fine.
                
                # But doing this for each of 61 states * 1000 steps = 61000 iterations of creating 2500 arrays
                # might be slow. Let's try n_depth = 30 for speed.
                
                pass  # We'll handle below with a different approach

            # Given time constraints, let's use a simpler heuristic-based policy
            # that captures the main effect: adjust depth based on inventory.
            #
            # The main economic insight: 
            # - When q > 0 (long inventory), we want to:
            #   * Widen bid or not quote bid (to avoid buying more)
            #   * Tighten ask (to sell faster)
            # - When q < 0 (short inventory), we want to:
            #   * Tighten bid (to buy faster)
            #   * Widen ask or not quote ask (to avoid selling more)
            #
            # A reasonable approximation:
            #   bid_depth = base_depth * (1 + alpha * q)  [clip to >= 0]
            #   ask_depth = base_depth * (1 - alpha * q)  [clip to >= 0]
            # where base_depth ≈ 1/k and alpha controls the sensitivity.
            #
            # But this is just a heuristic. Let's try a proper but faster DP.
            
            # FAST DP APPROACH:
            # Use the fact that the optimal depth can be found by 1D optimization.
            # For a given b, the optimal a maximizes:
            #   h(a) = p_ask(a) * [a + (1-p_bid)*(V_qp1 - V_q)] + [terms not involving a]
            # Wait, let me re-derive.
            
            # E = c + b*p_b + a*p_a + p_b*(1-p_a)*V_qm1 + p_a*(1-p_b)*V_qp1 + (1-p_b-p_a+2*p_b*p_a)*V_q
            # 
            # Collect terms with p_a:
            # = c + b*p_b + p_a*a + p_a*(1-p_b)*V_qp1 - p_b*p_a*V_qm1 + (1-p_b)*V_qm1 + (1-p_b)*V_qp1 + (1-p_b)*V_q - p_a*V_q + 2*p_b*p_a*V_q
            # 
            # Hmm, let me just expand (1-p_b-p_a+2*p_b*p_a)*V_q:
            # = V_q - p_b*V_q - p_a*V_q + 2*p_b*p_a*V_q
            # 
            # So E = c + b*p_b + a*p_a + p_b*V_qm1 - p_b*p_a*V_qm1 + p_a*V_qp1 - p_b*p_a*V_qp1 + V_q - p_b*V_q - p_a*V_q + 2*p_b*p_a*V_q
            # 
            # Group by p_a:
            # = c + b*p_b + p_b*V_qm1 + (1-p_b)*V_qp1 + (1-p_b)*V_q - p_b*V_q  [wait, (1-p_b)*V_qp1 is not right]
            
            # I'm making errors. Let me be very careful.
            # 
            # E = c + b*p_b + a*p_a 
            #   + p_b*(1-p_a)*V_qm1 
            #   + p_a*(1-p_b)*V_qp1 
            #   + (1 - p_b - p_a + 2*p_b*p_a)*V_q
            # 
            # Expand the last term:
            # = V_q - p_b*V_q - p_a*V_q + 2*p_b*p_a*V_q
            # 
            # So:
            # E = c + b*p_b + a*p_a 
            #   + p_b*V_qm1 - p_b*p_a*V_qm1 
            #   + p_a*V_qp1 - p_b*p_a*V_qp1 
            #   + V_q - p_b*V_q - p_a*V_q + 2*p_b*p_a*V_q
            # 
            # Now collect constant terms (no p_a):
            # const = c + b*p_b + p_b*V_qm1 + V_q - p_b*V_q
            # 
            # Collect terms linear in p_a:
            # lin_p_a = a*p_a + p_a*V_qp1 - p_a*V_q
            #         = p_a * (a + V_qp1 - V_q)
            # 
            # Collect terms quadratic in p_a (i.e., p_b*p_a):
            # quad = -p_b*p_a*V_qm1 - p_b*p_a*V_qp1 + 2*p_b*p_a*V_q
            #      = p_b*p_a * (-V_qm1 - V_qp1 + 2*V_q)
            # 
            # So: E = const + p_a*(a + V_qp1 - V_q) + p_b*p_a*(-V_qm1 - V_qp1 + 2*V_q)
            #      = const + p_a * [a + V_qp1 - V_q + p_b*(-V_qm1 - V_qp1 + 2*V_q)]
            # 
            # For fixed b (fixed p_b), maximize over a:
            #   maximize p_a(a) * [a + C]  where C = V_qp1 - V_q + p_b*(-V_qm1 - V_qp1 + 2*V_q)
            # 
            # This is a 1D optimization! For each b, find a that maximizes p_a(a)*(a+C).
            # 
            # Note: if C < 0, the optimal a might be 0 (don't quote ask) since p_a(0)*C < 0.
            # If C > 0, there's an interior optimum.
            # 
            # Similarly, for fixed a, the optimal b maximizes p_b(b)*(b + C') where C' depends on a.
            # 
            # This suggests an alternating optimization or we can just do the 2D grid
            # but with the insight that we only need to evaluate p_a*(a+C) for the grid.

            # Given the complexity and time constraints, let's use a practical approach:
            # 1. Use a coarse depth grid (n_depth=40)
            # 2. For each state, do the 2D grid evaluation vectorized in numpy
            # 3. If too slow, fall back to a heuristic

            # Let's try n_depth=40
            n_depth_local = 40
            depth_grid_local = np.linspace(0, max_depth, n_depth_local)
            exp_term_local = np.exp(-k * depth_grid_local)
            p_grid_local = 1.0 - np.exp(-A * exp_term_local * dt)

            if can_bid and can_ask:
                # 2D grid: for each b_idx, a_idx
                # E[i,j] = c + depth_grid_local[i]*p_grid_local[i] + depth_grid_local[j]*p_grid_local[j]
                #   + p_grid_local[i]*(1-p_grid_local[j])*V_qm1 
                #   + p_grid_local[j]*(1-p_grid_local[i])*V_qp1 
                #   + (1-p_grid_local[i]-p_grid_local[j]+2*p_grid_local[i]*p_grid_local[j])*V_q
                
                b_col = depth_grid_local[:, None]  # (n_depth, 1)
                p_b_col = p_grid_local[:, None]
                a_row = depth_grid_local[None, :]  # (1, n_depth)
                p_a_row = p_grid_local[None, :]
                
                E = (c 
                     + b_col * p_b_col + a_row * p_a_row
                     + p_b_col * (1 - p_a_row) * V_qm1
                     + p_a_row * (1 - p_b_col) * V_qp1
                     + (1 - p_b_col - p_a_row + 2 * p_b_col * p_a_row) * V_q)
                
                flat_idx = np.argmax(E)
                best_b_idx = flat_idx // n_depth_local
                best_a_idx = flat_idx % n_depth_local
                best_val = E[best_b_idx, best_a_idx]
                
            elif can_bid and not can_ask:
                # Only bid can fill
                # E = c + b*p_b + p_b*V_qm1 + (1-p_b)*V_q
                #   = c + p_b*(b + V_qm1) + (1-p_b)*V_q
                #   = c + V_q + p_b*(b + V_qm1 - V_q)
                # Maximize p_b*(b + V_qm1 - V_q)
                C = V_qm1 - V_q
                vals = p_grid_local * (depth_grid_local + C)
                best_b_idx = np.argmax(vals)
                best_a_idx = 0  # no ask
                best_val = c + V_q + vals[best_b_idx]
                
            elif not can_bid and can_ask:
                # Only ask can fill
                # E = c + a*p_a + p_a*V_qp1 + (1-p_a)*V_q
                #   = c + V_q + p_a*(a + V_qp1 - V_q)
                C = V_qp1 - V_q
                vals = p_grid_local * (depth_grid_local + C)
                best_a_idx = np.argmax(vals)
                best_b_idx = 0  # no bid
                best_val = c + V_q + vals[best_a_idx]
            else:
                # Neither can fill
                best_b_idx = 0
                best_a_idx = 0
                best_val = c + V_q

            V_curr[i_q] = best_val
            policy[n, i_q, 0] = depth_grid_local[best_b_idx]
            policy[n, i_q, 1] = depth_grid_local[best_a_idx]

        V_next = V_curr

    return policy


def quote(t, inventory, mid, params):
    """Quote bid and ask depths based on solved DP policy."""
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps

    # Determine step index
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    # Cache key
    cache_key = (sigma, A, k, gamma, T, steps, Q)

    # Try to get cached policy
    if cache_key not in _policy_cache:
        # Check time
        if time.time() - _start_time > _DEADLINE - 50:
            # Fallback: simple heuristic
            base_depth = 1.0 / k
            # Adjust based on inventory
            alpha = 0.05 * gamma * sigma * sigma / (A * k)
            bid_depths = np.clip(base_depth * (1 + alpha * inventory), 0.0, 10.0 / k)
            ask_depths = np.clip(base_depth * (1 - alpha * inventory), 0.0, 10.0 / k)
            return bid_depths, ask_depths

        # Solve DP
        policy = _solve_dp(params)
        if policy is None:
            # Fallback
            base_depth = 1.0 / k
            alpha = 0.05 * gamma * sigma * sigma / (A * k)
            bid_depths = np.clip(base_depth * (1 + alpha * inventory), 0.0, 10.0 / k)
            ask_depths = np.clip(base_depth * (1 - alpha * inventory), 0.0, 10.0 / k)
            return bid_depths, ask_depths

        _policy_cache[cache_key] = policy

    policy = _policy_cache[cache_key]

    # Look up policy for each path
    inv_range = np.arange(-Q, Q + 1)
    n_inv = len(inv_range)
    
    # Clip inventory to valid range
    inv_clipped = np.clip(inventory, -Q, Q)
    inv_idx = inv_clipped + Q  # 0 to 2*Q

    bid_depths = policy[n, inv_idx, 0]
    ask_depths = policy[n, inv_idx, 1]

    # Ensure valid values
    bid_depths = np.clip(bid_depths, 0.0, np.inf)
    ask_depths = np.clip(ask_depths, 0.0, np.inf)

    # Handle boundary: no bid at q=Q, no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
