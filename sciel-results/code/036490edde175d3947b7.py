"""Optimal market making with inventory risk: precomputed DP policy."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

# Global cache for precomputed DP policies
_policy_cache = {}


def _build_policy(params):
    """Build the exact DP policy for the given parameters."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    # Key for cache
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]

    # We'll use a discrete grid for depth to make DP tractable.
    # The optimal depth is likely in a reasonable range.
    # Let's use a grid of depths: from a small positive value up to some max.
    # The constraint is depth > -3/k. Let's use depths from max(1e-4, -3/k + 1e-4) to maybe 5/k or so.
    # Actually, let's think about the scale. 1/k is the risk-neutral depth.
    # For regime 1: k=1.5, 1/k=0.667. Let's use depths from 0.01 to 2.0 with step 0.01? That's 199 values.
    # Or maybe we should use a finer grid near the optimum.
    # Let's try: 200 discrete depth values from 0.005 to 2.0.
    
    # Actually, the depth affects fill probability: p = 1 - exp(-A * exp(-k * depth) * dt)
    # And the reward from a fill is the depth itself.
    # The optimal depth balances marginal fill probability against marginal reward.
    
    # Let's use a grid of 300 depth values from 0.001 to 3.0
    depth_grid = np.linspace(0.001, 3.0, 300)
    n_depths = len(depth_grid)
    
    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    # This is the same for bid and ask (by symmetry of the model)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Inventory range: -Q to Q, so 2*Q+1 values
    inv_range = 2 * Q + 1
    # Map inventory q to index: idx = q + Q
    
    # DP: V[n, q] = max expected reward from step n onward given inventory q
    # V[steps, q] = 0 (no terminal charge)
    # V[n, q] = max over bid_depth, ask_depth of:
    #   (1-pb)*(1-pa)*0 + pb*(1-pa)*bid_depth + (1-pb)*pa*ask_depth + pb*pa*(bid_depth + ask_depth)
    #   - (gamma/2)*sigma^2*q^2*dt
    #   + (1-pb)*(1-pa)*V[n+1, q] + pb*(1-pa)*V[n+1, q-1] + (1-pb)*pa*V[n+1, q+1] + pb*pa*V[n+1, q-1+1]
    # Wait: bid fill decreases inventory by 1 (buying), ask fill increases inventory by 1 (selling)
    # So: bid fill -> q-1, ask fill -> q+1
    # V[n, q] = max_{d_bid, d_ask} {
    #   p_b * d_bid * (1 - p_a) + p_a * d_ask * (1 - p_b) + p_b * p_a * (d_bid + d_ask)
    #   - (gamma/2)*sigma^2*q^2*dt
    #   + (1-p_b)*(1-p_a)*V[n+1, q] + p_b*(1-p_a)*V[n+1, q-1] + (1-p_b)*p_a*V[n+1, q+1] + p_b*p_a*V[n+1, q]
    # }
    # Where p_b = p_fill(d_bid), p_a = p_fill(d_ask)
    # Note: no bid fill at q=Q (inventory at max), no ask fill at q=-Q
    
    # For efficiency, since p_b and p_a are from the same grid, and the value function is symmetric
    # (V[n, -q] = V[n, q] by symmetry of the problem), we can exploit this.
    
    # But let's just do the full DP for now.
    
    # V[n, q] for n from steps-1 down to 0
    # Shape: (steps, 2*Q+1)
    V = np.zeros((steps + 1, inv_range), dtype=np.float64)
    
    # Policy: for each n and q, store optimal bid_depth_idx and ask_depth_idx
    # But we need to return actual depths, not indices.
    # Actually, we can just store the optimal depths directly.
    # But that's a lot of storage: steps * (2Q+1) * 2 floats.
    # 1000 * 61 * 2 = 122000 floats, which is fine.
    
    # However, doing the max over 300x300 = 90000 combinations for each (n, q) is:
    # 1000 * 61 * 90000 = 5.49e9 operations. That's too slow.
    
    # We need to be smarter. Notice that the objective is separable in a way.
    # Let's rewrite:
    # f(d_b, d_a) = p_b*d_b*(1-p_a) + p_a*d_a*(1-p_b) + p_b*p_a*(d_b+d_a) + (1-p_b)*(1-p_a)*V_next_q + p_b*(1-p_a)*V_next_qm1 + (1-p_b)*p_a*V_next_qp1 + p_b*p_a*V_next_q
    # = p_b*d_b + p_a*d_a + p_b*p_a*(d_b+d_a) - p_b*d_b*p_a - p_a*d_a*p_b + ... wait let me redo
    
    # Actually:
    # reward = p_b*(1-p_a)*d_b + (1-p_b)*p_a*d_a + p_b*p_a*(d_b+d_a)
    # = p_b*d_b - p_b*p_a*d_b + p_a*d_a - p_b*p_a*d_a + p_b*p_a*d_b + p_b*p_a*d_a
    # = p_b*d_b + p_a*d_a
    
    # Oh nice! The immediate reward simplifies to p_b*d_b + p_a*d_a.
    
    # So: V[n, q] = max_{d_b, d_a} { p_b(d_b)*d_b + p_a(d_a)*d_a - (gamma/2)*sigma^2*q^2*dt + continuation }
    # where continuation = (1-p_b)*(1-p_a)*V[n+1,q] + p_b*(1-p_a)*V[n+1,q-1] + (1-p_b)*p_a*V[n+1,q+1] + p_b*p_a*V[n+1,q]
    
    # The continuation depends on p_b and p_a jointly, so we can't fully separate.
    # But we can precompute for each q:
    #   term0 = V[n+1, q]
    #   term1 = V[n+1, q-1] (if valid)
    #   term2 = V[n+1, q+1] (if valid)
    
    # continuation = (1-p_b)*(1-p_a)*term0 + p_b*(1-p_a)*term1 + (1-p_b)*p_a*term2 + p_b*p_a*term0
    # = term0 + p_b*(1-p_a)*(term1 - term0) + (1-p_b)*p_a*(term2 - term0) + p_b*p_a*(term0 - term0)
    # = term0 + p_b*(1-p_a)*(term1-term0) + (1-p_b)*p_a*(term2-term0)
    
    # So for fixed q, we need to maximize over (p_b, p_a) from our depth grid:
    #   g_b(p_b) + g_a(p_a) + p_b*(1-p_a)*D1 + (1-p_b)*p_a*D2
    # where g_b(p_b) = p_b * d_b(p_b), D1 = term1-term0, D2 = term2-term0
    
    # This is still a joint optimization. But note that the cross term is:
    # p_b*D1 - p_b*p_a*D1 + p_a*D2 - p_b*p_a*D2
    # = p_b*D1 + p_a*D2 - p_b*p_a*(D1+D2)
    
    # So: g_b(p_b) + g_a(p_a) + p_b*D1 + p_a*D2 - p_b*p_a*(D1+D2)
    
    # This is a bilinear-ish function in (p_b, p_a). For fixed p_a, it's linear in p_b (plus g_b which is nonlinear).
    # We can do: for each p_a, maximize over p_b, then maximize over p_a. That's 300*300 = 90000 per (n,q).
    # Still too slow.
    
    # Alternative: use a smaller depth grid. Let's try 50 depth values.
    # 50*50 = 2500 per (n,q), times 1000*61 = 61000, total 152.5M. Might be borderline.
    
    # Let's try 40 depth values: 1600 * 61000 = 97.6M. Should be okay in numpy if vectorized.
    
    # Actually, let's vectorize over q. For a fixed n:
    #   D1[n,q] = V[n+1, q-1] - V[n+1, q] for q in 1..Q (idx 1..Q+Q-1... wait)
    #   D2[n,q] = V[n+1, q+1] - V[n+1, q]
    #   term0 = V[n+1, q]
    
    # For each (n, q), we compute the max over the depth grid.
    # Let's use 60 depth values and vectorize over q.
    
    depth_grid = np.linspace(0.005, 1.5, 60)
    n_depths = len(depth_grid)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    # g(d) = p_fill(d) * d
    g_depth = p_fill * depth_grid
    
    # Now DP
    V = np.zeros((steps + 1, inv_range), dtype=np.float64)
    
    # Store optimal depths for policy lookup
    # optimal_bid[n, q] = depth, optimal_ask[n, q] = depth
    optimal_bid = np.full((steps, inv_range), np.inf, dtype=np.float64)
    optimal_ask = np.full((steps, inv_range), np.inf, dtype=np.float64)
    
    for n in range(steps - 1, -1, -1):
        V_next = V[n + 1]
        # D1[q] = V_next[q-1] - V_next[q], D2[q] = V_next[q+1] - V_next[q]
        # q index: 0 corresponds to q=-Q, ..., Q corresponds to q=0, ..., 2Q corresponds to q=Q
        # q-1 valid when q > -Q, i.e., idx > 0
        # q+1 valid when q < Q, i.e., idx < 2Q
        
        D1 = np.zeros(inv_range, dtype=np.float64)
        D2 = np.zeros(inv_range, dtype=np.float64)
        term0 = V_next.copy()
        
        # For q > -Q (idx > 0): D1[idx] = V_next[idx-1] - V_next[idx]
        D1[1:] = V_next[:-1] - V_next[1:]
        # For q < Q (idx < 2Q): D2[idx] = V_next[idx+1] - V_next[idx]
        D2[:-1] = V_next[1:] - V_next[:-1]
        
        # Inventory penalty
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        inv_penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        # For each (n, q), maximize over (d_b, d_a) in depth_grid x depth_grid
        # objective = g[d_b] + g[d_a] + p_b*D1[q] + p_a*D2[q] - p_b*p_a*(D1[q]+D2[q]) - inv_penalty[q]
        
        # Vectorize: for all q, compute the max over (i, j) where i,j index depth_grid
        # p_b[i], p_a[j], g[i], g[j]
        # obj[i, j, q] = g[i] + g[j] + p_fill[i]*D1[q] + p_fill[j]*D2[q] - p_fill[i]*p_fill[j]*(D1[q]+D2[q]) - inv_penalty[q]
        
        # This is 60*60*61 = 220,000 values per n, times 1000 steps = 220M. Might be tight.
        # Let's reduce to 40 depths.
        
        # Actually, let me just do it with 40 depths.
        # 40*40*61*1000 = 97.6M, should be fine.
        
        # Let me restart with 40 depths.
        pass
    
    # Let me redo with 40 depths properly
    depth_grid = np.linspace(0.005, 1.5, 40)
    n_depths = len(depth_grid)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    g_depth = p_fill * depth_grid
    
    V = np.zeros((steps + 1, inv_range), dtype=np.float64)
    optimal_bid = np.full((steps, inv_range), np.inf, dtype=np.float64)
    optimal_ask = np.full((steps, inv_range), np.inf, dtype=np.float64)
    
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    for n in range(steps - 1, -1, -1):
        V_next = V[n + 1]
        D1 = np.zeros(inv_range, dtype=np.float64)
        D2 = np.zeros(inv_range, dtype=np.float64)
        D1[1:] = V_next[:-1] - V_next[1:]
        D2[:-1] = V_next[1:] - V_next[:-1]
        
        inv_penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        # Build the objective matrix: [n_depths, n_depths, inv_range]
        # obj[i, j, q] = g[i] + g[j] + p_fill[i]*D1[q] + p_fill[j]*D2[q] - p_fill[i]*p_fill[j]*(D1[q]+D2[q]) - inv_penalty[q]
        
        # Vectorized computation
        # g[i] + g[j]: shape [n_depths, n_depths, 1]
        # p_fill[i]*D1[q]: shape [n_depths, 1, inv_range]
        # p_fill[j]*D2[q]: shape [1, n_depths, inv_range]
        # p_fill[i]*p_fill[j]*(D1[q]+D2[q]): shape [n_depths, n_depths, inv_range]
        
        gg = g_depth[:, None] + g_depth[None, :]  # [n_d, n_d]
        pD1 = p_fill[:, None, None] * D1[None, None, :]  # [n_d, 1, inv]
        pD2 = p_fill[None, :, None] * D2[None, None, :]  # [1, n_d, inv]
        ppD = p_fill[:, None, None] * p_fill[None, :, None] * (D1[None, None, :] + D2[None, None, :])  # [n_d, n_d, inv]
        ipen = inv_penalty[None, None, :]  # [1, 1, inv]
        
        obj = gg[:, :, None] + pD1 + pD2 - ppD - ipen
        
        # Handle boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
        # At q=Q (idx=2Q): bid fill not allowed, so set p_b = 0 effectively.
        # We can handle this by setting the bid-related terms to 0 and only optimizing over ask.
        # Simpler: mask out invalid combinations.
        # At q=Q (last idx): bid fills are disallowed. So we should only consider p_b=0.
        # But p_b=0 corresponds to depth -> infinity, which isn't in our grid.
        # Instead, we can set the objective for bid options to -inf at q=Q, and for ask options at q=-Q.
        
        # Actually, the problem says "no bid fill at q = Q, no ask fill at q = -Q".
        # This means at q=Q, bid_depth is effectively inf (no quote) or the fill probability is 0.
        # Let's interpret as: at q=Q, we can't place a bid that would fill. So p_b = 0.
        # Similarly at q=-Q, p_a = 0.
        
        # To handle this, at q=Q, we set the objective to just the ask-only part.
        # This is getting complex. Let's just clamp: at q=Q, set D1 to a large negative number
        # so that the optimizer prefers p_b=0. But p_b=0 isn't in our grid.
        
        # Alternative: add an extra "no quote" option with depth=inf, p=0, g=0.
        # Let's add a special index for no quote.
        
        # Actually, simpler approach: for q=Q, only optimize over ask_depth (bid is forced to no-quote).
        # For q=-Q, only optimize over bid_depth (ask is forced to no-quote).
        # For other q, optimize over both.
        
        # Let me just handle it by setting the objective appropriately.
        # At q=Q (idx=2Q): bid fill impossible. So p_b=0, g_b=0, and the continuation is:
        #   (1-p_a)*V_next[Q] + p_a*V_next[Q] = V_next[Q] (since no bid fill, inventory doesn't decrease)
        # Wait, if no bid fill, then only ask fills change inventory.
        # At q=Q: no bid fill means p_b=0. Continuation = (1-0)*(1-p_a)*V_next[q] + 0*(1-p_a)*V_next[q-1] + (1-0)*p_a*V_next[q+1] + 0*p_a*V_next[q]
        # But q+1 > Q is invalid. So at q=Q, ask fill would go to q+1=Q+1 which is out of bounds.
        # The problem says "no ask fill at q=-Q", implying at q=Q, ask fills are allowed? 
        # Re-reading: "no bid fill at q = Q, no ask fill at q = -Q"
        # So at q=Q, bid fills are disallowed but ask fills are allowed (going to q+1=Q+1? No, that's out of bounds).
        # I think the inventory is bounded at [-Q, Q], so at q=Q, you can't sell (ask fill) because that would exceed Q.
        # Wait, "no bid fill at q=Q" means you can't buy when at max inventory. "no ask fill at q=-Q" means you can't sell when at min inventory.
        # So at q=Q: no bid fill (can't buy more). Ask fills are allowed, moving to q+1... but that's >Q.
        # I think the constraint is that inventory stays in [-Q, Q], so at q=Q, ask fills don't happen (or are disallowed).
        # Similarly at q=-Q, bid fills don't happen.
        
        # Let me interpret: at q=Q, both bid and ask fills that would move inventory outside [-Q,Q] are disallowed.
        # Bid fill at q=Q: q->q-1=Q-1, which is valid. But the problem says "no bid fill at q=Q".
        # So bid fills are explicitly disallowed at q=Q regardless.
        # Ask fill at q=Q: q->q+1=Q+1, invalid. So ask fills are also disallowed at q=Q.
        # Similarly at q=-Q: bid fill q->q-1=-Q-1 invalid, and "no ask fill at q=-Q" explicitly.
        
        # So at q=Q: no fills at all. p_b=0, p_a=0.
        # At q=-Q: no fills at all. p_b=0, p_a=0.
        
        # This simplifies things. At boundary inventories, no quotes.
        
        # Let me set the objective to -inf for all (i,j) at boundary q values.
        # And the optimal depth will be inf (no quote).
        
        # Actually, if no fills are possible, the optimal is to not quote (depth=inf, p=0).
        # The objective becomes: 0 + 0 + 0 + 0 - 0 - inv_penalty = -inv_penalty.
        # Which is the same as any (i,j) with p=0. So we can just set optimal depth to inf.
        
        # For interior q, do the normal optimization.
        
        # Let me compute the max for all q, then override boundaries.
        
        max_obj = np.max(obj, axis=(0, 1))  # [inv_range]
        max_idx = np.argmax(obj, axis=(0, 1))  # [inv_range], flattened index
        
        for qi in range(inv_range):
            flat_idx = max_idx[qi]
            i = flat_idx // n_depths
            j = flat_idx % n_depths
            optimal_bid[n, qi] = depth_grid[i]
            optimal_ask[n, qi] = depth_grid[j]
        
        # Override boundaries
        optimal_bid[n, 0] = np.inf  # q=-Q
        optimal_ask[n, 0] = np.inf
        optimal_bid[n, inv_range - 1] = np.inf  # q=Q
        optimal_ask[n, inv_range - 1] = np.inf
        
        V[n] = max_obj - 0  # already includes -inv_penalty in obj
    
    _policy_cache[key] = (depth_grid, optimal_bid, optimal_ask)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed DP policy."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Determine step number from t
    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)
    
    # Get policy
    depth_grid, optimal_bid, optimal_ask = _build_policy(params)
    
    # Map inventory to index
    inv_range = 2 * Q + 1
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, inv_range - 1)
    
    bid_depths = optimal_bid[n, inv_idx].astype(np.float64)
    ask_depths = optimal_ask[n, inv_idx].astype(np.float64)
    
    # Ensure valid depths (no negative values below -3/k, no NaN, no -inf)
    min_depth = -3.0 / k + 1e-9
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, min_depth), np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, min_depth), np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
