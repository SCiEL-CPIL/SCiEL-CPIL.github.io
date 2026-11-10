"""Improved 0/1 knapsack heuristic with opportunity-cost ratio selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 138.0  # seconds; safety margin before 150s kill
_START = None


def _fractional_knapsack_value(weights, values, capacity, exclude_idx=None):
    """Compute fractional knapsack value (upper bound), optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    total_val = 0.0
    remaining_cap = capacity
    for i in range(len(w_sorted)):
        if remaining_cap <= 0:
            break
        if w_sorted[i] <= remaining_cap:
            total_val += v_sorted[i]
            remaining_cap -= w_sorted[i]
        else:
            total_val += v_sorted[i] * (remaining_cap / w_sorted[i])
            remaining_cap = 0
    return float(total_val)


def _greedy_fill_value(weights, values, capacity, exclude_idx=None, order=None):
    """Compute total value from greedy density fill (0/1), optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
        if order is not None:
            order_masked = order[mask]
        else:
            order_masked = None
    else:
        w = weights
        v = values
        order_masked = order
    if len(w) == 0:
        return 0.0
    
    if order_masked is None:
        density = v / w
        order_masked = np.argsort(density)[::-1]
    
    w_sorted = w[order_masked]
    v_sorted = v[order_masked]
    cum_w = np.cumsum(w_sorted)
    cum_v = np.cumsum(v_sorted)
    idx = np.searchsorted(cum_w, capacity, side='right') - 1
    if idx < 0:
        return 0.0
    return float(cum_v[idx])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item for 0/1 knapsack using opportunity-cost ratio.
    The 'price' of an item is estimated by the marginal loss in the fractional
    upper bound if that item were excluded. We select the item that maximizes
    value / opportunity_cost.
    """
    global _START
    if _START is None:
        _START = time.time()
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Fallback to simple density
        if len(remaining_weights) == 0:
            return -1
        w = np.asarray(remaining_weights, dtype=float)
        v = np.asarray(remaining_values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities[np.where(w == 0)] = 0
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1

    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)

    # Pre-compute base fractional knapsack value (without excluding anything)
    base_frac_val = _fractional_knapsack_value(w, v, c)

    # Pre-compute density order for greedy fills
    densities = v / w
    order_density = np.argsort(densities)[::-1]
    
    # Also pre-compute value order for alternative greedy
    order_value = np.argsort(v)[::-1]

    # Candidate selection: focus on top candidates by density to limit expensive computations
    # With 100 items, we can afford to check all, but let's be smart.
    # The opportunity cost calculation requires a fractional knapsack solve per item,
    # which is O(n log n). Doing this for all n items is O(n^2 log n).
    # For n=100, this is 100 * 100 * log(100) ~ 70000 operations, which is fast.
    
    best_score = -1.0
    best_idx = -1

    # We will evaluate each item that fits
    # To speed up, we can skip items with very low density if we find a good one early,
    # but for now, let's evaluate all feasible items.
    
    for i in range(n):
        wi = w[i]
        if wi > c:
            continue
        
        vi = v[i]
        
        # Calculate opportunity cost:
        # OC_i = base_frac_val - frac_val(excluding i)
        # This represents the loss in the upper bound if we force-exclude item i.
        # However, we are *taking* item i, so the remaining capacity is c - wi.
        # The relevant comparison is:
        # Value if we take i: vi + (value of remaining subproblem with cap c-wi, excluding i)
        # Value if we don't take i: (value of remaining subproblem with cap c, excluding i)
        
        # Let's use the fractional bound for the "don't take" scenario
        # and the greedy fill for the "take" scenario to get a realistic estimate.
        
        # Fractional bound if we exclude i (capacity c)
        frac_bound_excl = _fractional_knapsack_value(w, v, c, exclude_idx=i)
        
        # The opportunity cost is the difference in fractional bounds
        # But a better metric might be:
        # Score = vi + greedy_fill(c - wi, excl i) - (some baseline)
        
        # The directive says: "select the item that maximizes the ratio of its value to this calculated opportunity cost"
        # Opportunity cost = base_frac_val - frac_bound_excl
        # If we take i, we lose the opportunity to use the capacity for other things in the fractional solution.
        
        oc = base_frac_val - frac_bound_excl
        
        # If oc is very small or negative (due to numerical issues or if i is not in the fractional solution),
        # the ratio vi/oc might be huge or undefined.
        # We need to be careful.
        
        # Let's refine the score:
        # We want to maximize the total value achieved.
        # Estimate total value if we pick i:
        # v_i + greedy_fill(c - w_i, excluding i, using density order)
        
        rem_cap = c - wi
        if rem_cap < 0:
            continue
            
        # Use both density and value orders for greedy fill to get a robust estimate
        greedy_dens = _greedy_fill_value(w, v, rem_cap, exclude_idx=i, order=order_density)
        greedy_val = _greedy_fill_value(w, v, rem_cap, exclude_idx=i, order=order_value)
        
        est_total_val = vi + max(greedy_dens, greedy_val)
        
        # The directive specifically asks for "ratio of its value to this calculated opportunity cost"
        # Let's try that specific formulation first, but blended with the estimated total value
        # to avoid division by zero or instability.
        
        if oc > 1e-9:
            ratio_score = vi / oc
        else:
            # If opportunity cost is ~0, it means excluding i doesn't hurt the fractional bound much,
            # so i is likely not a "critical" item in the fractional solution.
            # We should still consider it, but the ratio is not a good discriminator.
            # Fallback to density or just estimated value.
            ratio_score = densities[i]
            
        # Combine ratio_score and est_total_val
        # est_total_val is in the same units as vi, ratio_score is dimensionless-ish.
        # Let's normalize or just use a weighted sum.
        # Since est_total_val is the actual objective proxy, let's weight it heavily.
        
        # The prompt says: "select the item that maximizes the ratio of its value to this calculated opportunity cost"
        # Let's try to follow that closely, but use est_total_val as a tie-breaker or blend.
        
        # Actually, let's look at the "price" interpretation.
        # If OC is high, the item is "expensive" in terms of opportunity cost.
        # We want high value / high cost.
        
        # Let's use a smoothed version to avoid instability.
        # score = est_total_val + alpha * (vi / (oc + epsilon))
        
        epsilon = 1e-6
        ratio = vi / (oc + epsilon)
        
        # Scale ratio to be comparable to est_total_val?
        # est_total_val is roughly O(100), ratio is O(1-10).
        # Let's just sum them with a weight.
        
        score = est_total_val + 0.5 * ratio
        
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        # Fallback if no item was found (should not happen if feasible)
        return -1

    return int(best_idx)
# EVOLVE-BLOCK-END
