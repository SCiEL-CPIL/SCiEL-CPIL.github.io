"""Bounded lookahead 0/1 knapsack heuristic with Lagrangian-guided scoring."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _greedy_complete(w, v, cap, policy='density'):
    """Simulate greedy completion with LP-style fractional bonus. Returns total added value."""
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    if policy == 'density':
        order = np.argsort(-v / np.maximum(w, 1e-12))
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / np.maximum(w, 1e-12) + 0.15 * v))
    elif policy == 'lagrangian':
        # Use a heuristic lambda for ordering
        densities = v / np.maximum(w, 1e-12)
        lam_est = float(np.median(densities)) * 0.5
        order = np.argsort(-(v - lam_est * w))
    else:
        order = np.argsort(-v / np.maximum(w, 1e-12))

    rem = float(cap)
    total = 0.0
    for i in order:
        wi = float(w[i])
        if wi <= rem:
            rem -= wi
            total += float(v[i])
            if rem <= 0:
                break
        else:
            # Fractional bonus: add partial value for the first item that doesn't fit
            # This mimics LP relaxation and gives a better estimate
            if rem > 0:
                total += float(v[i]) * (rem / wi)
            break
    return total


def _lagrangian_lambda(w, v, cap):
    """
    Estimate the Lagrangian multiplier lambda such that the continuous
    relaxation (take all items with v_i - lambda*w_i > 0) approximately
    exhausts capacity. We use binary search on lambda with fractional
    item handling for tighter convergence.
    """
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0

    # Bounds for lambda: 0 to max(v/w)
    densities = v / np.maximum(w, 1e-12)
    lam_lo = 0.0
    lam_hi = float(np.max(densities)) + 1e-9

    # If even at lambda=0 the total weight exceeds capacity, we need positive lambda
    total_w = float(np.sum(w))
    if total_w <= cap:
        return 0.0

    # Binary search for lambda where the LP relaxation exactly fills capacity
    # In the LP relaxation, we take all items with v_i - lambda*w_i > 0 fully,
    # and one fractional item to fill the rest.
    for _ in range(60):
        lam_mid = (lam_lo + lam_hi) / 2.0
        # Reduced costs
        reduced_costs = v - lam_mid * w
        # Items with positive reduced cost are taken fully
        taken_mask = reduced_costs > 0
        weight_taken = float(np.sum(w[taken_mask]))

        if weight_taken >= cap:
            # All taken items already exceed capacity; need higher lambda
            lam_lo = lam_mid
        else:
            # Check if we can add a fractional item to fill remaining capacity
            remaining_cap = cap - weight_taken
            if remaining_cap > 1e-12:
                # Find the item with highest reduced cost among not-taken items
                not_taken_mask = ~taken_mask
                if np.any(not_taken_mask):
                    rc_not_taken = reduced_costs[not_taken_mask]
                    w_not_taken = w[not_taken_mask]
                    # The best fractional item is the one with highest reduced cost
                    # that also has weight <= remaining_cap (or we take it fractionally)
                    best_frac_idx = int(np.argmax(rc_not_taken))
                    best_frac_w = float(w_not_taken[best_frac_idx])
                    # In LP, we'd take min(1, remaining_cap/best_frac_w) fraction
                    # For lambda estimation, if this item's weight would fill capacity,
                    # we're at the right lambda; otherwise, we might need lower lambda
                    if best_frac_w <= remaining_cap + 1e-12:
                        # This item fits entirely; but in LP it's fractional
                        # The lambda is correct if the next best item has negative reduced cost
                        # For simplicity, if weight_taken + best_frac_w > cap, lambda is good
                        if weight_taken + best_frac_w > cap:
                            lam_lo = lam_mid
                        else:
                            lam_hi = lam_mid
                    else:
                        # Item is too heavy to fit even fractionally in a meaningful way
                        # Actually in LP we always can take a fraction
                        # If remaining_cap > 0 and there's an item, lambda should be adjusted
                        lam_hi = lam_mid
                else:
                    lam_hi = lam_mid
            else:
                # Exactly filled
                lam_lo = lam_mid
                lam_hi = lam_mid
                break

    return (lam_lo + lam_hi) / 2.0


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add using Lagrangian-guided scoring with lookahead.
    
    Strategy:
    1. Compute lambda from the continuous relaxation of the current state.
    2. Score each feasible item by its Lagrangian objective: v_i - lambda * w_i.
    3. For top candidates, do a full lookahead with greedy completion to verify.
    4. Return the best candidate.
    """
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Only consider feasible items (weight <= capacity)
    feasible = w <= cap
    if not np.any(feasible):
        return -1

    feasible_indices = np.where(feasible)[0]
    fw = w[feasible]
    fv = v[feasible]
    n_feasible = len(fw)

    # Check time budget
    if _remaining_budget() < 0.5:
        # Fallback to simple density
        density = fv / np.maximum(fw, 1e-12)
        best_local = int(np.argmax(density))
        return int(feasible_indices[best_local])

    # Compute Lagrangian lambda for the current state
    lam = _lagrangian_lambda(w, v, cap)

    # Score all feasible items by Lagrangian objective
    lag_scores = fv - lam * fw

    # Select top candidates for lookahead verification
    # Blend Lagrangian score and density to get diverse candidates
    if n_feasible <= 40:
        candidates = feasible_indices
    else:
        # Take top 40: mix of top Lagrangian and top density
        top_k = min(40, n_feasible)
        lag_rank = np.argsort(-lag_scores)
        density = fv / np.maximum(fw, 1e-12)
        dens_rank = np.argsort(-density)
        
        # Combine: first 25 from Lagrangian, next 15 from density (avoiding duplicates)
        selected = set()
        candidates_list = []
        for idx in lag_rank[:25]:
            candidates_list.append(idx)
            selected.add(idx)
        for idx in dens_rank:
            if len(candidates_list) >= top_k:
                break
            if idx not in selected:
                candidates_list.append(idx)
                selected.add(idx)
        candidates = feasible_indices[np.array(candidates_list)]

    best_score = -1.0
    best_candidate = None

    for ci in candidates:
        wi = float(w[ci])
        vi = float(v[ci])
        new_cap = cap - wi
        if new_cap < 0:
            continue

        # Create remaining arrays after taking this item
        mask = np.ones(n, dtype=bool)
        mask[ci] = False
        rem_w = w[mask]
        rem_v = v[mask]

        # Try multiple policies for completion, including Lagrangian-guided
        for policy in ('density', 'value', 'blend', 'lagrangian'):
            completion_value = _greedy_complete(rem_w, rem_v, new_cap, policy)
            total_value = vi + completion_value
            if total_value > best_score:
                best_score = total_value
                best_candidate = ci

    if best_candidate is None:
        # Fallback to highest Lagrangian score feasible item
        best_local = int(np.argmax(lag_scores))
        return int(feasible_indices[best_local])

    return int(best_candidate)
# EVOLVE-BLOCK-END
