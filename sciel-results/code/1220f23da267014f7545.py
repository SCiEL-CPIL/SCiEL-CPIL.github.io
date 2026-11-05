import numpy as np
import time
from scipy.optimize import linprog

_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill

def _solve_lp(remaining_capacity, w_feas, v_feas):
    """
    Solve the fractional knapsack LP for the feasible items.
    Returns a boolean mask of items that have positive weight in the optimal solution,
    or None if LP fails.
    """
    n = len(w_feas)
    if n == 0:
        return None
    
    # Maximize sum(v_i * x_i)
    # Minimize -sum(v_i * x_i)
    c = -v_feas
    
    # Constraints:
    # 1. sum(w_i * x_i) <= C
    # 2. x_i <= 1 for all i
    # 3. x_i >= 0 (handled by bounds)
    
    # A_ub * x <= b_ub
    # Row 1: [w_0, w_1, ..., w_{n-1}] <= C
    # Rows 2..n+1: e_i <= 1
    
    A_ub = np.zeros((n + 1, n))
    A_ub[0, :] = w_feas
    b_ub = np.zeros(n + 1)
    b_ub[0] = remaining_capacity
    
    for i in range(n):
        A_ub[i + 1, i] = 1.0
        b_ub[i + 1] = 1.0
    
    bounds = [(0, 1) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Items with x_i > epsilon are "active" in the fractional solution
            # In 0/1 knapsack, the optimal solution is often close to the fractional one.
            # Specifically, there is at most one fractional item in the optimal fractional solution.
            # The items with x_i = 1 are definitely in the optimal 0/1 solution? Not necessarily,
            # but they are good candidates. The item with 0 < x_i < 1 is the critical one.
            # A common heuristic is to pick the item that is fractional, or one of the items with x=1
            # that has the highest "regret" or something similar.
            # Actually, for 0/1, we can just pick the item with the largest x_i.
            # If multiple have x_i=1, pick the one with highest value/weight?
            # Let's look at the solution.
            x = res.x
            # Find the item with the largest x_i.
            # If x_i is close to 1, it's a strong candidate.
            # If x_i is fractional, it's the "pivot" item.
            
            # Strategy: Pick the item with the largest x_i.
            # If there's a tie (e.g., multiple 1s), break tie by density (v/w).
            # This effectively selects the "core" of the solution.
            
            # However, a better heuristic from the literature for 0/1 knapsack
            # using LP relaxation:
            # The optimal 0/1 solution is either the floor of the LP solution (take all x=1)
            # or the ceil (take all x=1 plus the fractional one, if it fits, and drop something else? No, 0/1).
            # Actually, a standard approach is:
            # 1. Solve LP.
            # 2. Let S1 = {i | x_i = 1}, S2 = {i | x_i = 0}, Sf = {i | 0 < x_i < 1}.
            # 3. The optimal 0/1 solution is either S1 or S1 U {i in Sf}.
            #    Wait, S1 U {i in Sf} might not fit if the fractional item doesn't fit in the remaining capacity?
            #    But x_i is defined such that sum w_i x_i = C. So sum_{i in S1} w_i + x_f w_f = C.
            #    So sum_{i in S1} w_i < C. The remaining capacity is C - sum_{i in S1} w_i = x_f w_f.
            #    Since w_f > x_f w_f (as x_f < 1), the item f does NOT fit in the remaining capacity of S1.
            #    So S1 U {f} is infeasible.
            #    
            #    Wait, the LP solution has sum w_i x_i = C.
            #    If we take all items with x_i=1, let W1 = sum_{i in S1} w_i.
            #    Remaining capacity C_rem = C - W1.
            #    The fractional item f has weight w_f.
            #    We know x_f w_f = C_rem. So w_f = C_rem / x_f. Since x_f < 1, w_f > C_rem.
            #    So item f does not fit in the remaining capacity.
            #    
            #    So the set S1 is feasible.
            #    What about other items?
            #    The optimal 0/1 solution must be a subset of the items.
            #    A known property is that the optimal 0/1 solution value is between the LP value minus max value item and LP value.
            #    
            #    Heuristic:
            #    1. Compute S1 (items with x_i >= 1 - eps).
            #    2. Check if any item from S2 (x_i < eps) fits in the remaining capacity of S1.
            #       If yes, pick the best one (by value).
            #    3. If no item from S2 fits, check if we can swap an item from S1 with an item from S2 to get better value?
            #       This is getting complex.
            #    
            #    Simpler Heuristic:
            #    Just pick the item with the highest x_i.
            #    If x_i is 1, it's a "sure" item.
            #    If x_i is fractional, it's the "critical" item.
            #    Picking the fractional item is often a good move because it's the one that determines the boundary.
            #    However, if we pick it, we might not be able to pick other high-value items that were excluded?
            #    
            #    Let's try a different approach:
            #    Pick the item that maximizes value, subject to being in the "core" (x_i > 0).
            #    Or, simply pick the item with the highest density among those with x_i > 0?
            #    
            #    Actually, the most robust simple heuristic from LP is:
            #    Pick the item with the largest x_i.
            #    If there are multiple items with x_i=1, pick the one with the highest value.
            #    
            #    Let's implement:
            #    1. Identify items with x_i > 0.5 (effectively 1 or close to 1).
            #    2. If there's an item with 0 < x_i < 0.5, it's the fractional one.
            #    3. Candidate set: Items with x_i > 0.5 AND the fractional item.
            #    4. From this candidate set, pick the one with the highest value.
            #       Why value? Because if an item has x_i=1, it's fully taken. If we must pick one,
            #       picking a high-value one is good.
            #       But wait, if we pick an item with x_i=1, we are "committing" to it.
            #       If we pick the fractional item, we are "committing" to it, which might force us to drop
            #       some x_i=1 items later? No, we are just picking ONE item now.
            #       The rest of the search will handle the rest.
            #       
            #       Actually, the "Core" of the knapsack instance is the set of items that appear in the LP solution.
            #       The optimal 0/1 solution is contained in the Core.
            #       So picking any item from the Core is better than picking an item outside.
            #       Among Core items, which one to pick?
            #       Picking the one with the highest value seems reasonable.
            #       Or the one with the highest density?
            #       
            #       Let's try: Pick the item in the Core with the highest value.
            
            eps = 1e-6
            core_mask = x > eps
            if not np.any(core_mask):
                return None
            
            core_indices = np.where(core_mask)[0]
            core_values = v_feas[core_mask]
            
            # Pick the item in the core with the highest value
            best_local_idx = np.argmax(core_values)
            return int(core_indices[best_local_idx])
            
        else:
            return None
    except Exception:
        return None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using LP relaxation (Fractional Knapsack)
    to identify the "core" items, then picks the highest value item from the core.
    Returns the index into remaining_weights/remaining_values, or -1 to stop.
    """
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights, dtype=np.float64)
        v = np.array(remaining_values, dtype=np.float64)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    
    # Solve LP
    pick_local = _solve_lp(C, w_feas, v_feas)
    
    if pick_local is not None:
        return int(feasible_indices[pick_local])
    
    # Fallback if LP fails
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_feas / w_feas
    densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
    best_local = np.argmax(densities)
    return int(feasible_indices[best_local])
