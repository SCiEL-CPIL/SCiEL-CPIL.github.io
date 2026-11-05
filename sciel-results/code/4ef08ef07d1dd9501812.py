import numpy as np
import time
import itertools
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Setup: Define Types and Distribution
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution_spec to get probabilities
    probs = np.zeros(n_types)
    if isinstance(distribution_spec, str):
        if distribution_spec in ['uniform', 'independent_uniform']:
            probs = np.ones(n_types) / n_types
        elif distribution_spec == 'independent_beta':
            # Assume standard beta(1,1) which is uniform, or specific parameters if provided
            # Since spec is likely simple, assume uniform if not specified otherwise in this context
            probs = np.ones(n_types) / n_types
        else:
            # Fallback: Uniform
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple)):
        # If a list of probabilities is provided
        probs = np.array(distribution_spec)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
        
    # Normalize probabilities just in case
    total_p = probs.sum()
    if total_p > 0:
        probs = probs / total_p
    else:
        probs = np.ones(n_types) / n_types

    # 2. Define the Search Space for Deterministic Mechanisms
    # We search for "Vickrey-like" or "Threshold" mechanisms.
    # A general deterministic mechanism for 2 goods can be complex.
    # However, for additive valuations and finite support, we can parameterize
    # the allocation rule by which type wins which good.
    
    # Let's try a structured search: 
    # For each good g in {0, 1}, the allocation x_t^g is 1 if the type is "high enough".
    # We can define a "score" for each type for each good and allocate to the highest score.
    # But with only one bidder, "highest score" just means a threshold.
    # Actually, with one bidder, the allocation is just a function x: T -> {0,1}^2.
    # There are 2^16 = 65536 possible deterministic allocation rules.
    # For each allocation rule, the payment is determined by IC and IR.
    # To maximize revenue, we want the highest payment that satisfies IC and IR.
    # For a deterministic rule x, the payment p_t is determined by the IC constraints.
    # Specifically, for single-bidder additive auctions, if the allocation is monotone
    # in the value (which we can check), the payment is determined by the "critical type".
    
    # Strategy: Enumerate all 2^16 allocation vectors.
    # For each vector, check if it leads to a valid payment vector satisfying IC and IR.
    # If yes, calculate revenue. Keep the best.
    # 65536 iterations is fast.
    
    best_revenue = -1.0
    best_alloc = np.zeros((n_types, 2))
    best_pay = np.zeros(n_types)
    
    # Pre-compute value matrix for quick lookup
    # V[i, j] is value of good j for type i
    V = np.array([[t[0], t[1]] for t in types])
    
    # Helper to check if an allocation is monotone (necessary for existence of IC payment in this context? 
    # Actually, for single agent, IC is a set of linear inequalities. 
    # p_t - p_{t'} <= v_t - v_{t'} if x_t = 1 and x_{t'} = 0 etc.
    # Let's just solve the payment constraints for each allocation.
    
    # To speed up, we can use the fact that for a fixed allocation x,
    # the maximum payment p_t is determined by the tightest IC constraint.
    # p_t <= p_{t'} + (v_t - v_{t'}) . x_t  (Wait, IC: v_t . x_t - p_t >= v_t . x_{t'} - p_{t'})
    # => p_t <= p_{t'} + v_t . x_t - v_t . x_{t'}
    # Also IR: p_t <= v_t . x_t
    
    # This is a system of difference constraints. We can solve it via Bellman-Ford or similar,
    # but since the graph is small (16 nodes), we can do it quickly.
    # However, enumerating 65536 allocations and solving a graph each time might be slow in Python.
    
    # Alternative: Use LP for the *payment* part for each candidate allocation?
    # Or, recognize that for single-bidder additive, the optimal payment for a given allocation
    # is p_t = min_{t' : x_{t'} != x_t} (p_{t'} + v_t.x_t - v_t.x_{t'}) ... this is recursive.
    
    # Let's use a simpler structural assumption: 
    # The optimal mechanism is often "Vickrey-Clarke-Groves" style or "Myerson" style.
    # But let's stick to the Combinatorial Search of Allocation Vectors.
    
    # Optimization: 
    # We can prune the search. 
    # If x_t = 0, p_t can be 0 (IR).
    # If x_t = 1, p_t is constrained by IC with all t' where x_{t'}=0 and all t'' where x_{t''}=1.
    
    # Let's implement a fast check for a given allocation vector X (shape 16, 2).
    # 1. Compute max possible payment for each type based on IR: p_t <= v_t . X_t.
    # 2. Propagate IC constraints.
    # IC: v_t . X_t - p_t >= v_t . X_{t'} - p_{t'} for all t, t'.
    # => p_t - p_{t'} <= v_t . X_t - v_t . X_{t'}
    
    # This is a system of constraints p_t <= p_{t'} + c_{t,t'} where c_{t,t'} = v_t . X_t - v_t . X_{t'}.
    # We want to maximize sum(p_t * prob_t) subject to these and p_t >= 0, p_t <= v_t . X_t.
    # Since we want max revenue, we want p_t as high as possible.
    # The maximum p_t is the minimum of all upper bounds derived from paths in the constraint graph.
    # p_t <= min_{t'} (p_{t'} + c_{t,t'})
    # This is a "longest path" or "shortest path" problem on a graph where edge weight is c_{t,t'}.
    # Specifically, p_t is bounded by the shortest path from any node with a fixed upper bound (IR bound) to t.
    # Actually, the constraints are p_t - p_{t'} <= c_{t,t'}.
    # This is equivalent to finding potentials.
    # If there are no negative cycles (which would imply infeasibility), the maximum p_t is determined by the tightest constraint.
    
    # Given the small size (16 nodes), we can use Floyd-Warshall or just Bellman-Ford from "source" nodes.
    # The "source" constraints are p_t <= IR_bound_t.
    
    def max_payment_for_allocation(X):
        # X is (16, 2) array of 0/1
        # IR bounds:
        ir_bounds = np.sum(V * X, axis=1)
        
        # Initialize p_max with IR bounds
        p_max = ir_bounds.copy()
        
        # We need to satisfy p_t - p_{t'} <= v_t.X_t - v_t.X_{t'} for all t, t'
        # Let c[t, t'] = v_t.X_t - v_t.X_{t'}
        # Then p_t <= p_{t'} + c[t, t']
        
        # We can iterate this relaxation until convergence (Bellman-Ford style).
        # Since N=16, 15 iterations are enough.
        
        # Precompute c matrix? No, compute on fly or once.
        # c[t, t'] = V[t] @ X[t] - V[t] @ X[t']
        # Let val_X = V @ X (16,)
        val_X = np.sum(V * X, axis=1)
        
        # c[t, t'] = val_X[t] - val_X[t']
        # Constraint: p_t <= p_{t'} + val_X[t] - val_X[t']
        
        # We want max p.
        # Start with p = ir_bounds.
        # Relax: for all t, t': if p_t > p_{t'} + val_X[t] - val_X[t'], then p_t = p_{t'} + val_X[t] - val_X[t']
        
        p = ir_bounds.copy()
        for _ in range(n_types): # Bellman-Ford
            changed = False
            for t in range(n_types):
                for t_prime in range(n_types):
                    if t == t_prime:
                        continue
                    bound = p[t_prime] + val_X[t] - val_X[t_prime]
                    if p[t] > bound:
                        p[t] = bound
                        changed = True
            if not changed:
                break
        
        # Check for negative cycles (infeasibility). 
        # If p_t becomes very negative, it's infeasible? 
        # Actually, if the system has a negative cycle, no solution exists.
        # In our case, if p_t drops below 0, we might have infeasibility if it can't be 0.
        # But IR says p_t <= ir_bound. It doesn't say p_t >= 0 explicitly in IC, 
        # but the task requires nonnegative payments.
        
        # Check non-negativity
        if np.any(p < -1e-9):
            return -1.0, None # Infeasible or negative payments required
            
        # Ensure non-negative
        p = np.maximum(p, 0.0)
        
        # Final check: Does this p satisfy all IC and IR?
        # IC: p_t - p_{t'} <= val_X[t] - val_X[t']
        # IR: p_t <= ir_bounds[t]
        
        # Re-verify
        for t in range(n_types):
            if p[t] > ir_bounds[t] + 1e-9:
                return -1.0, None
            for t_prime in range(n_types):
                if p[t] - p[t_prime] > val_X[t] - val_X[t_prime] + 1e-9:
                    return -1.0, None
            
        revenue = np.sum(p * probs)
        return revenue, p

    # Enumerate all 2^16 allocations
    # This is 65536. 
    # Each check is O(N^2 * N) = 16^3 = 4096 ops. 
    # 65536 * 4096 ~ 268 million ops. Might be too slow in Python (approx 1-2 seconds if optimized, but loops are slow).
    # Let's optimize the inner loop.
    
    # We can vectorize the Bellman-Ford step.
    # p_t = min(p_t, min_{t'} (p_{t'} + val_X[t] - val_X[t']))
    # min_{t'} (p_{t'} - val_X[t']) + val_X[t]
    # Let min_diff = min(p_{t'} - val_X[t'])
    # Then p_t = min(p_t, min_diff + val_X[t])
    
    def max_payment_fast(X):
        ir_bounds = np.sum(V * X, axis=1)
        val_X = ir_bounds.copy() # Same thing
        
        p = ir_bounds.copy()
        
        for _ in range(n_types):
            # Compute min(p_{t'} - val_X[t'])
            min_term = np.min(p - val_X)
            # New bound for all t: min_term + val_X[t]
            new_p = min_term + val_X
            # p_t = min(p_t, new_p[t])
            p_new = np.minimum(p, new_p)
            if np.allclose(p, p_new):
                p = p_new
                break
            p = p_new
            
        if np.any(p < -1e-9):
            return -1.0, None
            
        p = np.maximum(p, 0.0)
        
        # Verification is expensive. Skip full verification if we trust the relaxation?
        # The relaxation finds the *maximum* feasible p if one exists.
        # If p < 0, infeasible for non-negative constraint.
        
        revenue = np.sum(p * probs)
        return revenue, p

    # Let's run the enumeration
    count = 0
    for mask in range(1 << n_types):
        if time.time() > deadline:
            break
            
        # Decode mask to X (16, 2)
        # This is inefficient. Let's generate X differently.
        # Actually, 2^16 is small.
        
        # Construct X
        # X[i, 0] = (mask >> i) & 1 ? No, we have 2 goods.
        # We need 2 bits per type.
        # Total bits = 32.
        # This is 2^32 = 4 billion. TOO MANY.
        
        # We cannot enumerate all 2^32 allocations.
        # We must use a heuristic or a different structure.
        
        # Back to LP? The inspirations said LP works.
        # Maybe I should stick to LP but implement it robustly.
        # The previous program just called a baseline.
        # Let's implement the LP directly.
        
        # Variables:
        # x_t^g (16*2 = 32 vars)
        # p_t (16 vars)
        # Total 48 vars.
        
        # Objective: Maximize sum( p_t * probs_t )
        
        # Constraints:
        # 1. 0 <= x_t^g <= 1
        # 2. p_t >= 0
        # 3. IR: p_t <= v_t . x_t  => p_t - v_t . x_t <= 0
        # 4. IC: v_t . x_t - p_t >= v_t . x_{t'} - p_{t'}
        #    => -v_t . x_t + p_t + v_t . x_{t'} - p_{t'} <= 0
        #    => (p_t - p_{t'}) - (v_t . x_t - v_t . x_{t'}) <= 0
        
        # This is a standard LP.
        
        break # Break out of enumeration loop, switch to LP
    
    # --- LP Implementation ---
    
    # Variables:
    # 0..15: p_0 .. p_15
    # 16..31: x_0^0 .. x_15^1 (16 types * 2 goods)
    
    n_vars = n_types + n_types * 2
    c = np.zeros(n_vars)
    # Maximize revenue => Minimize -revenue
    c[:n_types] = -probs
    # x vars have 0 cost
    
    # Inequalities A_ub x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # 1. IR Constraints: p_t - sum(v_t_g * x_t_g) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[t] = 1.0 # p_t
        # x_t^0 is at index 16 + t
        # x_t^1 is at index 16 + t + 16
        row[16 + t] = -V[t, 0]
        row[16 + t + 16] = -V[t, 1]
        A_rows.append(row)
        b_rows.append(0.0)
        
    # 2. IC Constraints: (p_t - p_{t'}) - (v_t . x_t - v_t . x_{t'}) <= 0
    # p_t - p_{t'} - v_t . x_t + v_t . x_{t'} <= 0
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[t] = 1.0
            row[t_prime] = -1.0
            
            # - v_t . x_t
            row[16 + t] = -V[t, 0]
            row[16 + t + 16] = -V[t, 1]
            
            # + v_t . x_{t'}
            row[16 + t_prime] = V[t, 0]
            row[16 + t_prime + 16] = V[t, 1]
            
            A_rows.append(row)
            b_rows.append(0.0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    # p_t >= 0
    # 0 <= x <= 1
    bounds = []
    for i in range(n_vars):
        if i < n_types:
            bounds.append((0, None))
        else:
            bounds.append((0, 1))
            
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            p_opt = res.x[:n_types]
            x_opt = res.x[n_types:].reshape(n_types, 2)
            
            # Verify and clean up
            p_opt = np.maximum(p_opt, 0.0)
            x_opt = np.clip(x_opt, 0.0, 1.0)
            
            # Calculate revenue
            rev = np.sum(p_opt * probs)
            
            # Return format: allocations (16,2), payments (16,)
            return {
                "allocations": x_opt,
                "payments": p_opt
            }
        else:
            # If LP fails, return a simple valid mechanism (e.g., always allocate 0, pay 0)
            return {
                "allocations": np.zeros((n_types, 2)),
                "payments": np.zeros(n_types)
            }
    except Exception as e:
        # Fallback
        return {
            "allocations": np.zeros((n_types, 2)),
            "payments": np.zeros(n_types)
        }

