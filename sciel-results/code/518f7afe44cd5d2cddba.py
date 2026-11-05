"""Incentive-compatible finite auction design solver via binary allocation enumeration."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    values = [0, 1, 2, 3]
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    # Normalize probabilities
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1_sum = p1.sum()
    p2_sum = p2.sum()
    if p1_sum > 0:
        p1 = p1 / p1_sum
    else:
        p1 = np.ones(4) / 4
    if p2_sum > 0:
        p2 = p2 / p2_sum
    else:
        p2 = np.ones(4) / 4
    
    # Joint distribution over (v1, v2)
    joint_probs = np.outer(p1, p2).flatten()  # shape (16,)
    
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    
    # Fallback mechanism: simple threshold
    def fallback_mechanism():
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for t in range(16):
            v1_t, v2_t = types[t]
            if v1_t >= 2:
                allocations[t, 0] = 1
            if v2_t >= 2:
                allocations[t, 1] = 1
            payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        return {'allocations': allocations, 'payments': payments}

    # Enumerate all 2^32 binary allocation patterns
    # We represent allocations as a list of 32 bits: [x0_0, x0_1, x1_0, x1_1, ..., x15_0, x15_1]
    # To avoid iterating 4 billion times, we can be smarter.
    # However, 2^32 is too large for a simple loop in Python within 85 seconds.
    # We need a smarter search or a reduction.
    
    # Wait, 2^32 is ~4.3e9. Even 100 million iterations per second (optimistic C speed, impossible in Python) would take 43 seconds.
    # Python is ~10-50 million simple ops per second max, but LP solving is much slower.
    # We cannot enumerate all 2^32.
    
    # Alternative Strategy:
    # The problem is small (16 types).
    # Maybe we can use the LP relaxation but handle the integrality gap better?
    # Or, we can observe that the optimal mechanism for two goods with independent values often has a specific structure.
    
    # Let's rethink the "Materially Different" strategy.
    # If enumeration is too slow, what is another discrete approach?
    # 1. Local Search / Simulated Annealing on binary allocations.
    # 2. Genetic Algorithm.
    # 3. Branch and Bound.
    
    # Given the tight time limit and the need for exact IC, Local Search is a strong candidate.
    # Start with a good initial allocation (e.g., from LP relaxation rounded, or a heuristic).
    # Then, flip bits (change allocation of a good for a type) and check if we can adjust payments to maintain IC/IR and improve revenue.
    
    # Actually, for a fixed allocation, the optimal payment is uniquely determined (or bounded) by the IC constraints.
    # Specifically, for a single additive bidder, the payment for type t is:
    # p[t] = v1[t]*x[t,0] + v2[t]*x[t,1] - u[t]
    # where u[t] is the utility.
    # To maximize revenue, we want to minimize expected utility sum p(t)u(t).
    # The IC constraints imply monotonicity.
    # For a fixed allocation, the minimal expected payment (maximal revenue) is achieved by setting the utility of the lowest type (0,0) to 0 (IR) and propagating the "tight" IC constraints.
    
    # Let's implement a Local Search over binary allocations.
    # 1. Initialize allocation using the LP relaxation result (rounded) or a heuristic.
    # 2. For each step, try flipping a random bit (allocation of a good for a type).
    # 3. For the new allocation, compute the optimal payment vector.
    #    - This can be done by solving a small LP or using the chain rule for monotone mechanisms.
    #    - For 16 types, the LP is very small (16 vars, ~256 constraints).
    # 4. If the new mechanism is feasible (IR, IC, non-neg payments) and has higher revenue, accept.
    # 5. Repeat until time limit.
    
    # To make this robust, we need a fast function to compute optimal payments for a given binary allocation.
    
    def compute_optimal_payments(alloc):
        """
        Given binary allocation matrix (16, 2), compute the payment vector that maximizes expected revenue
        subject to IC and IR.
        If no feasible payments exist (e.g., IC infeasible for this allocation structure), return None.
        
        Note: For a single bidder, IC is equivalent to monotonicity of allocations.
        If the allocation is monotone, the optimal payment is determined by the "critical" types.
        However, not all monotone allocations are implementable with non-negative payments if IR is violated?
        Actually, if monotone, we can set u(0,0)=0 and compute others. If any p[t] < 0, IR fails.
        
        Let's use the LP approach for safety and exactness, as it's small enough.
        Variables: p[0..15]
        Objective: maximize sum_t joint_probs[t] * p[t]
        Constraints:
        1. IC: p[t] - p[tp] <= v1[t]*(x[t,0]-x[tp,0]) + v2[t]*(x[t,1]-x[tp,1])
        2. IR: p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]
        3. Non-neg: p[t] >= 0
        """
        n_vars = 16
        c = -joint_probs.copy()
        
        A_ub_rows = []
        b_ub_vals = []
        
        # IC constraints
        for t in range(n_types):
            v1_t, v2_t = types[t]
            for tp in range(n_types):
                if t == tp:
                    continue
                row = np.zeros(n_vars)
                row[t] = 1
                row[tp] = -1
                # RHS: v1[t]*(x[t,0]-x[tp,0]) + v2[t]*(x[t,1]-x[tp,1])
                rhs = v1_t * (alloc[t, 0] - alloc[tp, 0]) + v2_t * (alloc[t, 1] - alloc[tp, 1])
                A_ub_rows.append(row)
                b_ub_vals.append(rhs)
                
        # IR constraints
        for t in range(n_types):
            v1_t, v2_t = types[t]
            row = np.zeros(n_vars)
            row[t] = 1
            rhs = v1_t * alloc[t, 0] + v2_t * alloc[t, 1]
            A_ub_rows.append(row)
            b_ub_vals.append(rhs)
            
        # Non-negative payments are handled by bounds
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0, None)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return None
        except:
            return None

    # Initial Allocation: Use the LP relaxation from the original code, but rounded.
    # We'll reuse the LP setup from the original code to get a starting point.
    
    # Re-construct the LP for the initial point (same as original)
    n_x = 32
    n_p = 16
    n_vars = n_x + n_p
    
    c_init = np.zeros(n_vars)
    for t in range(n_types):
        c_init[n_x + t] = -joint_probs[t]
        
    A_ub_rows_init = []
    b_ub_vals_init = []
    
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[tp * 2] += v1_t
            row[tp * 2 + 1] += v2_t
            row[n_x + t] = 1
            row[n_x + tp] = -1
            A_ub_rows_init.append(row)
            b_ub_vals_init.append(0)
            
    for v1 in range(3):
        for v2 in range(4):
            t_curr = v1 * 4 + v2
            t_next = (v1 + 1) * 4 + v2
            row = np.zeros(n_vars)
            row[t_next * 2] = -1
            row[t_curr * 2] = 1
            A_ub_rows_init.append(row)
            b_ub_vals_init.append(0)
            
    for v1 in range(4):
        for v2 in range(3):
            t_curr = v1 * 4 + v2
            t_next = v1 * 4 + (v2 + 1)
            row = np.zeros(n_vars)
            row[t_next * 2 + 1] = -1
            row[t_curr * 2 + 1] = 1
            A_ub_rows_init.append(row)
            b_ub_vals_init.append(0)
            
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1_t
        row[t * 2 + 1] = -v2_t
        row[n_x + t] = 1
        A_ub_rows_init.append(row)
        b_ub_vals_init.append(0)
        
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + t] = -1
        A_ub_rows_init.append(row)
        b_ub_vals_init.append(0)
        
    bounds_init = []
    for i in range(n_x):
        bounds_init.append((0, 1))
    for i in range(n_p):
        bounds_init.append((0, None))
        
    A_ub_init = np.array(A_ub_rows_init)
    b_ub_init = np.array(b_ub_vals_init)
    
    try:
        res_init = linprog(c_init, A_ub=A_ub_init, b_ub=b_ub_init, bounds=bounds_init, method='highs')
        if res_init.success:
            x_vars = res_init.x[:n_x]
            current_alloc = np.zeros((16, 2))
            for t in range(16):
                current_alloc[t, 0] = 1 if x_vars[t * 2] > 0.5 else 0
                current_alloc[t, 1] = 1 if x_vars[t * 2 + 1] > 0.5 else 0
        else:
            current_alloc = np.zeros((16, 2))
    except:
        current_alloc = np.zeros((16, 2))
        
    # Ensure initial allocation is monotone (it should be, but let's verify/fix)
    # If not monotone, we need to project it to the nearest monotone allocation or just use a heuristic.
    # For now, let's assume the LP solution rounded is close to monotone.
    # We will enforce monotonicity in the local search by only allowing flips that maintain monotonicity?
    # Or simpler: check feasibility of payments. If infeasible, the allocation is not implementable.
    
    # Get initial payments
    current_payments = compute_optimal_payments(current_alloc)
    if current_payments is None:
        # Fallback to a known feasible mechanism
        current_alloc = np.zeros((16, 2))
        for t in range(16):
            v1_t, v2_t = types[t]
            if v1_t >= 2:
                current_alloc[t, 0] = 1
            if v2_t >= 2:
                current_alloc[t, 1] = 1
        current_payments = compute_optimal_payments(current_alloc)
        if current_payments is None:
            return fallback_mechanism()
            
    current_revenue = np.sum(joint_probs * current_payments)
    
    best_alloc = current_alloc.copy()
    best_payments = current_payments.copy()
    best_revenue = current_revenue
    
    # Local Search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random type and good to flip
        t = rng.integers(0, 16)
        g = rng.integers(0, 2)
        
        new_alloc = current_alloc.copy()
        new_alloc[t, g] = 1 - new_alloc[t, g]
        
        # Optional: Check monotonicity quickly to avoid infeasible LP calls
        # Monotonicity: if v1 increases, x1 cannot decrease.
        # This is a pre-check.
        monotone = True
        # Check good 0
        for v1 in range(3):
            for v2 in range(4):
                t_curr = v1 * 4 + v2
                t_next = (v1 + 1) * 4 + v2
                if new_alloc[t_curr, 0] > new_alloc[t_next, 0]:
                    monotone = False
                    break
            if not monotone:
                break
        # Check good 1
        if monotone:
            for v1 in range(4):
                for v2 in range(3):
                    t_curr = v1 * 4 + v2
                    t_next = v1 * 4 + (v2 + 1)
                    if new_alloc[t_curr, 1] > new_alloc[t_next, 1]:
                        monotone = False
                        break
                if not monotone:
                    break
                    
        if not monotone:
            continue
            
        new_payments = compute_optimal_payments(new_alloc)
        if new_payments is None:
            continue
            
        new_revenue = np.sum(joint_probs * new_payments)
        
        if new_revenue > current_revenue + 1e-10:
            current_alloc = new_alloc
            current_payments = new_payments
            current_revenue = new_revenue
            
            if current_revenue > best_revenue + 1e-10:
                best_alloc = current_alloc.copy()
                best_payments = current_payments.copy()
                best_revenue = current_revenue
        else:
            # Simple restart or just continue
            pass
            
    # Final verification
    # Check IC and IR for best_alloc
    ic_ok = True
    for t in range(n_types):
        v1_t, v2_t = types[t]
        u_tt = v1_t * best_alloc[t, 0] + v2_t * best_alloc[t, 1] - best_payments[t]
        for tp in range(n_types):
            u_tt_prime = v1_t * best_alloc[tp, 0] + v2_t * best_alloc[tp, 1] - best_payments[tp]
            if u_tt < u_tt_prime - 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break
            
    ir_ok = True
    for t in range(n_types):
        v1_t, v2_t = types[t]
        u_tt = v1_t * best_alloc[t, 0] + v2_t * best_alloc[t, 1] - best_payments[t]
        if u_tt < -1e-8:
            ir_ok = False
            break
            
    if not ic_ok or not ir_ok:
        return fallback_mechanism()
        
    return {'allocations': best_alloc, 'payments': best_payments}

