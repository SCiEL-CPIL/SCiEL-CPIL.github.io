"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _parse_distribution(distribution_spec):
    """Parse distribution_spec into a 16-element probability vector."""
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    
    if isinstance(distribution_spec, dict):
        for key in ['type_probs', 'probs', 'probabilities', 'p']:
            if key in distribution_spec:
                arr = np.array(distribution_spec[key], dtype=float)
                if len(arr) == 16:
                    arr = arr / arr.sum()
                    return arr
        # Try to find a 16-element array
        for k, v in distribution_spec.items():
            if isinstance(v, (list, np.ndarray)) and len(v) == 16:
                arr = np.array(v, dtype=float)
                arr = arr / arr.sum()
                return arr
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            arr = np.array(distribution_spec, dtype=float)
            arr = arr / arr.sum()
            return arr
    
    return np.ones(16) / 16.0

def _solve_lp_auction(probs, types, deadline=None):
    """
    Solve the LP for optimal incentive-compatible auction.
    
    Variables:
    - q[i, j] for i in 0..15, j in 0..1: allocation probability for good j to type i
    - p[i] for i in 0..15: payment to type i
    
    Objective: maximize sum_i probs[i] * p[i]
    
    Constraints:
    1. Feasibility: 0 <= q[i,j] <= 1
    2. Nonnegative payments: p[i] >= 0
    3. Individual rationality: v_i . q_i - p_i >= 0 for all i
    4. Incentive compatibility: v_i . q_i - p_i >= v_i . q_j - p_j for all i, j
    
    For a single additive bidder, IC can be expressed using the envelope theorem:
    p(i) = v_i . q_i - integral of q along the path from 0 to v_i
    
    For discrete support, we use the linear IC constraints directly.
    """
    n_types = 16
    n_goods = 2
    
    # Decision variables: q[0..15, 0..1] (32 vars), p[0..15] (16 vars)
    # Total: 48 variables
    # x = [q_0_0, q_0_1, q_1_0, q_1_1, ..., q_15_0, q_15_1, p_0, p_1, ..., p_15]
    
    n_vars = 32 + 16  # 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # linprog minimizes, so minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. Feasibility: -q[i,j] <= 0 (q >= 0)
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[i * 2 + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # 2. q[i,j] <= 1
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[i * 2 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # 3. Nonnegative payments: -p[i] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # 4. Individual rationality: v_i . q_i - p_i >= 0  =>  p_i - v_i . q_i <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[32 + i] = 1.0
        row[i * 2 + 0] = -v1
        row[i * 2 + 1] = -v2
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # 5. IC: v_i . q_i - p_i >= v_i . q_j - p_j  =>  v_i . (q_i - q_j) - p_i + p_j <= 0
    # For all i, j (i != j)
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # v_i . q_i - v_i . q_j - p_i + p_j <= 0
            row[i * 2 + 0] = v1_i
            row[i * 2 + 1] = v2_i
            row[j * 2 + 0] = -v1_i
            row[j * 2 + 1] = -v2_i
            row[32 + i] = -1.0
            row[32 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= q <= 1, p >= 0 (no upper bound on p needed for LP, but let's add a reasonable one)
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        return None, None
    
    x = result.x
    q = x[:32].reshape(16, 2)
    p = x[32:48]
    
    return q, p

def _verify_and_repair(allocations, payments, probs, types, tol=1e-8):
    """Verify constraints and repair if needed. Returns (valid, allocations, payments, revenue)."""
    n_types = 16
    
    # Check feasibility
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        allocations = np.clip(allocations, 0, 1)
    
    # Check nonnegative payments
    if np.any(payments < -tol):
        payments = np.maximum(payments, 0)
    
    # Check IR: v_i . q_i - p_i >= -tol
    for i in range(n_types):
        v1, v2 = types[i]
        util = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        if util < -tol:
            # Repair: reduce payment
            payments[i] = max(0, v1 * allocations[i, 0] + v2 * allocations[i, 1])
    
    # Check IC: v_i . q_i - p_i >= v_i . q_j - p_j - tol
    max_violation = 0.0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        true_util = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
        for j in range(n_types):
            misreport_util = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
            violation = misreport_util - true_util
            if violation > max_violation:
                max_violation = violation
    
    if max_violation > tol:
        return False, allocations, payments, 0.0
    
    revenue = np.sum(probs * payments)
    return True, allocations, payments, revenue

def _local_refinement(q, p, probs, types, max_iter=100, tol=1e-10):
    """
    Local refinement using projected gradient ascent on revenue.
    We parameterize IC mechanisms via the allocation rule q and derive payments.
    For a single additive bidder, IC implies:
    p(i) = v_i . q_i - sum of integrals
    
    For discrete support, we use the fact that for additive values,
    the payment is determined by the allocation rule via the envelope theorem.
    
    Alternative: directly optimize over q with IC constraints using a penalty method.
    """
    n_types = 16
    n_goods = 2
    
    best_q = q.copy()
    best_p = p.copy()
    best_revenue = np.sum(probs * best_p)
    
    # Compute revenue as function of q
    def compute_payments_from_allocations(allocs):
        """
        For a single additive bidder with values in {0,1,2,3} for each good,
        the IC payment is determined by the allocation rule.
        
        For good j with allocation q_j(v_j):
        p_j(v_j) = v_j * q_j(v_j) - sum_{k=0}^{v_j-1} q_j(k)
        
        Total payment: p(v1, v2) = p_1(v1) + p_2(v2)
        """
        # Extract q1 and q2 as functions of v1 and v2
        # types[i] = (v1, v2), i = v1*4 + v2
        q1 = np.zeros(4)
        q2 = np.zeros(4)
        
        # For each value v, the allocation should be the same regardless of the other value
        # Check consistency and average
        for v in range(4):
            # Good 1: all types with v1 = v
            mask1 = [i for i in range(16) if types[i][0] == v]
            if mask1:
                q1[v] = np.mean([allocs[i, 0] for i in mask1])
            
            # Good 2: all types with v2 = v
            mask2 = [i for i in range(16) if types[i][1] == v]
            if mask2:
                q2[v] = np.mean([allocs[i, 1] for i in mask2])
        
        # Compute payments
        payments = np.zeros(n_types)
        for i in range(n_types):
            v1, v2 = types[i]
            pay1 = v1 * q1[v1] - np.sum(q1[:v1]) if v1 > 0 else 0.0
            pay2 = v2 * q2[v2] - np.sum(q2[:v2]) if v2 > 0 else 0.0
            payments[i] = pay1 + pay2
        
        return payments, q1, q2
    
    # Projected gradient ascent on q
    lr = 0.1
    for it in range(max_iter):
        # Compute current payments
        p_current, q1, q2 = compute_payments_from_allocations(best_q)
        
        # Revenue
        revenue = np.sum(probs * p_current)
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_p = p_current.copy()
        
        # Compute gradient of revenue w.r.t. q
        # Revenue = sum_i probs[i] * (v1_i*q1(v1_i) + v2_i*q2(v2_i) - sum_{k<v1_i}q1(k) - sum_{k<v2_i}q2(k))
        # d(Revenue)/dq1(v) = sum_{i: v1_i=v} probs[i]*v1_i - sum_{i: v1_i>v} probs[i]
        # d(Revenue)/dq2(v) = sum_{i: v2_i=v} probs[i]*v2_i - sum_{i: v2_i>v} probs[i]
        
        grad_q1 = np.zeros(4)
        grad_q2 = np.zeros(4)
        
        for i in range(n_types):
            v1, v2 = types[i]
            grad_q1[v1] += probs[i] * v1
            grad_q2[v2] += probs[i] * v2
        
        for v in range(4):
            grad_q1[v] -= np.sum([probs[i] for i in range(16) if types[i][0] > v])
            grad_q2[v] -= np.sum([probs[i] for i in range(16) if types[i][1] > v])
        
        # Update q
        new_q = best_q.copy()
        for i in range(n_types):
            v1, v2 = types[i]
            new_q[i, 0] = best_q[i, 0] + lr * grad_q1[v1]
            new_q[i, 1] = best_q[i, 1] + lr * grad_q2[v2]
        
        # Project to [0, 1]
        new_q = np.clip(new_q, 0, 1)
        
        # Check if converged
        if np.max(np.abs(new_q - best_q)) < tol:
            break
        
        best_q = new_q
    
    # Final payment computation
    final_p, q1, q2 = compute_payments_from_allocations(best_q)
    final_revenue = np.sum(probs * final_p)
    
    return best_q, final_p, final_revenue

def _solve_dp_optimal(probs, types, deadline=None):
    """
    Dynamic programming approach to find optimal IC auction.
    
    For a single additive bidder with two goods, the optimal mechanism can be
    found by separately optimizing each good's allocation rule (due to additivity
    and independence in the payment structure).
    
    For each good j, we need to find q_j(v) for v in {0,1,2,3} that maximizes
    expected payment subject to:
    - 0 <= q_j(v) <= 1
    - q_j is non-decreasing in v (monotonicity for IC)
    - IR: v*q_j(v) - sum_{k<v} q_j(k) >= 0 for all v
    
    The expected payment for good j is:
    E[p_j] = sum_v P(v) * [v*q_j(v) - sum_{k<v} q_j(k)]
    
    This can be rewritten as:
    E[p_j] = sum_v [v*P(v) - P(v>...)] * q_j(v)  (with appropriate coefficients)
    
    Actually, let's expand:
    E[p_j] = sum_v P(v) * v * q_j(v) - sum_v P(v) * sum_{k<v} q_j(k)
           = sum_v P(v) * v * q_j(v) - sum_k q_j(k) * sum_{v>k} P(v)
           = sum_k q_j(k) * [k * P(k) - sum_{v>k} P(v)]
    
    So the coefficient for q_j(k) is: c_k = k * P(k) - P(k+1) - P(k+2) - ... - P(3)
    
    For monotone q_j, this is a linear program with monotonicity constraints.
    With only 4 values, we can enumerate all possible monotone sequences.
    """
    n_types = 16
    n_goods = 2
    
    # Compute marginal distributions for each good
    p1 = np.zeros(4)
    p2 = np.zeros(4)
    for i in range(n_types):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]
    
    def solve_good_allocation(marginal_probs):
        """Solve for optimal allocation rule for one good."""
        # Coefficients for q(v)
        coeffs = np.zeros(4)
        for v in range(4):
            coeffs[v] = v * marginal_probs[v] - np.sum(marginal_probs[v+1:])
        
        # We need to find q = [q0, q1, q2, q3] with 0 <= q0 <= q1 <= q2 <= q3 <= 1
        # that maximizes sum_v coeffs[v] * q[v]
        # subject to IR: v*q[v] - sum_{k<v} q[k] >= 0
        
        # Since we have only 4 variables and monotonicity, we can use LP or enumerate
        # Let's use a simple approach: try all possible "threshold" mechanisms first
        # Then refine with LP if needed
        
        # For now, use the Myerson-style threshold:
        # q[v] = 1 if coeffs[v] > 0 else 0, but ensure monotonicity
        
        # Actually, let's solve the LP properly using scipy
        # Variables: q0, q1, q2, q3
        # Maximize: c0*q0 + c1*q1 + c2*q2 + c3*q3
        # Subject to:
        #   0 <= q0 <= q1 <= q2 <= q3 <= 1
        #   IR constraints
        
        from scipy.optimize import linprog
        
        c_obj = -coeffs  # minimize negative
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        
        # Monotonicity: q0 - q1 <= 0, q1 - q2 <= 0, q2 - q3 <= 0
        A_ub.append([1, -1, 0, 0])
        b_ub.append(0)
        A_ub.append([0, 1, -1, 0])
        b_ub.append(0)
        A_ub.append([0, 0, 1, -1])
        b_ub.append(0)
        
        # IR constraints:
        # v=0: 0*q0 >= 0 (always satisfied)
        # v=1: 1*q1 - q0 >= 0 => -q1 + q0 <= 0
        A_ub.append([-1, 1, 0, 0])
        b_ub.append(0)
        # v=2: 2*q2 - q0 - q1 >= 0 => -2*q2 + q0 + q1 <= 0
        A_ub.append([1, 1, -2, 0])
        b_ub.append(0)
        # v=3: 3*q3 - q0 - q1 - q2 >= 0 => -3*q3 + q0 + q1 + q2 <= 0
        A_ub.append([1, 1, 1, -3])
        b_ub.append(0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, 1)] * 4
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback to simple threshold
            q = np.zeros(4)
            for v in range(4):
                if coeffs[v] > 1e-12:
                    q[v] = 1.0
            # Ensure monotonicity
            for v in range(1, 4):
                q[v] = max(q[v], q[v-1])
            return q
        
        q = result.x
        # Ensure monotonicity (numerical safety)
        for v in range(1, 4):
            q[v] = max(q[v], q[v-1])
        q = np.clip(q, 0, 1)
        
        return q
    
    q1_opt = solve_good_allocation(p1)
    q2_opt = solve_good_allocation(p2)
    
    # Construct allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = q1_opt[v1]
        allocations[i, 1] = q2_opt[v2]
        
        pay1 = v1 * q1_opt[v1] - np.sum(q1_opt[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2_opt[v2] - np.sum(q2_opt[:v2]) if v2 > 0 else 0.0
        payments[i] = pay1 + pay2
    
    payments = np.maximum(payments, 0.0)
    
    return allocations, payments

def _fallback_result(probs, types):
    """Return a simple valid fallback auction."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    # Simple reserve-based: allocate if value >= 1
    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = 1.0 if v1 >= 1 else 0.0
        allocations[i, 1] = 1.0 if v2 >= 1 else 0.0
        payments[i] = (1.0 if v1 >= 1 else 0.0) + (1.0 if v2 >= 1 else 0.0)
    
    return allocations, payments

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, three fixed distributions.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    
    probs = _parse_distribution(distribution_spec)
    
    # First, try the Myerson-style independent reserve approach
    p1 = np.zeros(4)
    p2 = np.zeros(4)
    for i in range(16):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]
    
    def compute_allocation(p):
        q = np.zeros(4)
        for v in range(4):
            if v == 3:
                coeff = 3 * p[3]
            else:
                coeff = v * p[v] - np.sum(p[v+1:4])
            q[v] = 1.0 if coeff > 1e-12 else 0.0
        return q
    
    q1_myerson = compute_allocation(p1)
    q2_myerson = compute_allocation(p2)
    
    allocations_myerson = np.zeros((16, 2))
    payments_myerson = np.zeros(16)
    for i in range(16):
        v1, v2 = types[i]
        allocations_myerson[i, 0] = q1_myerson[v1]
        allocations_myerson[i, 1] = q2_myerson[v2]
        pay1 = v1 * q1_myerson[v1] - np.sum(q1_myerson[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2_myerson[v2] - np.sum(q2_myerson[:v2]) if v2 > 0 else 0.0
        payments_myerson[i] = pay1 + pay2
    
    payments_myerson = np.maximum(payments_myerson, 0.0)
    valid_myerson, alloc_m, pay_m, rev_m = _verify_and_repair(allocations_myerson.copy(), payments_myerson.copy(), probs, types)
    
    best_alloc = alloc_m.copy()
    best_pay = pay_m.copy()
    best_rev = rev_m if valid_myerson else 0.0
    
    # Try DP-based approach if time permits
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 10:
        try:
            alloc_dp, pay_dp = _solve_dp_optimal(probs, types)
            valid_dp, alloc_dp, pay_dp, rev_dp = _verify_and_repair(alloc_dp.copy(), pay_dp.copy(), probs, types)
            if valid_dp and rev_dp > best_rev:
                best_alloc = alloc_dp.copy()
                best_pay = pay_dp.copy()
                best_rev = rev_dp
        except Exception:
            pass
    
    # Try LP-based approach if time permits
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 10:
        try:
            q_lp, p_lp = _solve_lp_auction(probs, types)
            if q_lp is not None:
                valid_lp, alloc_lp, pay_lp, rev_lp = _verify_and_repair(q_lp.copy(), p_lp.copy(), probs, types)
                if valid_lp and rev_lp > best_rev:
                    best_alloc = alloc_lp.copy()
                    best_pay = pay_lp.copy()
                    best_rev = rev_lp
        except Exception:
            pass
    
    # Try local refinement if time permits
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5:
        try:
            # Start from best known
            q_start = best_alloc.copy()
            p_start = best_pay.copy()
            q_ref, p_ref, rev_ref = _local_refinement(q_start, p_start, probs, types, max_iter=200)
            
            valid_ref, alloc_ref, pay_ref, rev_ref_checked = _verify_and_repair(q_ref.copy(), p_ref.copy(), probs, types)
            if valid_ref and rev_ref_checked > best_rev:
                best_alloc = alloc_ref.copy()
                best_pay = pay_ref.copy()
                best_rev = rev_ref_checked
        except Exception:
            pass
    
    # Final verification
    valid_final, alloc_final, pay_final, rev_final = _verify_and_repair(best_alloc.copy(), best_pay.copy(), probs, types)
    
    if not valid_final:
        # Use fallback
        alloc_final, pay_final = _fallback_result(probs, types)
        valid_final, alloc_final, pay_final, rev_final = _verify_and_repair(alloc_final, pay_final, probs, types)
    
    return {'allocations': alloc_final, 'payments': pay_final}
# EVOLVE-BLOCK-END
