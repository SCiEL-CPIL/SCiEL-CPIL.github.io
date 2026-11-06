"""Incentive-compatible finite auction design via direct LP solve with tight tolerances."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find a list of 16 values
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 16:
                    probs = np.array(val, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    probs = probs / probs.sum()
    
    # Parse support - should be 16 (v1, v2) pairs
    if support is None:
        support = [(i, j) for i in range(4) for j in range(4)]
    elif isinstance(support, dict):
        # Try common keys
        for key in ['values', 'support', 'types', 'reports']:
            if key in support:
                support = support[key]
                break
        else:
            support = [(i, j) for i in range(4) for j in range(4)]
    
    support = list(support)
    n_reports = len(support)
    
    if n_reports != 16:
        # Resample or pad
        if n_reports > 16:
            support = support[:16]
            n_reports = 16
        else:
            # Pad with remaining combinations
            existing = set(support)
            for i in range(4):
                for j in range(4):
                    if (i, j) not in existing:
                        support.append((i, j))
                        if len(support) == 16:
                            break
                if len(support) == 16:
                    break
            n_reports = 16
    
    # Ensure probs matches support length
    if len(probs) != n_reports:
        probs = np.ones(n_reports) / n_reports
    
    # Extract values: support[i] = (v1_i, v2_i)
    values = np.array(support, dtype=float)  # shape (16, 2)
    
    # The LP formulation for incentive-compatible auction design:
    # Variables: x_{i,j} = allocation probability for good j when report i is made
    #            p_i = payment when report i is made
    # 
    # For a single additive bidder with 2 goods:
    # Utility when true type is i and reports j: u(i,j) = v1_i * x_{j,1} + v2_i * x_{j,2} - p_j
    # 
    # Incentive compatibility: u(i,i) >= u(i,j) for all i,j
    # Individual rationality: u(i,i) >= 0 for all i
    # Feasibility: 0 <= x_{i,j} <= 1 for all i,j
    # Non-negative payments: p_i >= 0
    #
    # Objective: maximize sum_i probs[i] * p_i
    #
    # Standard approach: Use the fact that for IC mechanisms, we can express
    # payments in terms of allocations. The optimal mechanism can be found by:
    #
    # For each report i, the allocation x_i = (x_{i,1}, x_{i,2}) must satisfy IC.
    # The payment p_i is determined by: p_i = max_j [v1_i * x_{j,1} + v2_i * x_{j,2} - u_i]
    # where u_i is the utility at type i.
    #
    # A cleaner LP: Maximize revenue subject to IC and IR constraints.
    # Let's use the direct LP with variables x_{i,j} and p_i.
    
    n = n_reports  # 16
    # Variables: x_{0,1}, x_{0,2}, ..., x_{15,1}, x_{15,2}, p_0, ..., p_15
    # Total: 16*2 + 16 = 48 variables
    
    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so minimize -sum_i probs[i] * p_i
    c = np.zeros(2 * n + n)
    for i in range(n):
        c[2 * n + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 
    # IC constraints: For each true type i and report j:
    #   v1_i * x_{j,1} + v2_i * x_{j,2} - p_j <= v1_i * x_{i,1} + v2_i * x_{i,2} - p_i
    #   => v1_i * x_{j,1} + v2_i * x_{j,2} - p_j - v1_i * x_{i,1} - v2_i * x_{i,2} + p_i <= 0
    #
    # This gives n*n = 256 constraints (but i=j gives 0<=0, so 240 non-trivial)
    
    ic_constraints = []
    ic_bounds = []
    
    for i in range(n):  # true type
        for j in range(n):  # reported type
            if i == j:
                continue
            row = np.zeros(2 * n + n)
            # v1_i * x_{j,1} + v2_i * x_{j,2} - p_j - v1_i * x_{i,1} - v2_i * x_{i,2} + p_i <= 0
            row[2 * j + 0] += values[i, 0]
            row[2 * j + 1] += values[i, 1]
            row[2 * n + j] -= 1
            row[2 * i + 0] -= values[i, 0]
            row[2 * i + 1] -= values[i, 1]
            row[2 * n + i] += 1
            ic_constraints.append(row)
            ic_bounds.append(0.0)
    
    # IR constraints: u(i,i) >= 0
    #   v1_i * x_{i,1} + v2_i * x_{i,2} - p_i >= 0
    #   => -v1_i * x_{i,1} - v2_i * x_{i,2} + p_i <= 0
    for i in range(n):
        row = np.zeros(2 * n + n)
        row[2 * i + 0] = -values[i, 0]
        row[2 * i + 1] = -values[i, 1]
        row[2 * n + i] = 1
        ic_constraints.append(row)
        ic_bounds.append(0.0)
    
    # Allocation feasibility: 0 <= x_{i,j} <= 1
    # These are handled by bounds in linprog
    
    if ic_constraints:
        A_ub = np.array(ic_constraints)
        b_ub = np.array(ic_bounds)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds: 0 <= x_{i,j} <= 1, p_i >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_{i,1}
        bounds.append((0, 1))  # x_{i,2})
    for i in range(n):
        bounds.append((0, None))  # p_i >= 0
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        x_vars = result.x[:2 * n]
        p_vars = result.x[2 * n:]
        
        allocations = x_vars.reshape(n, 2)
        payments = p_vars
    else:
        # Fallback: simple mechanism - allocate both goods with probability 0.5, payment = 1
        allocations = np.full((n, 2), 0.5)
        payments = np.full(n, 1.0)
    
    # Verify and fix any numerical issues
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Check deadline
    if time.time() > deadline:
        return {
            'allocations': allocations,
            'payments': payments
        }
    
    # Verify IC and IR constraints, fix if violated
    max_violation = 0
    for i in range(n):
        # IR
        util_i = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1] - payments[i]
        if util_i < -1e-10:
            payments[i] = max(0, values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1])
            max_violation = max(max_violation, -util_i)
        
        # IC
        for j in range(n):
            util_ij = values[i, 0] * allocations[j, 0] + values[i, 1] * allocations[j, 1] - payments[j]
            util_ii = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1] - payments[i]
            if util_ij > util_ii + 1e-8:
                # Adjust payment to fix
                payments[j] = max(payments[j], values[i, 0] * allocations[j, 0] + values[i, 1] * allocations[j, 1] - util_ii)
                max_violation = max(max_violation, util_ij - util_ii)
    
    # If violations were large, try a safer mechanism
    if max_violation > 1e-6:
        # Use a simple VCG-like mechanism
        # For additive bidder: allocate good j if value_j > threshold
        # Simple approach: allocate each good with prob proportional to value
        for i in range(n):
            total_val = values[i, 0] + values[i, 1]
            if total_val > 0:
                allocations[i, 0] = values[i, 0] / total_val
                allocations[i, 1] = values[i, 1] / total_val
            else:
                allocations[i, 0] = 0
                allocations[i, 1] = 0
            payments[i] = 0.0
        
        # Recompute payments to ensure IC and IR
        # For IC: p_i >= max_j [v1_i * x_{j,1} + v2_i * x_{j,2} - v1_i * x_{i,1} - v2_i * x_{i,2}]
        # For IR: p_i <= v1_i * x_{i,1} + v2_i * x_{i,2}
        for i in range(n):
            own_val = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1]
            max_dev = 0
            for j in range(n):
                dev_val = values[i, 0] * allocations[j, 0] + values[i, 1] * allocations[j, 1] - own_val
                if dev_val > max_dev:
                    max_dev = dev_val
            payments[i] = max(0, max_dev)
            # Ensure IR
            if payments[i] > own_val:
                payments[i] = own_val
    
    # Final verification
    for i in range(n):
        util_i = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1] - payments[i]
        if util_i < -1e-8:
            payments[i] = max(0, values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1] - 1e-9)
        for j in range(n):
            util_ij = values[i, 0] * allocations[j, 0] + values[i, 1] * allocations[j, 1] - payments[j]
            util_ii = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1] - payments[i]
            if util_ij > util_ii + 1e-8:
                payments[j] = values[i, 0] * allocations[j, 0] + values[i, 1] * allocations[j, 1] - util_ii
    
    # Final cleanup
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
