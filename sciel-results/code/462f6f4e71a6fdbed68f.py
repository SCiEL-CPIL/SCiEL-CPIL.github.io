"""
Myerson-optimal finite auction for one additive bidder, two goods.
Values in {0,1,2,3} for each good. Three fixed distributions.

Approach: Solve the full mechanism design LP directly.
Variables: a[i,j] for allocation probability of good j to type i, p[i] for payment.
Constraints: feasibility (0<=a<=1), nonnegative payments, IR, IC.
Objective: maximize expected revenue.
"""
import numpy as np
from scipy.optimize import linprog


def _get_distribution(distribution_spec):
    """Return the probability vector over the 16 value types."""
    # Value types indexed by (v1, v2) with v1, v2 in {0,1,2,3}
    # Index = v1 * 4 + v2
    if distribution_spec == "uniform":
        probs = np.ones(16) / 16.0
    elif distribution_spec == "independent_bernoulli":
        probs = np.zeros(16)
        for v1 in range(4):
            for v2 in range(4):
                if v1 in (0, 1) and v2 in (0, 1):
                    p1 = 0.5 if v1 == 1 else 0.5
                    p2 = 0.5 if v2 == 1 else 0.5
                    probs[v1 * 4 + v2] = p1 * p2
                else:
                    probs[v1 * 4 + v2] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
    elif distribution_spec == "correlated":
        probs = np.zeros(16)
        probs[0 * 4 + 0] = 0.2
        probs[1 * 4 + 1] = 0.3
        probs[2 * 4 + 2] = 0.3
        probs[3 * 4 + 3] = 0.2
        total = probs.sum()
        if total > 0:
            probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    return probs


def _solve_optimal_auction(distribution_spec, support, constraints):
    """
    Solve the optimal auction LP for a single additive bidder with two goods.
    Values in {0,1,2,3} for each good.
    """
    probs = _get_distribution(distribution_spec)
    
    # Value types: (v1, v2) for v1, v2 in 0..3
    # Index i = v1 * 4 + v2
    value_types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    values = np.array([[v1, v2] for v1, v2 in value_types], dtype=float)
    
    # Variables:
    # a[i, j] for i in 0..15, j in 0..1: allocation probability for good j given type i
    # p[i] for i in 0..15: payment for type i
    # Total variables: 16*2 + 16 = 48
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # Minimize: -sum_i probs[i] * p[i]
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -probs[i]
    
    # Build constraint matrix A_ub such that A_ub @ x <= b_ub
    # x = [a[0,0], a[0,1], ..., a[15,0], a[15,1], p[0], ..., p[15]]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Feasibility: a[i,j] <= 1
    for i in range(16):
        for j in range(2):
            row = np.zeros(48)
            row[i * 2 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # IR: p[i] <= values[i]·a[i]
    # p[i] - values[i,0]*a[i,0] - values[i,1]*a[i,1] <= 0
    for i in range(16):
        row = np.zeros(48)
        row[32 + i] = 1.0
        row[i * 2 + 0] = -values[i, 0]
        row[i * 2 + 1] = -values[i, 1]
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC: for all i, j (i != j):
    # p[i] - p[j] <= values[i]·a[i] - values[i]·a[j]
    # p[i] - p[j] - values[i,0]*a[i,0] - values[i,1]*a[i,1] + values[i,0]*a[j,0] + values[i,1]*a[j,1] <= 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(48)
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            row[i * 2 + 0] = -values[i, 0]
            row[i * 2 + 1] = -values[i, 1]
            row[j * 2 + 0] = values[i, 0]
            row[j * 2 + 1] = values[i, 1]
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= a[i,j] <= 1, 0 <= p[i] <= inf
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback: no allocation, no payment
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return allocations, payments
    
    x = res.x
    allocations = x[:32].reshape(16, 2)
    payments = x[32:48]
    
    # Clip to ensure numerical feasibility
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    return allocations, payments


def construct_auction(distribution_spec, support, constraints):
    """
    Construct the optimal auction mechanism.
    
    Args:
        distribution_spec: string specifying the distribution
        support: the value support
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    allocations, payments = _solve_optimal_auction(distribution_spec, support, constraints)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
