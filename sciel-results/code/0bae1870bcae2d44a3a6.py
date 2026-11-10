"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, three fixed distributions.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the support: 16 type profiles (4 values x 4 values for 2 goods)
    # Each type is (v1, v2) where vi in {0,1,2,3}
    # We enumerate all 16 types
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    # types[i] = (v1_i, v2_i), i = 0..15
    
    # distribution_spec: need to interpret. Assume it gives probabilities for each type
    # or parameters for the three fixed distributions
    # Let's assume distribution_spec is a dict or list specifying the distribution
    # Common interpretation: distribution_spec could be the probability vector over 16 types
    
    # Parse distribution_spec - try multiple formats
    if isinstance(distribution_spec, dict):
        # Might have 'type_probs' or similar
        if 'type_probs' in distribution_spec:
            probs = np.array(distribution_spec['type_probs'], dtype=float)
        elif 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        else:
            # Try to find a 16-element array
            for k, v in distribution_spec.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == 16:
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Myerson auction for single additive bidder with two goods
    # For each type i = (v1_i, v2_i):
    #   Allocation probability for good j: q_j(i) = P(v_j >= t_j) where t_j is reserve
    #   Actually, for additive values, the optimal mechanism is:
    #   - Each good is allocated independently with a reserve price
    #   - q_j(i) = 1 if v_j_i >= r_j, else 0 (for discrete case)
    #   - Payment: p(i) = sum over goods of [v_j_i * q_j(i) - integral of q_j from 0 to v_j_i]
    #   
    # For finite support {0,1,2,3}, the payment formula is:
    # p(i) = sum_j [v_j_i * q_j(i) - sum_{k=0}^{v_j_i - 1} q_j(k, other goods)]
    # But since allocation depends only on own value for each good (additive, single bidder),
    # q_j depends only on v_j.
    
    # For a single bidder with additive values over two goods, the IC mechanism is:
    # q_j(v_j) is non-decreasing in v_j
    # p(v1, v2) = v1*q1(v1) + v2*q2(v2) - sum_{k=0}^{v1-1} q1(k) - sum_{k=0}^{v2-1} q2(k)
    
    # To maximize revenue, we choose q1, q2 to maximize:
    # sum_i probs[i] * p(i) = sum_i probs[i] * [v1_i*q1(v1_i) + v2_i*q2(v2_i) - sum_{k=0}^{v1_i-1}q1(k) - sum_{k=0}^{v2_i-1}q2(k)]
    
    # This separates by good! For each good j, maximize:
    # sum_i probs[i] * [v_j_i * q_j(v_j_i) - sum_{k=0}^{v_j_i-1} q_j(k)]
    # = sum_{v=0}^{3} q_j(v) * [sum_{i: v_j_i=v} probs[i] * v - sum_{i: v_j_i > v} probs[i]]
    # = sum_{v=0}^{3} q_j(v) * [v * P(v_j=v) - P(v_j > v)]
    
    # The coefficient for q_j(v) is: v * P(v_j = v) - P(v_j > v)
    # Set q_j(v) = 1 if coefficient > 0, else 0 (since q_j in [0,1])
    
    # Compute marginal probabilities for each good
    # For good 1: P(v1 = v) for v in {0,1,2,3}
    p1 = np.zeros(4)  # P(v1 = v)
    p2 = np.zeros(4)  # P(v2 = v)
    for i in range(16):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]
    
    # Compute optimal allocation for each good
    def compute_allocation(p):
        """p[v] = P(value = v) for v in {0,1,2,3}"""
        q = np.zeros(4)
        for v in range(4):
            coeff = v * p[v] - np.sum(p[v+1:]) if v < 3 else v * p[v]
            if v == 3:
                coeff = 3 * p[3]
            else:
                coeff = v * p[v] - np.sum(p[v+1:4])
            q[v] = 1.0 if coeff > 1e-12 else 0.0
        return q
    
    q1 = compute_allocation(p1)
    q2 = compute_allocation(p2)
    
    # Build allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = q1[v1]
        allocations[i, 1] = q2[v2]
        
        # Payment: p(v1, v2) = v1*q1(v1) + v2*q2(v2) - sum_{k=0}^{v1-1} q1(k) - sum_{k=0}^{v2-1} q2(k)
        pay1 = v1 * q1[v1] - np.sum(q1[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2[v2] - np.sum(q2[:v2]) if v2 > 0 else 0.0
        payments[i] = pay1 + pay2
    
    # Ensure nonnegative payments (IR constraint)
    payments = np.maximum(payments, 0.0)
    
    # Verify IC: check that no type wants to misreport
    # For type i reporting as type j, utility = v_i . q_j - p_j
    # True utility = v_i . q_i - p_i
    # IC requires: v_i . q_i - p_i >= v_i . q_j - p_j for all i, j
    
    # Check IC and fix if needed
    max_violation = 0.0
    for i in range(16):
        v1_i, v2_i = types[i]
        true_util = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
        for j in range(16):
            v1_j, v2_j = types[j]
            misreport_util = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
            violation = misreport_util - true_util
            if violation > max_violation:
                max_violation = violation
    
    # If IC is violated (shouldn't be for Myerson), we need to adjust
    # For now, if small violation, clip
    if max_violation > 1e-8:
        # Fallback: use simple reserve-based auction
        # Set reserves to median values
        r1 = 1.0
        r2 = 1.0
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            v1, v2 = types[i]
            a1 = 1.0 if v1 >= r1 else 0.0
            a2 = 1.0 if v2 >= r2 else 0.0
            allocations[i, 0] = a1
            allocations[i, 1] = a2
            payments[i] = r1 * a1 + r2 * a2
    
    # Final check: ensure feasibility
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
