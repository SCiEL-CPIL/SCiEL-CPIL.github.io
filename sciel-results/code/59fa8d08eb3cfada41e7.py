"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Support: values in {0,1,2,3} for each of 2 goods → 16 profiles
    # Index: i = v1 * 4 + v2, where v1, v2 in {0,1,2,3}
    values = np.array([[v1, v2] for v1 in range(4) for v2 in range(4)], dtype=float)
    
    # Parse distribution_spec to get probabilities over the 16 profiles
    # distribution_spec could be a dict, list, or array
    if isinstance(distribution_spec, dict):
        # Could be {'type': 'uniform'} or {'type': 'custom', 'probs': [...]}
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'type' in distribution_spec:
            t = distribution_spec['type']
            if t == 'uniform':
                probs = np.ones(16) / 16.0
            elif t == 'degenerate':
                idx = distribution_spec.get('index', 0)
                probs = np.zeros(16)
                probs[idx] = 1.0
            else:
                probs = np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            probs = probs / probs.sum()
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # For a single additive bidder, the optimal truthful mechanism:
    # Allocation of good j must be non-decreasing in v_j (monotonicity)
    # Payment: p(v) = v·q(v) - ∫₀^v q(t)·e_j dt (for each good)
    # 
    # With discrete values {0,1,2,3}, we can enumerate monotone allocation rules.
    # For each good, the allocation probability as a function of its value must be non-decreasing.
    # 
    # Key insight: For single-bidder multi-good with additive values, the optimal mechanism
    # allocates each good independently based on the value for that good.
    # The optimal allocation for good j given value v_j is:
    #   q_j(v_j) = 1 if v_j >= threshold_j, 0 otherwise
    # where threshold_j is chosen to maximize expected revenue.
    #
    # Actually, for a single bidder, the optimal auction is a second-price-like mechanism.
    # With one bidder, the optimal mechanism sets a reserve price for each good.
    #
    # Let's think more carefully. With one bidder and two goods, additive values:
    # The bidder's value for a bundle (x1, x2) is v1*x1 + v2*x2.
    # A truthful mechanism must have allocations that are monotone in each value.
    #
    # The optimal mechanism: for each good j, choose a reservation price r_j.
    # Allocate good j iff v_j >= r_j.
    # Payment: p(v) = sum_j [v_j * 1(v_j >= r_j) - integral_0^{v_j} 1(t >= r_j) dt]
    #         = sum_j [v_j * 1(v_j >= r_j) - max(0, v_j - r_j)]
    #
    # Wait, let me re-derive. The payment identity for good j:
    # p_j(v_j) = v_j * q_j(v_j) - ∫_0^{v_j} q_j(t) dt
    #
    # If q_j(t) = 1(t >= r_j), then:
    # ∫_0^{v_j} q_j(t) dt = ∫_0^{v_j} 1(t >= r_j) dt = max(0, v_j - r_j)
    # p_j(v_j) = v_j * 1(v_j >= r_j) - max(0, v_j - r_j)
    #
    # If v_j >= r_j: p_j = v_j - (v_j - r_j) = r_j
    # If v_j < r_j: p_j = 0
    #
    # So payment is r_j if v_j >= r_j, else 0. This is a take-it-or-leave-it at price r_j.
    #
    # Expected revenue = E[sum_j r_j * 1(v_j >= r_j)]
    #
    # For each good j, choose r_j to maximize r_j * P(v_j >= r_j).
    #
    # With values in {0,1,2,3} and distribution probs over 16 profiles:
    # P(v_j = k) = sum of probs over all profiles where value of good j is k.
    
    # Compute marginal distributions for each good
    # Good 0: value is v1 = index // 4
    # Good 1: value is v2 = index % 4
    v1_vals = np.array([i // 4 for i in range(16)])
    v2_vals = np.array([i % 4 for i in range(16)])
    
    # Marginal probabilities
    p_v1 = np.zeros(4)  # P(v1 = k)
    p_v2 = np.zeros(4)  # P(v2 = k)
    for k in range(4):
        p_v1[k] = probs[v1_vals == k].sum()
        p_v2[k] = np.zeros(4)
        p_v2[k] = probs[v2_vals == k].sum()
    
    # Find optimal reserve for each good
    # For good j, choose r in {0,1,2,3} to maximize r * P(v_j >= r)
    def find_optimal_reserve(p_marginal):
        best_r = 0
        best_rev = 0.0
        for r in range(4):  # r in {0,1,2,3}
            # P(v >= r) = sum_{k=r}^{3} p_marginal[k]
            prob_ge = p_marginal[r:].sum()
            rev = r * prob_ge
            if rev > best_rev + 1e-15:
                best_rev = rev
                best_r = r
        return best_r, best_rev
    
    r1, rev1 = find_optimal_reserve(p_v1)
    r2, rev2 = find_optimal_reserve(p_v2)
    
    # Construct allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1 = v1_vals[i]
        v2 = v2_vals[i]
        # Allocate good 0 if v1 >= r1
        if v1 >= r1:
            allocations[i, 0] = 1.0
            payments[i] += r1
        # Allocate good 1 if v2 >= r2
        if v2 >= r2:
            allocations[i, 1] = 1.0
            payments[i] += r2
    
    # Verify individual rationality: utility = v·q - p >= 0 for all types
    for i in range(16):
        v = values[i]
        q = allocations[i]
        p = payments[i]
        utility = v[0] * q[0] + v[1] * q[1] - p
        if utility < -1e-10:
            # IR violated, need to adjust
            # This shouldn't happen with the take-it-or-leave-it structure,
            # but let's clamp to be safe
            pass
    
    # Verify truthfulness (incentive compatibility)
    # For each type, check that reporting truthfully maximizes utility
    # Utility of type v reporting v': u(v, v') = v·q(v') - p(v')
    # Need: u(v, v) >= u(v, v') for all v, v'
    
    def utility(v_true, v_report):
        """Utility of true type v_true when reporting v_report"""
        idx_report = int(v_report[0] * 4 + v_report[1])
        q = allocations[idx_report]
        p = payments[idx_report]
        return v_true[0] * q[0] + v_true[1] * q[1] - p
    
    # Check IC for all type pairs
    ic_violated = False
    for i in range(16):
        v_true = values[i]
        u_truthful = utility(v_true, v_true)
        for j in range(16):
            v_report = values[j]
            u_report = utility(v_true, v_report)
            if u_report > u_truthful + 1e-8:
                ic_violated = True
                break
        if ic_violated:
            break
    
    if ic_violated:
        # Fallback: use a simpler mechanism - always allocate both goods, charge 0
        # This is IR and IC (trivially, since no one can gain by misreporting if they always get the same allocation and pay 0)
        # Actually, if allocation is always (1,1) and payment is 0, then utility = v1+v2 regardless of report, so IC holds.
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
    
    # Final verification
    # 1. Nonnegative payments
    if np.any(payments < -1e-12):
        payments = np.maximum(payments, 0.0)
    
    # 2. Feasible allocations (probabilities in [0,1])
    allocations = np.clip(allocations, 0.0, 1.0)
    
    # 3. IR check
    for i in range(16):
        v = values[i]
        q = allocations[i]
        p = payments[i]
        utility = v[0] * q[0] + v[1] * q[1] - p
        if utility < -1e-10:
            # Reduce payment to maintain IR
            payments[i] = v[0] * q[0] + v[1] * q[1]
    
    # 4. IC check with regret
    max_regret = 0.0
    for i in range(16):
        v_true = values[i]
        u_truthful = v_true[0] * allocations[i, 0] + v_true[1] * allocations[i, 1] - payments[i]
        for j in range(16):
            v_report = values[j]
            u_report = v_true[0] * allocations[j, 0] + v_true[1] * allocations[j, 1] - payments[j]
            regret = u_report - u_truthful
            if regret > max_regret:
                max_regret = regret
    
    if max_regret > 1e-8:
        # IC violated, use trivial mechanism
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
    
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
