import time
import numpy as np
from scipy.optimize import linprog
import sys

# EVOLVE-BLOCK-START
def construct_auction(distribution_spec, support, constraints):
    """
    Incentive-compatible finite auction design with explicit candidate API.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Helper: check deadline
    def time_up():
        return time.time() > deadline

    # 1. Parse inputs
    # support is a list of (v1, v2) pairs, length 16.
    # distribution_spec is likely a dict mapping (v1, v2) to probability, or a 4x4 matrix, or a list of 16 probs.
    # constraints: we assume standard individual rationality and incentive compatibility.
    # Let's infer the structure from 'support'.
    
    if not support:
        return {"allocations": np.zeros((0, 2)), "payments": np.zeros(0)}
        
    N = len(support)
    if N != 16:
        # The task says "each value in {0,1,2,3}", so 4x4=16 reports.
        # If N is different, we might need to handle it, but the task says "fixed distributions".
        # We'll proceed with whatever N is provided, but assume 16 for the grid.
        pass

    # Extract v1 and v2 values from support
    # support is a list of tuples or lists
    vals = np.array(support)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 2)
    v1 = vals[:, 0]
    v2 = vals[:, 1]

    # Parse probabilities
    probs = None
    if isinstance(distribution_spec, dict):
        probs = np.zeros(N)
        for i, rep in enumerate(support):
            key = tuple(rep) if not isinstance(rep, tuple) else rep
            probs[i] = distribution_spec.get(key, 0.0)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == N:
            probs = np.array(distribution_spec, dtype=float)
        elif len(distribution_spec) == 4:
            # Maybe it's a marginal for one good? Or 4x4?
            # If it's a 4-element list, it might be for a single good, but we have two.
            # The task says "three fixed distributions". This implies distribution_spec might select one of them.
            # However, the interface passes distribution_spec directly.
            # Let's assume if it's length 4, it's a mistake or a specific marginal.
            # Given "one additive bidder, two goods", and support 16, it's likely a joint distribution.
            # If distribution_spec is length 4, maybe it's a product of two identical marginals?
            # Let's try to be robust. If len is 4, assume it's the marginal for both, independent.
            p_marg = np.array(distribution_spec, dtype=float)
            p_marg = p_marg / p_marg.sum() if p_marg.sum() > 0 else np.ones(4)/4
            # Build joint
            probs = np.zeros(N)
            for i, (a, b) in enumerate(support):
                probs[i] = p_marg[int(a)] * p_marg[int(b)]
        elif len(distribution_spec) == 16:
            probs = np.array(distribution_spec, dtype=float)
        else:
            # Fallback: Uniform
            probs = np.ones(N) / N
    else:
        # Fallback: Uniform
        probs = np.ones(N) / N
        
    # Normalize probabilities
    p_sum = probs.sum()
    if p_sum > 1e-12:
        probs = probs / p_sum
    else:
        probs = np.ones(N) / N

    # 2. Define LP Variables
    # x1[i], x2[i]: allocation probabilities for good 1 and 2 at report i
    # p[i]: payment at report i
    # Total variables: 3*N
    
    # Objective: Maximize Expected Revenue = sum_i probs[i] * p[i]
    # c vector for linprog (minimization): -sum_i probs[i] * p[i]
    c = np.zeros(3 * N)
    # p is at indices 2*N to 3*N
    c[2*N : 3*N] = -probs

    # Bounds
    # x1, x2 in [0, 1]
    # p in [0, infinity)
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, None)] * N

    # 3. Constraints
    A_ub = []
    b_ub = []

    # Individual Rationality (IR): p[i] <= v1[i]*x1[i] + v2[i]*x2[i]
    # => -v1[i]*x1[i] - v2[i]*x2[i] + p[i] <= 0
    A_ir = np.zeros((N, 3 * N))
    for i in range(N):
        A_ir[i, i] = -v1[i]          # -v1[i] * x1[i]
        A_ir[i, N + i] = -v2[i]      # -v2[i] * x2[i]
        A_ir[i, 2 * N + i] = 1.0     # + p[i]
    A_ub.append(A_ir)
    b_ub.append(np.zeros(N))

    # Incentive Compatibility (IC)
    # For all true types k and reports r:
    # u(k, k) >= u(k, r)
    # v1[k]*x1[k] + v2[k]*x2[k] - p[k] >= v1[k]*x1[r] + v2[k]*x2[r] - p[r]
    # p[r] - p[k] - v1[k]*(x1[r] - x1[k]) - v2[k]*(x2[r] - x2[k]) <= 0
    # p[r] - p[k] - v1[k]*x1[r] + v1[k]*x1[k] - v2[k]*x2[r] + v2[k]*x2[k] <= 0
    
    # This creates N*N constraints. 16*16 = 256 constraints. Manageable.
    A_ic = np.zeros((N * N, 3 * N))
    b_ic = np.zeros(N * N)
    
    idx = 0
    for k in range(N):  # True type
        for r in range(N):  # Reported type
            # Coefficients for x1[r]
            A_ic[idx, r] = -v1[k]
            # Coefficients for x2[r]
            A_ic[idx, N + r] = -v2[k]
            # Coefficients for x1[k]
            A_ic[idx, k] = v1[k]
            # Coefficients for x2[k]
            A_ic[idx, N + k] = v2[k]
            # Coefficients for p[r]
            A_ic[idx, 2 * N + r] = 1.0
            # Coefficients for p[k]
            A_ic[idx, 2 * N + k] = -1.0
            
            b_ic[idx] = 0.0
            idx += 1
            
    A_ub.append(A_ic)
    b_ub.append(b_ic)

    A_ub = np.vstack(A_ub)
    b_ub = np.concatenate(b_ub)

    # 4. Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback: Zero allocation, zero payment
            allocations = np.zeros((N, 2))
            payments = np.zeros(N)
            return {
                "allocations": allocations,
                "payments": payments
            }
        
        x_sol = res.x
        x1 = x_sol[:N]
        x2 = x_sol[N:2*N]
        p = x_sol[2*N:3*N]
        
        # Clip small numerical errors
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        p = np.clip(p, 0, None)
        
        # Ensure IR holds strictly (clip p if it slightly exceeds value)
        val = v1 * x1 + v2 * x2
        p = np.minimum(p, val)
        
        allocations = np.column_stack((x1, x2))
        
        return {
            "allocations": allocations,
            "payments": p
        }
        
    except Exception as e:
        # Fallback on error
        allocations = np.zeros((N, 2))
        payments = np.zeros(N)
        return {
            "allocations": allocations,
            "payments": payments
        }

# EVOLVE-BLOCK-END
