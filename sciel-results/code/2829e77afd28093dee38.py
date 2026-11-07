"""
Strategy: Convex Quadratic Programming (QP) formulation.
The objective is min_t max_k (C_k(h)), where C_k(h) is a linear function of h.
This is equivalent to min_t t subject to C_k(h) <= t for all k, and h in [0,1], sum(h)=N/2.
This is a convex QP. We use scipy.optimize.linprog or similar, but since it involves
minimizing a linear objective (t) with linear constraints, it's actually an LP.
Wait, the objective is min t. The constraints are linear.
So it is a Linear Program (LP).
Variables: h[0..99], t.
Objective: min t.
Constraints:
1. sum(h) = 50
2. 0 <= h[i] <= 1
3. C_k(h) <= t for all k in [-99, 99].
   C_k(h) = sum_i h[i] * (1-h[i+k]) ... wait, correlate(h, 1-h).
   Let's check the definition of np.correlate(h, 1-h, mode="full").
   The k-th coefficient is sum_{i} h[i] * (1-h[i-k]) ? Or h[i+k]?
   Standard definition: (f * g)[k] = sum_n f[n] g[n-k].
   Here f=h, g=1-h.
   So C_k = sum_i h[i] * (1 - h[i-k]).
   Note: h[i] and h[i-k] are both variables. So C_k is quadratic?
   No, wait.
   np.correlate(h, 1-h) computes sum h[i] * (1-h)[i-k] = sum h[i] - sum h[i]h[i-k].
   This is quadratic because of the h[i]h[i-k] term.
   
   Let's re-read the task carefully.
   "value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2"
   
   If h[i]h[i-k] is present, the problem is NON-CONVEX.
   Inspiration 1 claimed: "Formulate the problem as a Convex Quadratic Program (QP) by minimizing the auxiliary variable t subject to the linearized correlation constraints C_k(h) <= t (using the valid relaxation...)"
   
   Is there a relaxation?
   h[i]h[j] <= (h[i]+h[j])/2 is a valid inequality (AM-GM).
   So -h[i]h[j] >= -(h[i]+h[j])/2.
   Therefore, C_k(h) = sum h[i] - sum h[i]h[i-k] >= sum h[i] - sum (h[i]+h[i-k])/2.
   Let L_k(h) = sum h[i] - 0.5 * sum (h[i] + h[i-k]).
   Then C_k(h) >= L_k(h).
   If we minimize max L_k(h), we get a LOWER bound on the true max C_k(h).
   But we want to MINIMIZE the true max C_k(h).
   Minimizing a lower bound doesn't guarantee minimizing the true value.
   However, if the optimal h for the relaxed problem satisfies h[i]h[i-k] = (h[i]+h[i-k])/2 (i.e., h[i]=h[i-k]), then it is exact.
   
   Actually, let's look at the structure.
   If we assume the relaxation is tight or close, solving the LP for the lower bound might guide us to a good solution.
   Alternatively, we can use a direct QP solver if we can express it as minimizing sum of squares or similar.
   
   Let's look at Inspiration 1's score: 1.3114.
   The current score is 1.1538.
   The LP relaxation approach (minimizing the lower bound) is what Inspiration 1 likely did.
   
   Let's implement the LP relaxation.
   Minimize t
   Subject to:
   L_k(h) <= t for all k
   sum(h) = 50
   0 <= h <= 1
   
   L_k(h) = sum_i h[i] - 0.5 * sum_i (h[i] + h[i-k])
   Note: sum_i h[i] is constant (50) for a given k? No, sum_i h[i] is always 50.
   So L_k(h) = 50 - 0.5 * (sum_i h[i] + sum_i h[i-k]).
   sum_i h[i] = 50.
   sum_i h[i-k] is the sum of h with indices shifted. Due to boundary effects in "full" mode, the number of terms varies.
   
   Let's compute L_k(h) explicitly for the LP.
   For a given lag k:
   The correlation sum is over valid i.
   If k >= 0: i ranges from k to N-1. (N-k terms)
   If k < 0: i ranges from 0 to N-1+k. (N+k terms)
   
   L_k(h) = sum_{i in valid} h[i] - 0.5 * sum_{i in valid} (h[i] + h[i-k])
   Let S1 = sum_{i in valid} h[i]
   Let S2 = sum_{i in valid} h[i-k]
   L_k(h) = S1 - 0.5*(S1 + S2) = 0.5*S1 - 0.5*S2.
   
   This is linear in h.
   
   We can solve this LP using scipy.optimize.linprog.
   Variables: h[0..N-1], t.
   Objective: c = [0, ..., 0, 1] (minimize t)
   Constraints:
   1. Equality: sum(h) = 50.
   2. Inequality: L_k(h) - t <= 0 for all k.
   3. Bounds: 0 <= h[i] <= 1.
   
   After finding the optimal h_relaxed, we compute the true value.
   Then, we can use this h_relaxed as a starting point for a local refinement (like the current program does) to see if we can improve further, or just return it if it's good.
   
   Actually, the relaxation h[i]h[j] <= (h[i]+h[j])/2 is tight if h[i]=h[j].
   If the optimal solution of the LP has many h[i]=h[i-k], it might be a very good candidate.
   
   Let's implement the LP solver.
"""

import time
import numpy as np
from scipy.optimize import linprog

N = 100
TIME_LIMIT = 120  # seconds, safety margin before 150s kill

def _value(h):
    """Compute the objective: max cross-correlation of h and 1-h, normalized."""
    return np.max(np.correlate(h, 1 - h, mode="full")) / N * 2

def _solve_lp():
    """
    Solve the LP relaxation:
    Minimize t
    Subject to:
    L_k(h) <= t for all k in [-N+1, N-1]
    sum(h) = N/2
    0 <= h <= 1
    
    L_k(h) = 0.5 * sum_{i valid} (h[i] - h[i-k])
    """
    # Variables: h[0..N-1], t
    # c = [0, ..., 0, 1]
    c = np.zeros(N + 1)
    c[N] = 1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # L_k(h) - t <= 0
    # 0.5 * sum_{i valid} h[i] - 0.5 * sum_{i valid} h[i-k] - t <= 0
    
    A_rows = []
    b_vals = []
    
    for k in range(-N + 1, N):
        # Determine valid indices i such that 0 <= i < N and 0 <= i-k < N
        # i >= 0, i < N
        # i >= k, i < N+k
        
        i_start = max(0, k)
        i_end = min(N, N + k)
        
        if i_start >= i_end:
            continue
            
        row = np.zeros(N + 1)
        row[N] = -1.0  # Coefficient for t
        
        # Sum over valid i
        for i in range(i_start, i_end):
            # Term h[i] has coeff 0.5
            row[i] += 0.5
            # Term h[i-k] has coeff -0.5
            idx2 = i - k
            row[idx2] -= 0.5
            
        A_rows.append(row)
        b_vals.append(0.0)
        
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
    else:
        A_ub = np.empty((0, N + 1))
        b_ub = np.empty(0)
        
    # Equality constraint: sum(h) = N/2
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([N / 2.0])
    
    # Bounds
    bounds = [(0, 1)] * N + [(0, None)]  # h[i] in [0,1], t >= 0
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        h_opt = res.x[:N]
        # Clip to ensure numerical precision
        h_opt = np.clip(h_opt, 0, 1)
        # Re-project to satisfy sum exactly if needed
        diff = N / 2.0 - np.sum(h_opt)
        if abs(diff) > 1e-9:
            # Simple projection: distribute diff
            # To keep it simple and fast, we can just adjust slightly
            # But linprog should be accurate.
            pass
        return h_opt
    else:
        return None

def construct():
    """Construct 100 step heights minimizing the overlap value."""
    t0 = time.time()
    
    # 1. Solve LP relaxation
    h_lp = _solve_lp()
    
    if h_lp is not None:
        h_best = h_lp
        # Ensure sum is exactly N/2 for validity check
        s = np.sum(h_best)
        if abs(s - N/2) > 1e-9:
            # Adjust
            h_best = h_best + (N/2 - s) / N
            h_best = np.clip(h_best, 0, 1)
            
        val_best = _value(h_best)
    else:
        # Fallback to seed if LP fails
        h_best = np.full(N, 0.5)
        val_best = _value(h_best)
        
    # 2. Local refinement from the LP solution
    # The LP solution is a lower bound minimizer. The true objective is >= LP objective.
    # We can try to reduce the true objective further by local search.
    
    h_current = h_best.copy()
    val_current = val_best
    
    # Simple local search: try small perturbations
    # Use a deterministic or random walk with acceptance if better
    np.random.seed(42)
    
    steps = 0
    max_steps = 5000
    step_size = 0.05
    
    while steps < max_steps:
        if time.time() - t0 > TIME_LIMIT - 10:
            break
            
        # Pick a random index
        idx = np.random.randint(N)
        # Pick a random delta
        delta = np.random.choice([-1, 1]) * step_size
        
        h_trial = h_current.copy()
        h_trial[idx] += delta
        
        # Project to feasible set
        # 1. Clip
        h_trial = np.clip(h_trial, 0, 1)
        # 2. Adjust sum
        s = np.sum(h_trial)
        target = N / 2.0
        diff = target - s
        if abs(diff) > 1e-9:
            # Distribute diff to elements not at bounds
            mask = (h_trial > 1e-6) & (h_trial < 1 - 1e-6)
            if np.any(mask):
                h_trial[mask] += diff / np.sum(mask)
                h_trial = np.clip(h_trial, 0, 1)
            else:
                # If all at bounds, we can't adjust sum easily without breaking bounds.
                # In this case, discard trial or try a different move.
                # For simplicity, if we can't project, skip.
                continue
                
        val_trial = _value(h_trial)
        
        if val_trial < val_current:
            h_current = h_trial
            val_current = val_trial
            # Decay step size occasionally
            if steps % 500 == 0:
                step_size *= 0.9
        else:
            # No improvement, keep step size or decay slightly to ensure convergence
            if steps % 100 == 0:
                step_size *= 0.99
                
        steps += 1
        
    return h_current
