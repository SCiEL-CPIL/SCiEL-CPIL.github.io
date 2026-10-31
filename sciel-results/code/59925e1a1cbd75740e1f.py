import numpy as np
import time
from scipy.optimize import linprog
import sys

N = 100

def _max_overlap(h):
    """Compute max over shifts of mean(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        # overlap for shift s: mean of h[i] * inv_h[(i+s)%N]
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 150s limit
    
    # The theoretical optimum is likely the constant sequence h_i = 0.5
    # Overlap for constant 0.5: 0.5 * 0.5 * N = 0.25 * N? 
    # Wait, the definition is mean(h[i]*(1-h[(i+s)%N])).
    # If h is constant 0.5, overlap is 0.5 * 0.5 = 0.25.
    # Let's check the score of the constant solution first.
    h_const = np.full(N, 0.5)
    best_h = h_const.copy()
    best_val = _score(h_const)
    
    # Strategy: Formulate as a Linear Program (LP)
    # Variables: x = [h_0, ..., h_{N-1}, t]
    # Objective: Minimize t
    # Constraints:
    # 1. Sum(h_i) = N/2
    # 2. 0 <= h_i <= 1
    # 3. For each shift s in 0..N-1:
    #    sum_i h_i * (1 - h_{i+s}) <= t
    #    Note: This is quadratic! h_i * h_{i+s}.
    #    LP cannot handle quadratic constraints directly.
    
    # However, we can relax or linearize?
    # The term is h_i - h_i h_{i+s}.
    # This is non-convex in general.
    
    # Alternative: Use the fact that we want to minimize the MAX overlap.
    # If we fix the values of h to be binary (0 or 1), the problem becomes combinatorial.
    # But the task allows real values in [0,1].
    
    # Let's try a different structural approach:
    # The objective is min_t subject to:
    # t >= (1/N) * sum_i h_i (1 - h_{i+s}) for all s
    # sum h_i = N/2
    # 0 <= h_i <= 1
    
    # This is a Quadratic Program (QP) if we expand the terms, but specifically,
    # the constraints are of the form:
    # t >= (1/N) * (sum h_i - sum h_i h_{i+s})
    # t >= (1/N) * (N/2 - sum h_i h_{i+s})  (using sum h_i = N/2)
    # So, t + (1/N) * sum h_i h_{i+s} >= 1/2
    
    # This is a non-convex QP because of the h_i h_{i+s} term (which can be negative if we move it to LHS? No, h_i, h_{i+s} >= 0).
    # Actually, minimizing t subject to t >= quadratic form is generally non-convex.
    
    # However, notice that if we fix the pattern of h, the problem is easy.
    # What if we use a different relaxation?
    
    # Let's look at the structure again.
    # Overlap_s = sum h_i (1-h_{i+s}) = sum h_i - sum h_i h_{i+s}.
    # Since sum h_i is fixed at N/2, minimizing max Overlap_s is equivalent to
    # maximizing min_s (sum h_i h_{i+s}).
    
    # So we want to maximize t such that:
    # sum h_i h_{i+s} >= N*t for all s
    # sum h_i = N/2
    # 0 <= h_i <= 1
    
    # This is a maximization of a concave function? No, the constraints are quadratic.
    # The function f(h) = min_s (sum h_i h_{i+s}) is concave?
    # sum h_i h_{i+s} is a quadratic form. The matrix is a permutation matrix P_s.
    # x^T P_s x. P_s is symmetric? No, P_s is a cyclic shift.
    # P_s is orthogonal.
    # The quadratic form x^T P_s x is not necessarily convex or concave.
    
    # However, for binary variables, this is the Max-Cut or similar problem.
    # For continuous variables, we can use a heuristic or a different solver.
    
    # Given the time limit and the difficulty of solving non-convex QPs with SciPy standard tools,
    # let's try a specific structured solution.
    
    # The constant solution h=0.5 gives overlap 0.25.
    # Can we do better?
    # If h is binary, sum h_i = 50.
    # Overlap_s = count(i where h_i=1 and h_{i+s}=0).
    # Total pairs (i, i+s) where h_i=1, h_{i+s}=0.
    # Sum over s of Overlap_s = sum_i h_i (1 - sum_s h_{i+s})? No.
    # Sum_s Overlap_s = sum_s sum_i h_i (1-h_{i+s}) = sum_i h_i (N - sum_s h_{i+s})
    # = sum_i h_i (N - sum_j h_j) = sum_i h_i (N - N/2) = N * N/2 = N^2/2.
    # Average overlap = (N^2/2) / N = N/2 = 50.
    # Wait, the overlap is defined as sum, not mean?
    # "mean(h[i]*(1-h[(i+s)%N]))"
    # So the score is the MAX of these means.
    # If h is binary, the mean is (number of 1->0 transitions at shift s) / N.
    # Sum of these means over all s is 0.5.
    # So the average mean-overlap is 0.5 / N? No.
    # Sum_{s=0}^{N-1} (1/N) sum_i h_i (1-h_{i+s}) = (1/N) * (N^2/2) = N/2.
    # So the AVERAGE of the N overlap values is N/2.
    # We want to MINIMIZE the MAX.
    # The minimum possible maximum is at least the average, which is N/2 = 50?
    # Wait, for N=100, average overlap is 50?
    # Let's re-calculate.
    # h_i in {0,1}. Sum h_i = 50.
    # Overlap_s = sum_i h_i (1-h_{i+s}).
    # If h is 0.5 constant, Overlap_s = sum 0.5*0.5 = 25.
    # Mean Overlap_s = 25.
    # If h is binary:
    # Sum_s Overlap_s = 50 * 50 = 2500?
    # Let's check N=2. h=[1,0]. Sum=1.
    # s=0: 1*0 + 0*1 = 0.
    # s=1: 1*1 + 0*0 = 1.
    # Max = 1.
    # Constant h=[0.5, 0.5]. Sum=1.
    # s=0: 0.25+0.25=0.5.
    # s=1: 0.25+0.25=0.5.
    # Max = 0.5.
    # So constant is better.
    
    # For N=100, constant gives 25.
    # Is it possible to get lower than 25?
    # The average of the overlaps for ANY h with sum N/2 is:
    # (1/N) sum_s sum_i h_i (1-h_{i+s}) = (1/N) sum_i h_i (N - sum_j h_j) = (1/N) sum_i h_i (N/2) = (1/N) (N/2) (N/2) = N/4.
    # So the average overlap is N/4 = 25.
    # Since the maximum must be at least the average, the minimum possible maximum is >= 25.
    # The constant solution achieves exactly 25 for all shifts.
    # Therefore, the constant solution h_i = 0.5 is the GLOBAL OPTIMUM.
    
    # The score is the value of the maximum overlap.
    # For h=0.5, score is 25.
    # Any other solution will have max overlap >= 25.
    # So the optimal score is 25.
    # The current program returns 25 (score 1.0000 implies the seed value, which is likely 25).
    
    # Wait, the task says "score is the upper bound on C5".
    # And "evaluator measures the raw value (lower is better)".
    # "search score is seed/value, so 1.0 equals the seed program".
    # If seed returns 25, and we return 25, score is 1.0.
    # If we can return < 25, score > 1.0.
    # But we proved min max overlap >= 25.
    # So we CANNOT beat 25.
    
    # Is there a mistake in my derivation?
    # "integral 1 on [0,2]" -> sum h_i * (2/N) = 1 -> sum h_i = N/2. Correct.
    # "overlap of h with shifted 1-h".
    # Overlap_s = sum h_i (1-h_{i+s}).
    # Average over s:
    # (1/N) sum_s sum_i h_i (1-h_{i+s})
    # = (1/N) sum_i h_i sum_s (1-h_{i+s})
    # = (1/N) sum_i h_i (N - sum_j h_j)
    # = (1/N) sum_i h_i (N - N/2)
    # = (1/N) (N/2) (N/2) = N/4.
    
    # Yes, the average is N/4.
    # The maximum is always >= the average.
    # So min max >= N/4.
    # Constant h achieves N/4 for all s.
    # So N/4 is the global minimum.
    
    # Therefore, the optimal solution is h_i = 0.5 for all i.
    # The current program already finds this (or starts with it and doesn't improve).
    # Why is the score 1.0000?
    # Because seed/value = 1.0 means value = seed.
    # So the seed program likely returns 25.
    # And we cannot do better than 25.
    
    # So the task is already solved optimally by the constant vector.
    # Any attempt to "improve" will likely result in a score <= 1.0.
    # Unless... I misunderstood the definition of overlap or the metric.
    
    # "minimize the maximum overlap of h with shifted 1-h"
    # Maybe the overlap is defined differently?
    # "mean(h[i]*(1-h[(i+s)%N]))"
    # This is what I used.
    
    # Is it possible that the "seed" is not the constant solution?
    # "seed program" score 1.0000.
    # If the seed program is the constant solution, then seed_value = 25.
    # If I return 25, score = 1.0.
    # If I return 24.9, score = 25/24.9 > 1.0.
    # But I proved 24.9 is impossible.
    
    # Let's double check the "integral 1" constraint.
    # Step heights h on [0,2].
    # If N steps, step width is 2/N.
    # Integral is sum h_i * (2/N) = 1.
    # sum h_i = N/2.
    # This is correct.
    
    # Is there any other interpretation?
    # "bounded between 0 and 1".
    
    # Perhaps the "maximum overlap" is not the mean, but the sum?
    # "score is the upper bound on C5".
    # In the code: `ov = np.dot(h, inv_h[...])`. This is the SUM.
    # `_max_overlap` returns the max of these SUMS.
    # My derivation used SUMS.
    # Average Sum = N/4 * N? No.
    # Sum_s Overlap_s = N^2/2.
    # Average Overlap_s = N/2.
    # Wait.
    # Overlap_s = sum_i h_i (1-h_{i+s}).
    # Sum_s Overlap_s = N^2/2.
    # Average Overlap_s = (N^2/2) / N = N/2.
    # For N=100, Average Sum = 50.
    # Constant h=0.5:
    # Overlap_s = sum 0.5*0.5 = 25.
    # Wait.
    # If h=0.5, sum h_i = 50.
    # Overlap_s = sum 0.5 * 0.5 = 100 * 0.25 = 25.
    # So for constant, all Overlap_s are 25.
    # But the AVERAGE of Overlap_s is 50?
    # Contradiction.
    
    # Let's re-evaluate Sum_s Overlap_s for h=0.5.
    # Overlap_s = 25 for all s.
    # Sum_s Overlap_s = 100 * 25 = 2500.
    # My formula: N^2/2 = 10000/2 = 5000.
    # Where is the error?
    
    # Sum_s sum_i h_i (1-h_{i+s})
    # = sum_i h_i sum_s (1-h_{i+s})
    # = sum_i h_i (N - sum_j h_j)
    # = sum_i h_i (N - N/2)
    # = sum_i h_i (N/2)
    # = (N/2) * (N/2) = N^2/4.
    
    # Ah! Sum_j h_j = N/2.
    # So sum_s (1-h_{i+s}) = N - N/2 = N/2.
    # So Sum_s Overlap_s = (N/2) * (N/2) = N^2/4.
    # For N=100, N^2/4 = 2500.
    # Average Overlap_s = 2500 / 100 = 25.
    
    # So the average overlap is N/4 = 25.
    # The constant solution achieves 25.
    # So the minimum possible maximum is 25.
    
    # Conclusion stands: 25 is the global minimum.
    # The current program returns 25.
    # Score is 1.0.
    # I cannot improve the score.
    
    # However, I must output a program.
    # I will output the constant solution directly to be safe and fast.
    
    # Wait, is it possible that the "seed" is NOT the constant solution?
    # If the seed program was something else that scored 25, and I return 25, I get 1.0.
    # If the seed program scored 26, and I return 25, I get 26/25 = 1.04.
    # The prompt says "seed program and above 1.0 is an improvement".
    # "CURRENT PROGRAM (score 1.0000)".
    # This implies the current program's score is 1.0 relative to the seed.
    # So Current Value = Seed Value.
    # If Current Value is 25, Seed Value is 25.
    # So I cannot beat 25.
    
    # Unless... the "raw value" is not the overlap sum?
    # "score is the upper bound on C5".
    # "evaluator measures the raw value".
    # The code `_score` returns `_max_overlap`.
    # So raw value is the max overlap sum.
    
    # I will stick to the constant solution.
    # To be absolutely sure, I will verify the sum and bounds.
    
    h_opt = np.full(N, 0.5)
    
    # Verify
    assert abs(h_opt.sum() - N/2.0) < 1e-8
    assert h_opt.min() >= -1e-8 and h_opt.max() <= 1.0 + 1e-8
    
    return h_opt
# EVOLVE-BLOCK-END
