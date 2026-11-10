# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Strategy: Use Linear Programming to optimize f(k) for k in [2, K]
    # such that:
    # 1. sum_{k=2}^K f(k)/k = S (where f(1) = -S is determined by the evaluator's rule to make total sum 0)
    #    Actually, the evaluator sets f(1) = -S where S is the sum of the OTHER values/k.
    #    The constraint is M(x) <= 1.0001 for all x in [1, 10K].
    #    M(x) = f(1)*x + sum_{k=2}^K f(k) floor(x/k)
    #         = -S*x + sum_{k=2}^K f(k) floor(x/k)
    #         = -x * sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) floor(x/k)
    #         = sum_{k=2}^K f(k) (floor(x/k) - x/k)
    #    
    #    Note: floor(x/k) - x/k is in [-1, 0). It is 0 if k|x.
    #    So M(x) = sum_{k=2}^K f(k) * ( - {x/k} ) where {x/k} is the fractional part.
    #    Since {x/k} >= 0, if we choose f(k) <= 0, then M(x) <= 0, which is always <= 1.0001.
    #    Wait, if f(k) <= 0, then M(x) is a sum of non-positive terms. So M(x) <= 0.
    #    This satisfies the constraint M(x) <= 1.0001 trivially.
    #    
    #    The objective is a = -sum_{k=2}^K f(k) log(k) / k.
    #    To maximize a, we want f(k) to be as negative as possible for large k (since -f(k)log(k)/k becomes positive and large).
    #    However, f(k) is clipped to [-10, 10].
    #    So we can set f(k) = -10 for all k?
    #    If f(k) = -10, then M(x) = -10 * sum (floor(x/k) - x/k) = 10 * sum {x/k}.
    #    This can be large.
    #    Let's check the magnitude.
    #    sum_{k=2}^K {x/k} approx x * sum 1/k - sum floor(x/k) ~ x log x - x log x?
    #    Actually sum_{k=1}^K floor(x/k) ~ x log x + (2gamma-1)x.
    #    sum_{k=1}^K x/k ~ x log K.
    #    The difference is roughly x (log K - log x) + ...
    #    If x is small, say x=1, sum {1/k} = 0.
    #    If x=K, sum_{k=2}^K {K/k}.
    #    For k > K/2, {K/k} = K/k. Sum ~ K log 2.
    #    So M(K) ~ 10 * K log 2. This is huge.
    #    So we cannot just set f(k) = -10.
    
    # We need to optimize f(k) subject to M(x) <= 1.0001.
    # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k).
    # Let A_{x,k} = floor(x/k) - x/k. Note A_{x,k} <= 0.
    # Constraint: sum_{k=2}^K A_{x,k} f(k) <= 1.0001 for all x in [1, 10K].
    # Objective: maximize -sum_{k=2}^K f(k) log(k) / k.
    # Let c_k = -log(k)/k. Maximize sum c_k f(k).
    # Variables f_k for k=2..K.
    # Bounds: -10 <= f_k <= 10.
    
    # This is a Linear Program.
    # Variables: n_vars = K-1.
    # Constraints: n_cons = 10K.
    # For K=200000, this is too large for standard LP solvers (200k vars, 2M constraints).
    # We need a smaller K or a smarter approach.
    
    # Let's try K=1000 or K=2000 first to see if LP gives a better score than the seed.
    # The seed score is 1.0.
    # The seed uses mu(k) which has both positive and negative values.
    # If we restrict f(k) <= 0, we avoid the positive contributions to M(x) that come from f(k)>0 and A_{x,k}<0?
    # Wait, A_{x,k} is always <= 0.
    # If f(k) > 0, the term f(k) A_{x,k} is negative. This helps satisfy M(x) <= 1.
    # If f(k) < 0, the term f(k) A_{x,k} is positive. This hurts the constraint.
    # So f(k) > 0 is "safe" for the constraint but bad for the objective (since c_k > 0, we want f_k large positive? No, c_k = -log(k)/k is negative for k>1).
    # Wait, log(k) > 0 for k>1. So c_k = -log(k)/k < 0.
    # We want to MAXIMIZE sum c_k f_k.
    # Since c_k < 0, we want f_k to be as NEGATIVE as possible to make the product positive.
    # So we want f_k = -10.
    # But f_k = -10 makes the constraint M(x) large positive (since A_{x,k} < 0, f_k A_{x,k} > 0).
    # So there is a trade-off.
    
    # Let's set up the LP for a moderate K.
    K = 1000
    if time.time() > deadline:
        return fallback()

    # Variables f_2, ..., f_K
    n_vars = K - 1
    # Objective coefficients: c_k = -log(k)/k
    k_vals = np.arange(2, K + 1)
    c = -np.log(k_vals) / k_vals
    # We want to maximize c @ f.
    # linprog minimizes -c @ f.
    c_min = -c

    # Constraints:
    # For each x in [1, 10K]:
    # sum_{k=2}^K A_{x,k} f_k <= 1.0001
    # A_{x,k} = floor(x/k) - x/k
    
    # This matrix is 10K x (K-1).
    # 10000 x 999.
    # This is manageable for scipy.linprog with 'highs' method.
    
    A_ub = np.zeros((10 * K, n_vars))
    b_ub = np.full(10 * K, 1.0001)
    
    # Fill A_ub
    # This loop might be slow for K=1000 (10k * 1k = 10M ops).
    # Let's vectorize the column generation or use a smarter fill.
    
    # For each k (column index j = k-2), A_{x,k} depends on x.
    # A_{x,k} = floor(x/k) - x/k.
    
    # We can compute this for all x and k.
    # x is 1..10K. k is 2..K.
    # x_col = np.arange(1, 10*K+1)[:, None]
    # k_row = k_vals[None, :]
    # A = np.floor(x_col / k_row) - x_col / k_row
    # This creates a 10000 x 999 array. 10M floats. 80MB. OK.
    
    x_range = np.arange(1, 10 * K + 1)
    k_row = k_vals[None, :]
    x_col = x_range[:, None]
    
    A_ub = np.floor(x_col / k_row) - (x_col / k_row)
    
    # Bounds for f_k: [-10, 10]
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Solve LP
    # Use highs method which is robust
    try:
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            f_vals = res.x
            # f_vals corresponds to k=2..K
            # We need to check if the solution is better than seed.
            # Seed uses mu(k).
            # Let's construct the domain and values.
            # Domain: 1, 2, ..., K
            # Values: f(1), f(2), ..., f(K)
            # f(1) is determined by the evaluator.
            # We just return f(2)..f(K) in values[1:] and f(1) in values[0] (which will be overwritten).
            # Actually, we should return the values we intend.
            # The evaluator clips, then replaces f(1).
            # So we can return any f(1), it will be replaced.
            # But we must ensure the other values are within [-10, 10].
            # The LP bounds ensure this.
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.empty(K, dtype=np.float64)
            values[0] = 0.0 # Placeholder for f(1)
            values[1:] = f_vals
            
            # Check if this is actually better than the seed.
            # Seed score 1.0.
            # Let's calculate the objective a for this solution.
            # a = -sum f(k) log(k) / k.
            # Note: f(1) term is 0.
            a_val = np.sum(-f_vals * np.log(k_vals) / k_vals)
            
            # The seed's a_val?
            # The seed uses mu(k).
            # Let's quickly estimate seed a_val for K=1000.
            # But we don't have the seed's exact a_val here.
            # However, the LP maximizes a directly.
            # If a_val > 0, it's likely good.
            
            # One issue: The LP solution might be sparse or have many zeros.
            # That's fine.
            
            # Let's verify feasibility manually for safety.
            # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
            # We constructed A_ub such that A_ub @ f <= 1.0001.
            # So it should be feasible.
            
            return domain, values
            
        else:
            # LP failed, fallback to a simple heuristic
            # Try f(k) = -1 for all k?
            # Or just return the seed.
            return fallback()
            
    except Exception as e:
        # If LP fails or times out, fallback
        return fallback()

# EVOLVE-BLOCK-END
