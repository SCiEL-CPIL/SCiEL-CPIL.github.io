# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
    1. Select a domain D = {1, ..., K}.
    2. Formulate an LP to maximize a = -sum f(k) log(k) / k
       subject to:
       - M(x) = sum_{k in D} f(k) floor(x/k) <= 1 for all x in [1, 10K]
       - sum_{k in D} f(k) / k = 0 (Note: The evaluator enforces this by adjusting f(1), 
         but we can include it as a constraint to help the LP or just let the evaluator do it.
         Actually, the evaluator replaces f(1) with f(1) - S. 
         The objective a uses the *original* f(k) values? 
         "The metric is a = -sum_k f(k) * log(k) / k". 
         Usually, metrics are computed on the returned values. 
         Let's re-read carefully: 
         "values are clipped... f(1) is replaced... M(x) is computed... if infeasible score 0. 
         Otherwise the metric is a = -sum_k f(k) * log(k) / k".
         It is ambiguous if 'f(k)' in the metric refers to the pre-adjustment or post-adjustment values.
         However, usually in these tasks, the returned values are the ones scored.
         But the constraint M(x) <= 1 is checked on the *post-adjustment* values?
         "Then f(1) is replaced... so that afterwards sum_k f(k)/k = 0. 
         M(x) = ... is computed... if any M(x) > 1.0001 ... infeasible".
         This implies M(x) is checked on the *modified* values.
         And the metric a is likely computed on the *modified* values too, or the original?
         "The metric is a = -sum_k f(k) * log(k) / k".
         If it's the modified values, then sum f(k)/k = 0.
         Let's assume the metric is computed on the values *after* the evaluator's modification.
         Why? Because the evaluator describes the process: Clip -> Modify -> Check Feasibility -> Compute Metric.
         If the metric was on the original values, the modification of f(1) wouldn't affect the score directly, 
         except through feasibility. But f(1) has log(1)/1 = 0, so f(1) doesn't contribute to 'a' anyway!
         So it doesn't matter which f(1) is used for 'a'.
         However, the *other* f(k) are unchanged.
         So we want to maximize -sum_{k>1} f(k) log(k) / k.
         Subject to:
         Let g(k) be the values we return.
         Let S = sum_{k} g(k)/k.
         The evaluator sets g'(1) = g(1) - S, and g'(k) = g(k) for k>1.
         Then it checks M'(x) = sum g'(k) floor(x/k) <= 1.
         M'(x) = g'(1) floor(x/1) + sum_{k>1} g(k) floor(x/k)
              = (g(1) - S) * x + sum_{k>1} g(k) floor(x/k)
              = g(1)x - x sum_{k} g(k)/k + sum_{k>1} g(k) floor(x/k)
              = g(1)x - x(g(1)/1 + sum_{k>1} g(k)/k) + sum_{k>1} g(k) floor(x/k)
              = -x sum_{k>1} g(k)/k + sum_{k>1} g(k) floor(x/k)
              = sum_{k>1} g(k) (floor(x/k) - x/k)
        
        This is a very specific constraint structure.
        Let h(k) = g(k) for k > 1.
        Constraint: sum_{k=2}^K h(k) (floor(x/k) - x/k) <= 1 for all x in [1, 10K].
        
        Objective: Maximize a = -sum_{k=2}^K h(k) log(k) / k.
        
        This is an LP in variables h(2)...h(K).
        g(1) is free (or can be 0) and does not affect feasibility or objective.
        So we can set g(1)=0 and solve for h(k).
    """
    start_time = time.time()
    deadline = 600.0
    
    # We need to choose K.
    # The LP size is K-1 variables and 10K constraints.
    # For K=1000, 10000 constraints. Doable.
    # For K=10000, 100000 constraints. Might be slow.
    
    # Let's try a moderate K first, maybe 2000 or 5000.
    # The Möbius function suggests K should be large to approach 1.0.
    # But LP might allow better packing.
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try a few K values
    # K=100 is safe but likely low score.
    # K=1000 is a good start.
    K_candidates = [100, 500, 1000, 2000]
    
    for K in K_candidates:
        if time.time() > deadline - 60:
            break
            
        # Variables: h(2), ..., h(K)
        n_vars = K - 1
        if n_vars == 0:
            continue
            
        # Objective: Maximize -sum h(k) log(k) / k
        # linprog minimizes c^T x. So c_k = log(k) / k
        c = np.zeros(n_vars)
        ks = np.arange(2, K + 1)
        c[:] = np.log(ks) / ks
        
        # Constraints:
        # For each x in [1, 10K]:
        # sum_{k=2}^K h(k) (floor(x/k) - x/k) <= 1
        
        max_x = 10 * K
        # We can reduce the number of constraints by noting that the function 
        # phi_x(k) = floor(x/k) - x/k is periodic or has structure?
        # Actually, for a fixed x, floor(x/k) changes only when x/k crosses an integer.
        # But we need to satisfy it for ALL x.
        # We can just generate all 10K constraints.
        
        A_ub = np.zeros((max_x, n_vars))
        b_ub = np.ones(max_x)
        
        # Fill A_ub
        # A_ub[i, j] corresponds to x = i+1, k = j+2
        # We can vectorize this partially or use loops if K is small.
        # For K=1000, max_x=10000, n_vars=999. 10M operations. Python loops might be slow.
        # Let's use numpy broadcasting if possible.
        
        x_vals = np.arange(1, max_x + 1)[:, np.newaxis] # shape (max_x, 1)
        k_vals = ks[np.newaxis, :] # shape (1, n_vars)
        
        # floor(x/k)
        # Note: x and k are integers.
        # We need to be careful with integer division vs float.
        # floor(x/k) is integer division.
        # In numpy, // does floor division for positive integers.
        
        # This creates a max_x x n_vars array.
        # For K=2000, 20000 x 1999 ~ 40M elements. 320MB float64. OK.
        
        try:
            floors = x_vals // k_vals
            terms = floors - x_vals / k_vals
            A_ub = terms.T # We need rows for x, cols for k. terms is (max_x, n_vars).
            # Wait, linprog expects A_ub shape (m, n) where m is constraints, n is vars.
            # So A_ub = terms.
        except MemoryError:
            # Fallback to smaller K or loop
            continue
            
        # Bounds: h(k) can be any real?
        # The problem says "finite reals". Clipped to [-10, 10].
        # So we should bound h(k) to [-10, 10].
        bounds = [(-10, 10)] * n_vars
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0:
                h_sol = res.x
                # Construct full domain and values
                # g(1) = 0
                # g(k) = h(k-2) for k=2..K
                domain = np.arange(1, K + 1)
                values = np.zeros(K + 1)
                values[1] = 0.0
                values[2:] = h_sol
                
                # Verify feasibility with the actual evaluator logic to be safe
                # S = sum values/k
                S = np.sum(values / domain)
                mod_values = values.copy()
                mod_values[0] = mod_values[0] - S # index 0 is k=1
                
                # Check M(x)
                # We can do a quick check on a subset or all if fast enough
                # For K=1000, 10000 x's. 10M ops.
                # Let's just trust the LP for now, but maybe do a spot check.
                
                # Calculate objective
                # a = -sum f(k) log(k) / k
                # Using mod_values or values? 
                # f(1) term is 0. So it's the same.
                logs = np.log(domain.astype(np.float64))
                a_val = -np.sum(mod_values * logs / domain)
                
                if a_val > best_score:
                    best_score = a_val
                    best_domain = domain.copy()
                    best_values = mod_values.copy()
                    
        except Exception:
            pass
            
    if best_domain is None:
        # Fallback to Möbius for small K if LP failed
        K = 100
        mu = _compute_mobius(K)
        mask = mu != 0
        domain = np.where(mask)[0] + 1
        values = mu[mask].astype(np.float64)
        # Apply evaluator logic to get final values
        S = np.sum(values / domain)
        # find index of 1
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            values[idx_1[0]] -= S
        else:
            # If 1 not in domain, evaluator sets f(1)=0-S = -S? 
            # "f(1) = 0 before this if 1 is not in the domain"
            # So we need to add 1 to domain? 
            # The task says "returning (domain, values)". 
            # If 1 is not in domain, the evaluator adds it? 
            # "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)"
            # This implies the evaluator *adds* 1 to the domain if it's missing? 
            # Or does it just treat the sum as if f(1) existed?
            # "M(x) = sum_k f(k) ...". If 1 is not in domain, the sum doesn't include it.
            # But the text says "f(1) is replaced...". This strongly implies f(1) is part of the function.
            # To be safe, always include 1 in the domain.
            pass
            
    return best_domain, best_values

# EVOLVE-BLOCK-END
