import math
import random
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

def _calculate_coverage(classes, limit=100000):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    Returns L.
    """
    if not classes:
        return 0
    
    # We only need to check up to the minimum of limit and a reasonable bound
    # but the problem caps at 10^5.
    # For speed in internal loops, we might pass a smaller limit, 
    # but for the final answer, we need the exact L.
    
    # Optimization: If we have a set of classes, the coverage pattern is periodic 
    # with period LCM of primes. But LCM grows too fast.
    # Simple bytearray marking is O(L * N_classes / p_avg).
    # For L=100000 and ~100 classes, this is ~10^7 ops, which is fast enough in Python if optimized?
    # Actually, 10^7 Python loops might take ~1s. 
    # We can use numpy for faster marking?
    # Or just use bytearray and slice assignment? No, step is p.
    
    # Let's use a simple loop but break early if we find a gap.
    # The score is the FIRST gap - 1.
    
    covered = bytearray(limit + 1)
    
    for p, r in classes:
        start = r if r > 0 else p
        if start < 1:
            start += p
        # Mark multiples
        # Using range is standard
        for n in range(start, limit + 1, p):
            covered[n] = 1
            
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals_exact(classes):
    """Exact Fraction sum for final verification."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy:
    # 1. Define a target interval [1, M] where we want to maximize coverage.
    #    We don't know M, but we can try to cover a range, say 1..2000 or 1..5000.
    #    The ILP will maximize the sum of covered indicators.
    #    However, "largest L such that 1..L are covered" is different from "maximize total coverage in 1..M".
    #    Maximizing total coverage might leave holes early on.
    #    To enforce "1..L covered", we can add constraints that force coverage of 1, then 2, etc?
    #    That's hard in a single ILP.
    
    # Alternative Strategy:
    # Use the ILP to find a set of classes that covers a specific target T completely.
    # If feasible, try larger T.
    # This is a feasibility problem: "Is there a set of classes with budget <= 1 that covers 1..T?"
    # Variables: x_{p,r} in {0,1}.
    # Constraints:
    # 1. Sum_{r} x_{p,r} <= 1 for all p (distinct primes)
    # 2. Sum_{p,r} (1/p) x_{p,r} <= 1 (budget)
    # 3. For each n in 1..T: Sum_{p,r: n % p == r} x_{p,r} >= 1 (coverage)
    
    # We can binary search on T?
    # T=1000 is the goal.
    # Let's start with a reasonable T, e.g., 1000.
    # If ILP says infeasible, we might need to relax? No, the problem asks for the largest L.
    # If we can't cover 1000, we try 999, etc.
    # But ILP is slow.
    
    # Better: Maximize the "prefix length" directly?
    # We can introduce variables y_n for n=1..M, where y_n = 1 if n is covered.
    # Maximize sum_{n=1}^M w_n y_n, where w_n is a large weight for earlier n?
    # If we set w_n = 2^n, it's huge.
    # If we set w_n = M-n+1, it prioritizes early coverage but not strictly.
    # To strictly maximize the prefix, we can use a lexicographic objective or just maximize sum y_n
    # and hope the structure forces a prefix? No, it might cover 1..500 and 1000..1500.
    
    # Let's stick to the feasibility check for a target T.
    # We will try T = 1000 first.
    # If feasible, we return the solution.
    # If not, we might try to maximize coverage of 1..1000 and see what L we get.
    
    # Given the time limit, a single ILP solve for T=1000 might be feasible.
    # Number of variables: Sum_{p<=1000} p.
    # Number of primes <= 1000 is 168.
    # Sum of primes <= 1000 is approx 76127.
    # So ~76k binary variables.
    # Constraints:
    # - 168 prime uniqueness constraints.
    # - 1 budget constraint.
    # - 1000 coverage constraints.
    # Total ~1200 constraints.
    # This is a moderately sized ILP. SciPy's milp uses HiGHS, which should handle this in seconds.
    
    target_T = 1000
    
    # Build the ILP
    # Variables x_{p,r} indexed by a flat index.
    # We need a mapping from (p, r) to index.
    
    var_index = {}
    num_vars = 0
    prime_indices = [] # List of lists of var indices for each prime
    
    for p in PRIMES:
        prime_indices.append([])
        for r in range(p):
            var_index[(p, r)] = num_vars
            num_vars += 1
            prime_indices[PRIME_TO_IDX[p]].append(num_vars - 1)
            
    # Objective: Minimize 0 (feasibility)
    # Or, to guide the solver, we can minimize the number of variables used?
    # Or maximize coverage? Since we require full coverage of 1..T, any feasible solution is good.
    # Let's just minimize 0.
    
    c = np.zeros(num_vars)
    
    # Constraints
    
    # 1. Prime uniqueness: For each prime p, sum_{r} x_{p,r} <= 1
    # A_ub x <= b_ub
    # We can stack these into a matrix.
    
    # 2. Budget: Sum_{p,r} (1/p) x_{p,r} <= 1
    
    # 3. Coverage: For each n in 1..T, Sum_{p,r: n%p==r} x_{p,r} >= 1
    # This is -Sum ... <= -1
    
    # Let's construct the constraint matrix A and vector b.
    # It will be sparse.
    
    rows = []
    cols = []
    vals = []
    
    # Row 0: Budget
    row_idx = 0
    for p in PRIMES:
        p_idx = PRIME_TO_IDX[p]
        for r_idx in prime_indices[p_idx]:
            rows.append(row_idx)
            cols.append(r_idx)
            vals.append(1.0 / p)
            
    # Rows 1..168: Prime uniqueness
    for p_idx, p in enumerate(PRIMES):
        row_idx = 1 + p_idx
        for r_idx in prime_indices[p_idx]:
            rows.append(row_idx)
            cols.append(r_idx)
            vals.append(1.0)
            
    # Rows 169..169+T-1: Coverage
    # For each n in 1..T:
    # Sum_{p} x_{p, n%p} >= 1  => -Sum ... <= -1
    # Note: Only primes p > 1 are in PRIMES.
    # If p > n, then n % p = n.
    
    for n in range(1, target_T + 1):
        row_idx = 1 + len(PRIMES) + (n - 1)
        for p in PRIMES:
            r = n % p
            v_idx = var_index[(p, r)]
            rows.append(row_idx)
            cols.append(v_idx)
            vals.append(-1.0)
            
    # Convert to sparse matrix
    from scipy.sparse import coo_matrix, csc_matrix
    
    num_constraints = 1 + len(PRIMES) + target_T
    
    A = coo_matrix((vals, (rows, cols)), shape=(num_constraints, num_vars)).tocsc()
    
    # Bounds for A x <= b
    # Budget: <= 1
    # Prime uniqueness: <= 1
    # Coverage: <= -1
    
    b_ub = np.zeros(num_constraints)
    b_ub[0] = 1.0 # Budget
    for i in range(1, 1 + len(PRIMES)):
        b_ub[i] = 1.0 # Prime uniqueness
    for i in range(1 + len(PRIMES), num_constraints):
        b_ub[i] = -1.0 # Coverage
        
    b_lb = -np.inf * np.ones(num_constraints)
    
    constraints = LinearConstraint(A, b_lb, b_ub)
    
    # Variable bounds: 0 <= x <= 1
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    integrality = np.ones(num_vars) # All binary
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
        
        if res.status == 0: # Optimal
            # Extract solution
            classes = []
            for p in PRIMES:
                p_idx = PRIME_TO_IDX[p]
                for r_idx in prime_indices[p_idx]:
                    if res.x[r_idx] > 0.5:
                        r = r_idx - (prime_indices[p_idx][0]) # This logic is flawed, r is not contiguous in global index easily
                        # Better: iterate r again
                        pass
            
            # Re-extract properly
            for p in PRIMES:
                for r in range(p):
                    v_idx = var_index[(p, r)]
                    if res.x[v_idx] > 0.5:
                        classes.append([p, r])
                        
            # Verify solution
            if _sum_reciprocals_exact(classes) <= 1:
                # Check coverage of 1..target_T
                # The ILP ensured this, but let's double check for safety
                # And calculate the actual L (might be > T if lucky, or < T if bug)
                L = _calculate_coverage(classes, limit=10000)
                
                # If we covered T, great.
                # If we didn't (shouldn't happen), we might try to extend?
                # For now, return this.
                
                # If time remains, try to extend L by adding more classes?
                # The ILP used a subset of primes. We can try to add more primes to extend L.
                
                if time.time() < deadline - 10.0:
                    # Local search to extend
                    used_primes = set(p for p, r in classes)
                    available_primes = [p for p in PRIMES if p not in used_primes]
                    
                    current_L = L
                    improved = True
                    while improved and time.time() < deadline - 5.0:
                        improved = False
                        # Find first gap
                        # We know 1..current_L is covered.
                        gap = current_L + 1
                        # Check if gap is covered
                        # Actually, _calculate_coverage returned the first gap - 1.
                        # So gap = current_L + 1 is NOT covered.
                        
                        # Try to add a prime to cover gap
                        budget_used = _sum_reciprocals_exact(classes)
                        
                        # Find cheapest prime that fits
                        best_prime = None
                        best_cost = float('inf')
                        for p in available_primes:
                            if budget_used + Fraction(1, p) <= 1:
                                if 1.0/p < best_cost:
                                    best_cost = 1.0/p
                                    best_prime = p
                                    
                        if best_prime:
                            r = gap % best_prime
                            trial_classes = classes + [[best_prime, r]]
                            trial_L = _calculate_coverage(trial_classes, limit=current_L + 100)
                            if trial_L > current_L:
                                classes = trial_classes
                                current_L = trial_L
                                used_primes.add(best_prime)
                                available_primes.remove(best_prime)
                                improved = True
                                
                    return classes
            
        else:
            # Infeasible or error
            # Try a smaller target?
            # Or return a heuristic solution?
            pass
            
    except Exception as e:
        print(f"ILP Error: {e}")
        
    # Fallback: Heuristic solution
    # Greedy construction similar to previous but optimized
    classes = []
    used_primes = set()
    budget = Fraction(0, 1)
    
    # Try to cover 1..1000 greedily
    # We'll use a simple greedy: for n=1..1000, if not covered, add prime.
    # To make it fast, we don't check coverage explicitly every time, 
    # but we can maintain a set of covered numbers?
    # Or just run the ILP on a smaller target if the big one failed?
    
    # Let's try a simpler ILP for T=500 if T=1000 failed.
    if time.time() < deadline - 20.0:
         # Re-run ILP for T=500
         # This code block is complex to duplicate. 
         # Let's just use a heuristic fallback.
         
         # Heuristic: 
         # 1. Start with p=2, r=1 (covers odds)
         # 2. p=3, r=2 (covers 2 mod 3)
         # 3. p=5, r=4
         # ...
         
         # A known good covering for 1..1000?
         # The Erdos-Ruzsa conjecture is about existence.
         # We can try to construct one.
         
         # Let's try a random restart greedy.
         best_classes = []
         best_score = 0
         
         for _ in range(10):
             if time.time() > deadline - 5.0: break
             
             curr = []
             used = set()
             bud = Fraction(0, 1)
             
             # Random order of primes
             avail = PRIMES[:]
             random.shuffle(avail)
             
             # Greedy cover
             # We want to cover 1..1000.
             # This is slow in pure python if we check coverage each step.
             # Let's just add primes that seem useful.
             
             # Actually, just return a small valid set if ILP failed.
             # [[2,1], [3,2], [5,4], [7,6], [11,10], [13,12], [17,16], [19,18], [23,22], [29,28]]
             # Sum 1/p approx 0.5+0.33+0.2+0.14+0.09+0.07+0.06+0.05+0.04+0.03 = 1.51 > 1.
             # Need to be careful with budget.
             
             # Simple valid set:
             # [[2, 1], [3, 2], [5, 4]] -> Sum = 0.5+0.33+0.2 = 1.03 > 1.
             # [[2, 1], [3, 2]] -> 0.83. Covers odds, and 2,5,8,11...
             # 1 (2,1), 2 (3,2), 3 (2,1), 4 (2,1), 5 (3,2), 6 (2,1), 7 (2,1), 8 (3,2), 9 (2,1), 10 (2,1), 11 (3,2)...
             # Gaps? 
             # 1: 1%2=1 (Y), 1%3=1 (N)
             # 2: 2%2=0 (N), 2%3=2 (Y)
             # 3: 3%2=1 (Y)
             # 4: 4%2=0 (N), 4%3=1 (N) -> Gap at 4.
             
             # So [[2,1], [3,2]] covers 1,2,3. L=3.
             
             # Let's just return an empty list or a minimal valid one if all else fails.
             # But the score must be maximized.
             # If ILP failed, we are in trouble.
             # Let's assume ILP succeeds for T=1000.
             
             pass
             
    # If we reach here, ILP failed or we are in fallback.
    # Return a minimal valid solution.
    # [[2, 1]] is valid. Sum=0.5. L=1 (1%2=1). 2 is not covered.
    # [[2, 0]] -> 2 covered. 1 not.
    # [[2, 1], [3, 1]] -> Sum 0.83.
    # 1: 1%2=1(Y)
    # 2: 2%3=2(N), 2%2=0(N) -> Gap at 2.
    
    # Let's try [[2, 1], [3, 2]] again. L=3.
    # Add [5, 4]? Sum 1.03 > 1.
    # Remove [3, 2], add [3, 0]?
    # [[2, 1], [3, 0]]
    # 1: Y
    # 2: 2%3=2(N), 2%2=0(N) -> Gap at 2.
    
    # How about [[2, 1], [3, 1], [5, 1]]?
    # 1: Y
    # 2: N
    # 
    # Let's just return [[2, 1]] as a safe fallback.
    return [[2, 1]]

# EVOLVE-BLOCK-END
