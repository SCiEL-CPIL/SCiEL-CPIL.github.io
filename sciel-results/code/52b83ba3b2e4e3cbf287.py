"""
Strategy: Greedy Set Cover with Gap-Filling Heuristic.

The previous strategy focused on covering the *next* integer n sequentially, 
often leading to suboptimal prime choices that don't efficiently cover 
subsequent integers. This strategy reformulates the problem as a set cover 
problem where we want to cover the range [1, L] with a set of arithmetic 
progressions (classes) such that the sum of 1/p is minimized (or kept <= 1).

Instead of a strict sequential "cover n, then n+1" approach, we:
1. Identify the current "frontier" of uncovered integers.
2. For each candidate prime p (not yet used, and affordable), calculate how 
   many *currently uncovered* integers in a lookahead window [current_frontier, 
   current_frontier + W] would be covered by the best residue for p.
   Note: The residue is fixed by the requirement to cover the current frontier? 
   No, the residue is a choice. 
   
   Wait, if we are covering a specific set of numbers S, for a prime p, we can 
   choose any residue r. The set of numbers covered is {x in S | x ≡ r (mod p)}.
   To maximize coverage of S, we should choose the residue r that appears most 
   frequently in S modulo p.
   
   However, we *must* cover the current frontier. Let F be the smallest 
   uncovered integer. We *must* add a class that covers F. 
   If we add (p, r), then r must be F % p.
   So the residue is forced if we want to cover F with p.
   
   Therefore, the choice for covering F is indeed just which prime p to use.
   But, we can look ahead. If we pick p for F, it covers F, F+p, F+2p...
   The "value" of p is how many *other* uncovered numbers it covers in the 
   near future.
   
   Algorithm:
   1. Maintain a set `uncovered` of integers in [1, L_max] not yet covered.
      Initially, we don't know L. We can simulate up to a limit, or dynamically 
      expand.
      Better: Maintain a boolean array `covered` up to some limit, say 2000.
      Start with all False.
      `current_min_uncovered` = 1.
      
   2. While `current_min_uncovered` <= limit and time remains:
      a. If `covered[current_min_uncovered]` is True, increment 
         `current_min_uncovered` and continue.
      b. Else, `current_min_uncovered` needs coverage. Let n = current_min_uncovered.
         We must add a class (p, n % p).
         Evaluate all available primes p (unused, budget allows).
         For each p, calculate the "gain": number of integers k in 
         [n, n + LOOKAHEAD] such that `covered[k]` is False AND `k % p == n % p`.
         (Note: n itself is always covered by this choice).
         
         We want to maximize gain / cost, or just gain, subject to cost constraint.
         Since we *must* cover n, we can't skip. We must pick one p.
         To preserve budget for the future, we prefer smaller cost (larger p) 
         UNLESS the gain is significantly higher for a smaller p.
         
         Heuristic Score: gain * weight_gain - cost * weight_cost?
         Or simply: Pick the p that maximizes gain. If tie, pick largest p.
         Or: Pick p that maximizes gain / (1/p) = gain * p.
         
         Let's try: Score = gain * p.
         This rewards primes that cover many future gaps efficiently.
         
      c. Add the chosen class. Update `covered` array for all k in range where 
         k % p == r.
         Update budget.
         Mark p as used.
         
      d. Update `current_min_uncovered`.
      
   3. To handle the "L" maximization, we keep going until we can't cover the 
      next number (no affordable prime) or we hit the time limit.
      
   Optimization:
   - The `covered` array update can be slow if done naively for every step.
   - We can maintain a list of active classes.
   - To check if n is covered: any(n % p == r for p, r in classes).
   - To calculate gain for candidate p: 
     count = 0
     r = n % p
     for k in range(n, n + LOOKAHEAD):
         if not covered[k] and k % p == r:
             count += 1
     This is O(LOOKAHEAD * num_candidates). 
     If LOOKAHEAD=100 and num_candidates=100, that's 10k ops per step.
     If we have 1000 steps, that's 10M ops. Feasible.
     
   - We need to be careful with the `covered` array. 
     When we add a class, we should update `covered[k]` for all k in the 
     relevant range. 
     Actually, we don't need to update `covered` for all k up to limit, 
     just up to `current_min_uncovered + LOOKAHEAD`. 
     But future steps will need to know if numbers are covered.
     So we should update `covered[k]` for all k in [n, limit] where k % p == r.
     This is O(limit / p). Summing over all primes, this is manageable.
     
   Let's refine the heuristic.
   If we pick a large p, gain is likely 1 (just n).
   If we pick a small p, gain might be high (n, n+p, n+2p... many in window).
   But small p costs 1/p which is large.
   
   Example: n=1.
   p=2, r=1. Covers 1, 3, 5... In window [1, 100], covers ~50 odds. Gain=50. Cost=0.5.
   p=997, r=1. Covers 1, 998... In window [1, 100], covers 1. Gain=1. Cost=1/997.
   
   If we pick p=2, we spend 0.5 budget to cover 50 numbers.
   If we pick p=997, we spend ~0 budget to cover 1 number.
   Clearly p=2 is better *if* we have budget.
   
   But wait, if we use p=2 for n=1, we cover all odds.
   n=2 is even. Not covered.
   Next step n=2.
   p=3, r=2. Covers 2, 5, 8... 
   Note: 5 is already covered by p=2. 
   So gain for p=3 at n=2:
   k=2 (uncovered, 2%3=2=2%3) -> Yes
   k=5 (covered) -> No
   k=8 (uncovered, 8%3=2=2%3) -> Yes
   ...
   
   The greedy "max gain" approach seems strong.
   We must ensure we don't waste budget on primes that cover already covered numbers.
   The gain calculation `if not covered[k]` handles this.
   
   One risk: Local minima. 
   Using small primes early might block using them later where they are more useful?
   No, because coverage is a union. Using a prime earlier is never "worse" 
   for coverage than using it later, provided the residue is the same. 
   But the residue is determined by the first number it covers.
   If we use p=2 for n=1 (r=1), we cover odds.
   If we saved p=2 for n=2 (r=0), we would cover evens.
   Which is better?
   If we cover odds first, we handle 1, 3, 5...
   If we cover evens first, we handle 2, 4, 6...
   Symmetry suggests it doesn't matter much, but the interaction with other primes 
   might.
   
   Let's stick with: 
   At step n, for each candidate p, compute gain = count of uncovered k in 
   [n, n+LOOKAHEAD] with k % p == n % p.
   Select p maximizing gain. Tie-break: larger p (cheaper).
   If max gain is 1 for all, select largest p (cheapest).
   
   We also need to handle the budget constraint strictly.
   
   Implementation details:
   - `covered` is a list of bools, size ~2000 (safe margin for L).
   - `classes` is a list of (p, r).
   - `used_primes` is a set.
   - `budget` is a Fraction.
   - `LOOKAHEAD` = 100 (tunable).
   
   Time complexity:
   Loop until stuck or time out.
   Each step:
   - Check if n covered: O(K) where K is num classes.
   - If not, evaluate ~100 candidates.
   - For each candidate, scan LOOKAHEAD window: O(LOOKAHEAD).
   - Total per step: O(K + 100 * LOOKAHEAD).
   - If L=1000, K~100, LOOKAHEAD=100.
   - 1000 * (100 + 10000) = 10M ops. Fast.
   
   Let's implement this.
"""

import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _get_primes(limit=1000):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    all_primes = _get_primes(1000)
    
    # We will simulate up to a limit. 
    # The score is capped at 10^5, but realistically we won't reach that.
    # Let's set a simulation limit of 2000 to be safe and fast.
    SIM_LIMIT = 2000
    LOOKAHEAD = 200
    
    # State
    covered = [False] * (SIM_LIMIT + 1)
    classes = []
    used_primes = set()
    budget = Fraction(0)
    
    # Pre-sort primes descending for tie-breaking (cheaper is better if gain is same)
    # But we need to evaluate all.
    
    n = 1
    while n <= SIM_LIMIT:
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
            
        # Check if n is already covered
        is_covered = False
        for p, r in classes:
            if n % p == r:
                is_covered = True
                break
        
        if is_covered:
            covered[n] = True
            n += 1
            continue
        
        # Not covered. Must add a class.
        # Find best prime p.
        best_p = None
        best_gain = -1
        best_r = 0
        
        # Calculate max affordable prime? 
        # We just iterate all unused primes and check budget.
        
        for p in all_primes:
            if p in used_primes:
                continue
            
            # Check budget
            if budget + Fraction(1, p) > 1:
                continue
                
            r = n % p
            
            # Calculate gain
            gain = 0
            # We count uncovered numbers in [n, n + LOOKAHEAD] that would be covered
            # Note: n is currently uncovered, so it counts as 1.
            # We must ensure we don't count numbers > SIM_LIMIT? 
            # The gain heuristic is local, so we just check up to min(n+LOOKAHEAD, SIM_LIMIT)
            end_check = min(n + LOOKAHEAD, SIM_LIMIT)
            
            # Optimization: Instead of checking every k, we can iterate 
            # k = n, n+p, n+2p... within the window.
            # This is faster if p is large.
            # If p is small, it's similar.
            
            k = n
            while k <= end_check:
                if not covered[k]:
                    gain += 1
                k += p
            
            # Heuristic: Maximize gain. Tie-break: Larger p (cheaper).
            # If gain is same, prefer larger p.
            # If gain is higher, prefer that p even if smaller?
            # Let's try: Score = gain * 1000 + p (to favor gain primarily, then cost)
            # Or simply: if gain > best_gain, or (gain == best_gain and p > best_p)
            
            if gain > best_gain or (gain == best_gain and (best_p is None or p > best_p)):
                best_gain = gain
                best_p = p
                best_r = r
        
        if best_p is None:
            # No affordable prime to cover n. Stuck.
            break
        
        # Commit to best_p
        used_primes.add(best_p)
        classes.append((best_p, best_r))
        budget += Fraction(1, best_p)
        
        # Update covered array
        # Mark all k in [1, SIM_LIMIT] where k % best_p == best_r as covered
        # We can do this by iterating k = best_r, best_r + best_p, ...
        # But we only care about k >= n? 
        # Actually, previous k < n are already processed. 
        # But marking them doesn't hurt.
        # For efficiency, start from the first k >= n such that k % best_p == best_r.
        # k_start = n + ((best_r - n) % best_p)
        # Wait, if n % best_p == best_r, then n is the first one.
        
        k = n
        # Ensure k is the first number >= n with residue best_r
        # Since we chose r = n % p, n is the first.
        while k <= SIM_LIMIT:
            covered[k] = True
            k += best_p
        
        # Advance n
        n += 1
        
    # Determine L: the largest L such that 1..L are covered.
    # Since we stopped at some n, L is n-1.
    # But we might have covered numbers beyond n?
    # No, we only stop if n is not covered and can't be covered.
    # So L = n - 1.
    
    # However, we should verify that 1..L are indeed covered.
    # The simulation ensures this.
    
    # Return the classes
    return [[p, r] for p, r in classes]

# EVOLVE-BLOCK-END
