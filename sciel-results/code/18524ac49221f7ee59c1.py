"""
Strategy: Greedy Backtracking with Lookahead for Sequential Coverage.

Instead of fixing a set of primes and optimizing residues (local search),
this strategy builds the covering set sequentially.
1. We iterate $n = 1, 2, 3, \dots$.
2. If $n$ is already covered by existing classes, we proceed.
3. If $n$ is not covered, we must add a new class $(p, r)$ such that $n \equiv r \pmod p$.
   Since $r$ is determined by $n$ and $p$ ($r = n \pmod p$), the choice is solely which prime $p$ to use.
4. We select $p$ from available primes (not yet used, $p \le 1000$) such that:
   a) The new total budget $\sum 1/p \le 1$.
   b) The choice maximizes "coverage potential". We use a heuristic score:
      score(p) = (number of integers in [n+1, n+K] covered by $x \equiv n \pmod p$) / (1/p)
      This favors primes that cover many nearby integers per unit budget cost.
5. We commit to this choice and continue.
6. If we reach a point where no prime can be added to cover the current $n$ without
   exceeding the budget, we have a dead end. In a full backtracking search, we would
   undo the last choice and try the next best prime. 
   
   Given the time limit (120s) and the depth (up to ~1000 steps), full backtracking 
   might be too slow if we explore many branches. However, the problem structure 
   (covering 1..L) is very rigid. Often, the "greedy" choice is forced or has few 
   alternatives. We will implement a depth-limited backtracking that explores 
   alternative primes if the immediate greedy choice leads to a quick dead end.
   
   To keep it fast, we primarily use a strong greedy heuristic with limited 
   backtracking (e.g., try top 5 candidates if the top one fails).
"""

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin

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
    
    # State
    # classes: list of (p, r)
    # used_primes: set of primes already used
    # budget_used: Fraction
    # covered: boolean array or set? 
    # Since we go sequentially, we just need to know if current n is covered.
    # We can maintain a list of active classes.
    
    best_classes = []
    best_L = 0
    
    # We will use a recursive backtracking function with a depth limit 
    # or a greedy approach with "undo" capabilities.
    # Given the strict budget (sum 1/p <= 1), the number of primes is limited (~168 max, but realistically fewer due to cost).
    # The length L is capped at 10^5, but we stop when we can't cover n+1.
    
    # To make it efficient, we precompute residues for all primes for a given n? 
    # No, n changes.
    
    # Let's use a greedy strategy with limited backtracking.
    # We maintain a stack of decisions.
    
    # Heuristic for choosing prime p to cover n:
    # 1. Must be unused.
    # 2. budget_used + 1/p <= 1.
    # 3. Maximize "future coverage". 
    #    A prime p covers n, n+p, n+2p, ...
    #    The "density" is 1/p.
    #    We want to cover the immediate next few numbers if possible?
    #    Actually, adding (p, n%p) covers n, n+p, n+2p...
    #    It does NOT help cover n+1, n+2, etc. directly unless p is small.
    #    If p=2, it covers n, n+2, n+4. It leaves n+1, n+3 to others.
    #    If p is large, it covers n and then a far future number.
    #    
    #    Key Insight: To cover 1..L continuously, we need a mix of small primes 
    #    to handle the "gaps" between larger prime hits.
    #    Small primes are expensive (1/p is large).
    #    We should use small primes ONLY when necessary to cover a number 
    #    that cannot be covered by a cheaper (larger) prime? 
    #    Wait, a larger prime p covers n. It doesn't matter if p is large or small 
    #    for covering n itself. The cost is 1/p.
    #    So, to cover n, we should ALWAYS pick the LARGEST available prime p 
    #    that fits the budget? 
    #    Why? Because 1/p is smaller for larger p, saving budget for future numbers.
    #    Does using a large prime hurt future coverage?
    #    Using p covers n, n+p, n+2p...
    #    Using q (smaller) covers n, n+q, n+2q...
    #    The set of covered numbers is different.
    #    However, since we are covering 1..L sequentially, the "future" is just n+1, n+2...
    #    The class (p, n%p) will help cover n+p. 
    #    If we pick a very large p, the next hit is far away. 
    #    If we pick a small p, the next hit is sooner.
    #    But we have other primes to cover the gap.
    #    
    #    Hypothesis: Greedy by "Cheapest Cost" (Largest Prime) is the primary strategy.
    #    If that leads to a dead end, we try the next largest, etc.
    
    def try_greedy_backtrack(max_depth=3):
        """
        Attempts to extend the coverage as far as possible.
        Uses greedy choice (largest prime fitting budget) but backtracks if stuck.
        """
        classes = []
        used_primes = set()
        budget_used = Fraction(0)
        L = 0
        
        # We simulate up to some limit, say 10000, or until stuck.
        # Since we want max L, we go until we can't cover L+1.
        
        # To allow backtracking, we need to store states.
        # A full DFS is too slow. 
        # We will use a "beam search" or limited backtracking.
        
        # Actually, let's just do a very fast greedy first.
        # If it gets stuck early, we can restart with a different seed or heuristic.
        
        # Let's implement a simple greedy first to see the baseline.
        # Then, if time permits, add backtracking for the "stuck" cases.
        
        # Greedy Step:
        # Current uncovered number is L+1.
        # Find all primes p not in used_primes such that budget_used + 1/p <= 1.
        # Among these, pick the one that maximizes "utility".
        # Utility: 
        #   If we pick p, we cover L+1.
        #   We also cover (L+1)+k*p.
        #   The "value" of p might be related to how many OTHER numbers it covers 
        #   that are currently UNCOVERED by existing classes?
        #   But existing classes only cover numbers <= L (since we are at L+1).
        #   So (L+1)+k*p is > L+1.
        #   So all future hits are "new" coverage.
        #   Therefore, the utility is just "saving budget".
        #   So we should ALWAYS pick the largest prime p available that fits the budget.
        
        # Let's test this logic.
        # If we always pick the largest prime, we save the most budget.
        # Does this maximize L?
        # Example: 
        # n=1. Primes available: 2,3,5...
        # Pick 997. Cost 1/997. Covers 1, 998, ...
        # n=2. 2 not covered. Pick 991. Cost 1/991. Covers 2, 993...
        # ...
        # We will use up the budget with ~168 primes (sum 1/p ~ 1).
        # The primes will be the largest ones.
        # Will this cover 1..L?
        # With 168 large primes, the density of coverage is sum(1/p) ~ 1.
        # The "gaps" between hits of large primes are large.
        # e.g., p=997 hits every 997 numbers.
        # With many such primes, the union of arithmetic progressions might cover everything?
        # This is the Jacobsthal function territory.
        # If we use the largest primes, the moduli are large.
        # The Chinese Remainder Theorem implies that the system of congruences 
        # x != r_i (mod p_i) has a solution if the p_i are distinct primes.
        # The gap between covered numbers is bounded by the product of p_i? No.
        # The maximum gap is related to the Jacobsthal function h(k) for the set of primes.
        # For a set of primes with sum of reciprocals close to 1, the coverage is dense.
        # However, using LARGE primes means the individual progressions are sparse.
        # Using SMALL primes means the progressions are dense.
        # 
        # Counter-argument to "Largest Prime":
        # If I pick p=2 for n=1, I cover 1, 3, 5, 7... (all odds).
        # Cost 1/2.
        # If I pick p=997 for n=1, I cover 1, 998, 1995...
        # Cost 1/997.
        # For n=2, if I used p=2, 2 is not covered (2%2=0, residue was 1). 
        # Wait, if I pick (2, 1) for n=1, it covers 1, 3, 5...
        # n=2 is even. Not covered.
        # I need another prime for 2. Say p=3, r=2. Covers 2, 5, 8...
        # Cost 1/3. Total 5/6.
        # n=3. Covered by p=2 (3%2=1).
        # n=4. 4%2=0 (no), 4%3=1 (no, r=2). Not covered.
        # Need p=5, r=4. Covers 4, 9, 14...
        # Cost 1/5. Total 5/6 + 1/5 = 31/30 > 1. Budget exceeded!
        # 
        # So, using small primes is very expensive.
        # Using large primes is cheap but sparse.
        # 
        # The optimal strategy is likely a mix.
        # We should use large primes to "fill in" gaps where small primes are too expensive, 
        # but we must ensure that the large primes actually hit the numbers we need.
        
        # Revised Heuristic:
        # When at n, we must pick a prime p.
        # The choice of p determines r = n % p.
        # This class will cover n, n+p, n+2p...
        # It will NOT cover n+1, n+2, etc. (unless p is small).
        # So the "immediate" future (n+1, n+2...) is unaffected by the choice of p 
        # EXCEPT for the budget constraint.
        # Therefore, to maximize the number of future steps we can take, 
        # we should MINIMIZE the cost 1/p.
        # This implies picking the LARGEST available prime.
        
        # So, the greedy strategy "Pick Largest Prime" is theoretically sound 
        # for maximizing the NUMBER OF CLASSES we can add.
        # Does maximizing the number of classes maximize L?
        # Not necessarily. We need the classes to be "aligned" to cover 1..L.
        # But since we are forced to cover 1, 2, 3... sequentially, 
        # and each new class covers the current n, 
        # the only degree of freedom is WHICH prime covers n.
        # 
        # If we pick the largest prime, we save budget.
        # If we pick a smaller prime, we spend more budget.
        # Why would we ever pick a smaller prime?
        # Maybe because the larger prime's progression will interfere with future coverage?
        # No, coverage is a union. More coverage is always better (or neutral).
        # The only constraint is the budget.
        # So, saving budget is strictly better.
        # 
        # Conclusion: The optimal strategy is to ALWAYS pick the largest available prime 
        # that fits the budget.
        
        # Let's implement this and see how far it gets.
        
        # Optimization: 
        # We don't need to check all primes. We can sort them descending.
        # We maintain a pointer to the largest unused prime.
        
        primes_desc = sorted(all_primes, reverse=True)
        used = [False] * 1001
        
        # We need to check if n is covered by EXISTING classes.
        # Existing classes: list of (p, r).
        # To check if n is covered: any(n % p == r for p, r in classes)
        # This is O(K) where K is number of classes.
        
        # We will simulate.
        
        classes = []
        budget = Fraction(0)
        L = 0
        
        # Pre-compute indices for primes_desc
        # We need to find the largest unused prime p such that budget + 1/p <= 1.
        
        # To speed up "is n covered", we can use a numpy array or a set of residues?
        # No, residues depend on n.
        # Just loop through classes.
        
        # Time limit check
        while time.time() - _START_TIME < _TIME_LIMIT:
            n = L + 1
            
            # Check if n is covered
            covered = False
            for p, r in classes:
                if n % p == r:
                    covered = True
                    break
            
            if covered:
                L += 1
                continue
            
            # Not covered. Must add a class.
            # Find largest unused prime p such that budget + 1/p <= 1.
            
            best_p = None
            best_idx = -1
            
            # Iterate primes_desc from largest to smallest
            # Stop if 1/p is too big? 
            # budget + 1/p <= 1  =>  1/p <= 1 - budget
            # p >= 1 / (1 - budget)
            
            threshold = Fraction(1, 1) - budget
            if threshold <= 0:
                break # Budget exhausted
            
            # We need p such that 1/p <= threshold => p >= 1/threshold
            min_p_needed = int(1 / float(threshold)) + 1
            
            for i, p in enumerate(primes_desc):
                if used[p]:
                    continue
                if p < min_p_needed:
                    # Since primes_desc is sorted, all subsequent p are smaller.
                    # If p < min_p_needed, then 1/p > 1/min_p_needed >= threshold.
                    # So 1/p > threshold. Budget exceeded.
                    break
                
                # Check budget exactly
                if budget + Fraction(1, p) <= 1:
                    best_p = p
                    best_idx = i
                    break
            
            if best_p is None:
                # No prime can be added. Stuck.
                break
            
            # Add class (best_p, n % best_p)
            r = n % best_p
            classes.append((best_p, r))
            used[best_p] = True
            budget += Fraction(1, best_p)
            
            # Update L? 
            # We just covered n. So L becomes at least n.
            # But maybe n+1 is also covered?
            # We will check in the next iteration.
            L = n
            
            # Optimization: 
            # If L is very large, the loop `for p, r in classes` becomes slow.
            # If L > 10000, we might need a faster coverage check.
            # But with ~168 primes, 168 * 10000 = 1.68M ops.
            # Python can do ~10M ops/sec. So 1.68M is ~0.1s.
            # Total steps 10000 * 0.1s = 1000s. Too slow.
            
            # We need a faster coverage check.
            # Idea: Use a bitset for the next K numbers?
            # Or, note that if we have many classes, the density is high.
            # 
            # Let's optimize the coverage check.
            # Instead of checking all classes, we can maintain a "next_hit" for each class?
            # No, n increments by 1.
            
            # Alternative: 
            # If L is small, the loop is fast.
            # If L is large, we have many classes.
            # 
            # Let's add a time check inside the loop to break if too slow.
            # But we want to maximize L.
            
            # Let's try to optimize the inner loop.
            # Use local variables.
            
            if L > 5000:
                # Switch to a faster method?
                # Maybe break if time is running out.
                if time.time() - _START_TIME > _TIME_LIMIT - 5:
                    break
        
        return [[p, r] for p, r in classes]
    
    # Run the greedy strategy
    result = try_greedy_backtrack()
    
    # Verify budget
    total = Fraction(0)
    for p, r in result:
        total += Fraction(1, p)
    
    if total > 1:
        # Fallback: remove the last few classes until valid
        while total > 1 and result:
            p, r = result.pop()
            total -= Fraction(1, p)
            
    return result

# EVOLVE-BLOCK-END
