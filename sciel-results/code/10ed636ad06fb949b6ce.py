import math
import random
import time
from fractions import Fraction

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
# Primes sorted by size, small primes are more "valuable" for density but cost more budget
# For covering long intervals, we generally want a mix.

def _calculate_coverage(classes, limit=5000):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    Uses a bytearray for speed.
    Returns L.
    """
    if not classes:
        return 0
    
    covered = bytearray(limit + 1)
    
    # Mark covered positions
    for p, r in classes:
        start = r if r > 0 else p
        if start < 1:
            start += p
        # Use slicing if possible? No, step is p.
        # Loop is fine for p >= 2.
        for n in range(start, limit + 1, p):
            covered[n] = 1
            
    # Find the largest L such that 1..L are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals_float(classes):
    """Fast float sum of reciprocals for budget checking."""
    return sum(1.0 / p for p, r in classes)

def _sum_reciprocals_exact(classes):
    """Exact Fraction sum for final verification."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 seconds safety margin
    
    # Initial heuristic solution
    # Strategy: Use the smallest primes? No, budget is tight.
    # Let's start with a known good covering system if possible, or a greedy start.
    # A simple greedy start: 
    # 1. Start with empty set.
    # 2. While budget allows, pick a prime that covers the most new numbers in 1..1000.
    #    Actually, just start with the seed's logic or a random valid set.
    # The seed uses primes >= 13. Let's try to construct a better initial state.
    
    # Greedy Initialization:
    # We want to cover 1..L.
    # Let's try to cover 1, 2, 3, ... sequentially.
    # For each number n, if it's not covered, add a prime class that covers it.
    # To be efficient, prefer smaller primes for larger density, but we must manage budget.
    
    classes = []
    used_primes = set()
    budget_used = 0.0
    max_budget = 1.0
    
    # We'll use a list of [p, r]
    # Let's try to cover 1..1000 initially.
    # For n from 1 to 1000:
    #   Check if covered.
    #   If not, find a prime p not in used_primes such that 1/p fits budget.
    #   Among those, pick one that covers the most *future* numbers? 
    #   Simplest: pick the smallest available prime that fits budget? 
    #   Smallest prime p=2 costs 0.5. p=3 costs 0.33.
    #   If we pick p=2, r=1, we cover odd numbers.
    #   If we pick p=2, r=0, we cover even numbers.
    
    # Let's do a randomized greedy initialization to avoid local minima.
    # We will run multiple random restarts of the greedy construction.
    
    best_classes = []
    best_score = 0
    
    # Run several greedy constructions with different randomizations
    for restart in range(50):
        if time.time() > deadline - 20.0: break
        
        current_classes = []
        used_primes = set()
        current_budget = 0.0
        
        # Shuffle primes to break symmetry
        avail_primes = [p for p in PRIMES if p <= 1000]
        random.shuffle(avail_primes)
        
        # Greedy: Try to cover 1..2000
        # We don't check coverage explicitly every step for speed, 
        # but we can be smart about residues.
        
        # Actually, a better greedy:
        # 1. Start with empty.
        # 2. For n = 1 to 2000:
        #    if n is not covered by current classes:
        #       find a prime p in avail_primes not in used_primes 
        #       such that current_budget + 1/p <= 1.0
        #       If found, pick the one with smallest p (highest density) 
        #       or one that covers the most subsequent numbers?
        #       Let's pick the smallest p to maximize future coverage potential.
        #       Set r = n % p.
        #       Add [p, r] to current_classes.
        #       Update budget.
        
        # To check coverage efficiently during construction, we can maintain a set or array.
        # But 2000 is small.
        covered = [False] * 2001
        
        for n in range(1, 2001):
            if covered[n]:
                continue
                
            # Find a prime to cover n
            candidate_p = None
            candidate_r = None
            min_cost = 1.0 # We want smallest p (largest 1/p) to be "greedy" on density?
            # Actually, we want to conserve budget for later. 
            # But covering n now requires a prime.
            # Using a small prime is expensive. Using a large prime is cheap but covers fewer numbers.
            # However, if we use a large prime, we might run out of cheap primes later.
            # Let's try picking the *largest* available prime that fits budget? 
            # No, that covers very few numbers. 
            # Let's pick the *smallest* available prime that fits budget? 
            # This burns budget fast but covers many numbers.
            
            # Let's try a mix: pick a prime randomly from those that fit?
            # Or pick the one that minimizes 1/p (cheapest) to save budget?
            # If we pick cheapest (largest p), we cover 1 number per p.
            # If we pick expensive (smallest p), we cover 1 number per p/p.
            
            # Let's try picking the smallest p (most efficient coverage per unit budget? 
            # No, 1/p is the cost. Coverage is 1/p of the numbers. 
            # So efficiency (coverage/cost) is constant! 
            # So it doesn't matter for *total* coverage, only for *local* gaps.
            # Small primes have smaller gaps. So they are better for covering contiguous ranges.
            
            # So, pick the smallest available prime that fits the budget.
            
            for p in avail_primes:
                if p in used_primes:
                    continue
                if current_budget + 1.0/p <= 1.0 + 1e-9:
                    candidate_p = p
                    break
            
            if candidate_p is None:
                # No prime fits budget to cover n. 
                # We can't extend coverage further with this strategy.
                break
                
            r = n % candidate_p
            current_classes.append([candidate_p, r])
            used_primes.add(candidate_p)
            current_budget += 1.0 / candidate_p
            
            # Mark covered numbers
            start = r if r > 0 else candidate_p
            if start < 1:
                start += candidate_p
            for k in range(start, 2001, candidate_p):
                covered[k] = True

        # Evaluate this solution
        score = _calculate_coverage(current_classes, limit=5000)
        if score > best_score:
            best_score = score
            best_classes = [c[:] for c in current_classes]
            
    # Now, Local Search / Refinement on the best initial solution
    # We will try to fix gaps by changing residues or swapping primes.
    
    current_classes = [c[:] for c in best_classes]
    current_score = best_score
    best_classes = [c[:] for c in current_classes]
    
    used_primes = set(p for p, r in current_classes)
    available_primes = [p for p in PRIMES if p not in used_primes]
    
    # Sort available primes by size
    available_primes.sort()
    
    # Local Search Loop
    # We will repeatedly find the first gap and try to fix it.
    # If we can't fix it, we might try to remove a "bad" class and re-optimize.
    
    # To speed up, we'll use a smaller limit for intermediate checks, say 1000.
    # But we want to maximize L.
    
    improvement_found = True
    while improvement_found and time.time() < deadline - 5.0:
        improvement_found = False
        
        # Find the first gap
        # We check up to current_score + 100
        check_limit = current_score + 200
        if check_limit > 5000:
            check_limit = 5000
            
        # Re-calculate coverage up to check_limit to find gap
        covered = bytearray(check_limit + 1)
        for p, r in current_classes:
            start = r if r > 0 else p
            if start < 1:
                start += p
            for n in range(start, check_limit + 1, p):
                covered[n] = 1
                
        gap = -1
        for i in range(1, check_limit + 1):
            if not covered[i]:
                gap = i
                break
        
        if gap == -1:
            # No gap found in this range, we might have covered more than we thought?
            # Or we are stuck. Let's try to extend the limit.
            # Actually, if no gap in 1..check_limit, then L >= check_limit.
            # We should update current_score to at least check_limit.
            if check_limit > current_score:
                current_score = check_limit
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
                improvement_found = True
                continue
            else:
                # Stuck. Try a different move type: random perturbation?
                # Or break.
                break
                
        # We have a gap at 'gap'.
        # Try to cover it.
        
        # Option 1: Change residue of an existing prime.
        # For each existing prime p, if we set r = gap % p, does it help?
        # It definitely covers 'gap'. But it might uncover other numbers.
        # We need to check if the new coverage extends L.
        
        best_new_score = current_score
        best_new_classes = None
        
        # Try changing residues
        for idx, (p, r) in enumerate(current_classes):
            new_r = gap % p
            if new_r == r:
                continue
                
            # Create a copy
            trial_classes = [c[:] for c in current_classes]
            trial_classes[idx] = [p, new_r]
            
            # Quick check: does this cover the gap? Yes, by definition.
            # Does it cover 1..gap? We don't know.
            # We need to evaluate coverage.
            # To save time, only evaluate if the prime is "small" or if we haven't tried many.
            # Let's just evaluate.
            
            # Optimization: Only evaluate if p is not too large?
            # No, any prime can help.
            
            # Let's limit the number of trials per gap to avoid O(N^2) explosion.
            # Actually, N is small (< 100).
            
            trial_score = _calculate_coverage(trial_classes, limit=current_score + 100)
            
            if trial_score > best_new_score:
                best_new_score = trial_score
                best_new_classes = trial_classes
                
        # Option 2: Add a new prime.
        # If we have budget, we can add a prime p with r = gap % p.
        current_budget = _sum_reciprocals_float(current_classes)
        
        # Find a prime that fits budget
        # Prefer small primes?
        for p in available_primes:
            if p in used_primes:
                continue
            if current_budget + 1.0/p <= 1.0 + 1e-9:
                new_r = gap % p
                trial_classes = [c[:] for c in current_classes]
                trial_classes.append([p, new_r])
                
                trial_score = _calculate_coverage(trial_classes, limit=current_score + 100)
                
                if trial_score > best_new_score:
                    best_new_score = trial_score
                    best_new_classes = trial_classes
                # Don't break, maybe a smaller prime is better? 
                # We iterate available_primes sorted by size, so first fit is smallest.
                # But a larger prime might be better if it doesn't conflict?
                # Residue conflict doesn't exist between different primes.
                # So smaller prime is generally better for density.
                # But we only need to cover 'gap'.
                # Let's try just the first few candidates.
                if len([c for c in trial_classes if c[0] == p]) > 0:
                     pass # Already added
                # Let's just try the smallest available prime that fits.
                break 

        # Option 3: Swap a prime.
        # Remove a prime that is "least useful" and add a better one?
        # This is complex. Let's skip for now and rely on residue changes and additions.
        
        if best_new_classes is not None and best_new_score > current_score:
            current_classes = best_new_classes
            current_score = best_new_score
            best_classes = [c[:] for c in current_classes]
            best_score = current_score
            improvement_found = True
            
            # Update used/available
            used_primes = set(p for p, r in current_classes)
            available_primes = [p for p in PRIMES if p not in used_primes]
            available_primes.sort()
            
    # Final verification with exact rationals
    if _sum_reciprocals_exact(best_classes) > 1:
        # Fallback to a known valid simple solution
        # e.g. just cover 1..100 with a few primes
        fallback = [[2, 1], [3, 2], [5, 4], [7, 6]]
        # Check validity
        if _sum_reciprocals_exact(fallback) <= 1:
            return fallback
        else:
            return []
            
    return best_classes

# EVOLVE-BLOCK-END
