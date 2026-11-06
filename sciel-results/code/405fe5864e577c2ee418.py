"""
Strategy: Greedy prefix extension with exact budget checks.
1. Start with an empty set of classes.
2. Find the first uncovered integer n (initially 1).
3. For each candidate prime p not yet used, check if adding (p, n % p) covers n.
4. Among all such candidates, select the one with the smallest reciprocal cost 1/p 
   that keeps the total budget sum(1/p) <= 1.
5. Add the chosen class and repeat.
6. If no prime can be added without exceeding the budget, stop.
This approach directly targets the goal of maximizing L by always extending the 
covered prefix, using the cheapest available prime to do so, ensuring we get the 
most "coverage per unit cost" at each step.
"""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit=1000):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _compute_coverage(primes, residues, cap=100000):
    """
    Returns the largest L such that every integer 1..L is covered.
    """
    if not primes:
        return 0
    
    covered = [False] * (cap + 1)
    for p, r in zip(primes, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        # Correct logic for start:
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            continue
        for n in range(start, cap + 1, p):
            covered[n] = True
            
    L = 0
    while L < cap and covered[L + 1]:
        L += 1
    return L

def construct():
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return []

    all_primes = _get_primes(1000)
    
    # We want to select a subset of primes and assign residues to cover 1..L.
    # Greedy strategy:
    # 1. Find the first uncovered number n.
    # 2. For each prime p not yet used, consider adding (p, n % p).
    # 3. Choose the prime p that minimizes 1/p (i.e., largest p) among those 
    #    where adding (p, n % p) does not exceed the budget.
    #    Wait, minimizing 1/p means choosing the LARGEST p.
    #    But larger p covers fewer numbers.
    #    Actually, the cost is 1/p. We have a budget of 1.
    #    We want to maximize L.
    #    Using small primes (large 1/p) is expensive but covers many numbers.
    #    Using large primes (small 1/p) is cheap but covers few numbers.
    #    
    #    Let's re-evaluate.
    #    The seed uses primes >= 13. Sum(1/p) ~ 0.92.
    #    If we use 2, cost is 0.5. We have 0.5 left.
    #    If we use 3, cost is 0.333. We have 0.666 left.
    #    
    #    The greedy approach should be:
    #    At each step, find the smallest prime p (not yet used) such that 
    #    adding (p, r) where r = n % p keeps the total cost <= 1.
    #    Why smallest p? Because it covers the most numbers (densest coverage).
    #    But it's also the most expensive.
    #    
    #    Let's try a different greedy:
    #    For the first uncovered number n, we MUST cover it.
    #    We can cover it with any prime p by setting r = n % p.
    #    We want to pick the p that allows us to cover the most FUTURE numbers 
    #    for the least cost.
    #    This is complex.
    #    
    #    Let's stick to the budget constraint.
    #    We have budget B = 1.
    #    Current cost C.
    #    For each candidate prime p, if C + 1/p <= B, we can add it.
    #    Among all valid p, which one to pick?
    #    The one that maximizes the new coverage length L.
    #    This requires evaluating coverage for each candidate p, which is expensive.
    #    
    #    Heuristic: Pick the smallest prime p (not yet used) that fits the budget.
    #    Small primes provide denser coverage.
    
    used_primes = set()
    classes = []
    current_cost = Fraction(0, 1)
    L = 0
    
    while time.time() - _START_TIME < _TIME_LIMIT:
        # Find next uncovered number
        n = L + 1
        
        # If we've gone beyond 1000, we can stop, but the cap is 10^5.
        # However, the budget might run out before that.
        
        best_p = None
        best_r = None
        
        # Try all unused primes
        for p in all_primes:
            if p in used_primes:
                continue
            
            r = n % p
            new_cost = current_cost + Fraction(1, p)
            
            if new_cost <= 1:
                # This prime is affordable.
                # Heuristic: prefer smaller p (denser coverage)
                # But we need to check if this actually helps extend L.
                # Since we are covering n, L will at least become n.
                # To maximize L, we want the prime that covers the most subsequent numbers.
                # Smaller p covers more.
                # So, pick the smallest affordable p.
                best_p = p
                best_r = r
                break # Since all_primes is sorted, the first affordable one is the smallest.
        
        if best_p is None:
            # No affordable prime left to cover n.
            # We cannot extend L further.
            break
        
        # Add the chosen class
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        current_cost += Fraction(1, best_p)
        
        # Update L
        # We know n is covered. But maybe n+1, n+2... are also covered by the new class or existing ones.
        # We need to recompute L or extend it.
        # Efficient extension:
        # Check n, n+1, n+2... until we find an uncovered one.
        # But we only added one class. The existing classes cover 1..L.
        # The new class covers n.
        # It might also cover n+1, etc.
        # Let's just recompute L from scratch? That's O(L * num_classes).
        # L can be up to 10^5. num_classes up to 168.
        # 10^5 * 168 = 1.68e7 ops. This is fine in Python if done in C (numpy) but slow in pure Python.
        # However, we do this in a loop. How many iterations?
        # Each iteration adds one prime. Max 168 iterations.
        # 168 * 1.68e7 = 2.8e9 ops. Too slow in pure Python.
        
        # We need a faster way to update L.
        # Or, we can just run the full _compute_coverage at the end?
        # But we need L to know which n to cover next.
        
        # Alternative:
        # Maintain a boolean array `covered` of size 10^5.
        # When we add a class (p, r), update the array.
        # Then find the first False in the array starting from L+1.
        
    # Let's refactor to maintain the covered array incrementally.
    
    # Restart with incremental update
    used_primes = set()
    classes = []
    current_cost = Fraction(0, 1)
    L = 0
    
    cap = 100000
    covered = [False] * (cap + 1)
    
    # Initialize covered for 1..0 (none)
    
    while time.time() - _START_TIME < _TIME_LIMIT:
        # Find next uncovered number
        # L is the last covered. So n = L + 1.
        # But we need to make sure L is correct.
        # Initially L=0. n=1.
        
        n = L + 1
        
        if n > cap:
            break
            
        best_p = None
        best_r = None
        
        # Try all unused primes
        for p in all_primes:
            if p in used_primes:
                continue
            
            r = n % p
            new_cost = current_cost + Fraction(1, p)
            
            if new_cost <= 1:
                # This prime is affordable.
                # Pick the smallest affordable p.
                best_p = p
                best_r = r
                break
        
        if best_p is None:
            # No affordable prime left to cover n.
            break
        
        # Add the chosen class
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        current_cost += Fraction(1, best_p)
        
        # Update covered array
        p = best_p
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        if start < 1:
            continue
        for k in range(start, cap + 1, p):
            covered[k] = True
        
        # Update L
        # Extend L as far as possible
        while L < cap and covered[L + 1]:
            L += 1
            
    # Final check
    # The loop breaks when no prime can be added to cover L+1.
    # This means L is the maximum possible with this greedy strategy.
    
    # However, is it possible that a different choice of prime for a previous n 
    # would have led to a larger L?
    # Yes, greedy is not always optimal.
    # But it's a strong heuristic.
    
    # Let's also try to see if we can improve by swapping primes?
    # No, time is limited.
    
    return classes
# EVOLVE-BLOCK-END
