"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import prod

# EVOLVE-BLOCK-START

# Precompute primes up to 1000
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

PRIMES = _sieve(1000)
# Reciprocal values as Fractions for exact budget tracking
RECIPS = {p: Fraction(1, p) for p in PRIMES}
# Sort primes by size (larger primes are "cheaper" in reciprocal sum per unit of modulus)
# Actually, for covering, smaller primes are more "efficient" per residue but cost more in budget.
# We want to maximize L. The budget is sum(1/p) <= 1.

def _coverage_length(classes, max_check=100000):
    """
    Returns the largest L such that every integer 1..L is covered by some class [p, r].
    classes: list of [p, r]
    """
    if not classes:
        return 0
    # Build a set of covered residues for efficiency? No, L can be up to 10^5.
    # Checking each n from 1 to max_check is O(max_check * len(classes)).
    # 10^5 * ~20 = 2*10^6 operations, which is fast in Python if optimized.
    
    # Optimization: Instead of checking every n, we can compute the first gap.
    # But checking sequentially is simpler and robust.
    # We can stop early if we find a gap.
    
    # To speed up, let's pre-extract (p, r) pairs.
    pairs = [(p, r) for p, r in classes]
    
    n = 1
    while n <= max_check:
        covered = False
        for p, r in pairs:
            if n % p == r:
                covered = True
                break
        if not covered:
            return n - 1
        n += 1
    return max_check

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Initial heuristic: Use primes in descending order of size? 
    # Or use a known good covering set?
    # The seed program uses primes from 13 upwards.
    # Let's try a greedy approach: 
    # 1. Start with an empty set.
    # 2. At each step, try to add a prime p and a residue r that maximizes the new coverage length L, 
    #    subject to the budget constraint.
    # However, the choice of r matters greatly. 
    # A common strategy for covering systems is to choose residues that "align" with existing covers to extend the run.
    
    # Alternative structural change: 
    # Use a set of small primes with carefully chosen residues.
    # The reciprocal sum of primes 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71...
    # Sum of 1/p for first 10 primes: 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29
    # 0.5 + 0.333 + 0.2 + 0.142 + 0.09 + 0.076 + 0.058 + 0.052 + 0.043 + 0.034 = ~1.528 > 1.
    # So we can't use all small primes.
    
    # Let's try a different design: 
    # We will use a subset of primes. 
    # We'll use a local search (hill climbing) on the set of primes and their residues.
    
    # 1. Generate candidate primes.
    # We can afford primes with 1/p small. 
    # Let's consider primes from 2 up to some limit.
    # The budget is 1. 
    # If we pick primes p1, p2, ..., pk, sum(1/pi) <= 1.
    
    # Let's start with a greedy construction:
    # Start with the largest primes? No, small primes cover more frequently.
    # Actually, the "density" of a class p,r is 1/p.
    # To cover 1..L continuously, we need the union of these arithmetic progressions to be dense.
    
    # Let's try a specific known construction or a heuristic one.
    # Heuristic: 
    # Start with p=2, r=0 (covers evens).
    # Then p=3, r=1 (covers 1, 4, 7, 10... -> 1 mod 3). 
    # Evens: 2,4,6,8,10,12...
    # 1 mod 3: 1,4,7,10,13...
    # Union: 1,2,4,6,7,8,10,12,13...
    # Missing: 3,5,9,11...
    # Add p=5, r=3 (covers 3,8,13... -> 3 mod 5).
    # 3 mod 5: 3,8,13,18...
    # Union: 1,2,3,4,6,7,8,10,12,13...
    # Missing: 5,9,11,14...
    
    # This is a complex optimization. Let's use a randomized local search.
    
    # Representation: A list of (p, r) pairs.
    # Objective: Maximize L.
    # Constraints: Distinct primes, sum(1/p) <= 1.
    
    # Step 1: Initial population / solution.
    # Let's try to construct a solution using a greedy algorithm that adds primes one by one.
    
    def greedy_construct():
        used_primes = set()
        current_classes = []
        current_budget = Fraction(0)
        
        # Try adding primes in a specific order.
        # Order: Primes that provide the most "new" coverage for the current L.
        # But L depends on the whole set.
        
        # Let's try a simple heuristic:
        # Add primes in increasing order of size, choosing the residue that maximizes the coverage of the next few integers.
        
        # Actually, let's just try a few fixed sets and then optimize residues.
        
        # Set 1: Primes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97...
        # We need sum(1/p) <= 1.
        
        # Let's find a subset of primes with sum(1/p) <= 1.
        # We can use a knapsack-like approach or just greedy.
        
        # Greedy: Sort primes by 1/p (descending, i.e., smallest primes first).
        # Add prime if it fits in the budget.
        
        primes_to_consider = PRIMES[:]  # All primes up to 1000
        # Sort by 1/p descending (smallest primes first)
        primes_to_consider.sort(key=lambda p: -RECIPS[p])
        
        selected_primes = []
        budget = Fraction(0)
        for p in primes_to_consider:
            if budget + RECIPS[p] <= 1:
                selected_primes.append(p)
                budget += RECIPS[p]
            else:
                # Try smaller primes? No, we are going from smallest to largest.
                # If a small prime doesn't fit, larger ones won't either (since 1/p is smaller).
                # Wait, 1/p decreases as p increases.
                # So if 1/p doesn't fit, 1/q for q>p also doesn't fit.
                # So we can break?
                # No, we might have skipped some primes? No, we are iterating in order of increasing p.
                # So if p doesn't fit, no larger prime will fit.
                break
        
        # Now we have a set of primes.
        # We need to assign residues.
        # Let's use a local search on residues.
        
        return selected_primes

    # Step 2: Assign residues and optimize.
    
    def optimize_residues(primes, max_iters=5000, time_limit=100):
        # Initialize residues randomly
        residues = [p % p for p in primes]  # 0
        # Or random
        import random
        residues = [random.randint(0, p-1) for p in primes]
        
        best_L = _coverage_length([[p, r] for p, r in zip(primes, residues)])
        best_residues = residues[:]
        
        start_opt = time.time()
        iters = 0
        
        while iters < max_iters:
            if time.time() > start_opt + time_limit:
                break
            
            # Pick a random prime index
            idx = random.randint(0, len(primes)-1)
            p = primes[idx]
            old_r = residues[idx]
            
            # Try a new residue
            # Strategy: Choose r such that it covers a currently missing number near the current L.
            # Or just random.
            new_r = random.randint(0, p-1)
            
            if new_r == old_r:
                continue
            
            residues[idx] = new_r
            L = _coverage_length([[p, r] for p, r in zip(primes, residues)])
            
            if L > best_L:
                best_L = L
                best_residues = residues[:]
                # Maybe restart randomization?
            else:
                # Revert
                residues[idx] = old_r
            
            iters += 1
        
        return best_residues, best_L

    # Let's try multiple subsets of primes.
    
    # Subset 1: Greedy small primes
    primes_set_1 = greedy_construct()
    
    # Subset 2: Try to include 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Let's check the budget for the first 25 primes.
    # We can just let the greedy function handle it.
    
    # Let's also try a different greedy: 
    # Start with large primes? No, that's inefficient for covering.
    
    # Let's run optimization on the greedy set.
    
    best_classes = []
    best_L = 0
    
    # Run optimization
    import random
    random.seed(42)
    
    res, L = optimize_residues(primes_set_1, max_iters=20000, time_limit=50)
    if L > best_L:
        best_L = L
        best_classes = [[p, r] for p, r in zip(primes_set_1, res)]
    
    # Try another set: Maybe skip 2?
    # If we skip 2, we have more budget for other primes.
    # But 2 covers 50% of integers. It's very valuable.
    
    # Try a set that includes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101...
    # The greedy function already picks the smallest primes that fit.
    
    # Let's try to manually pick a set that might be better.
    # What if we use primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
    # The sum of reciprocals of all primes diverges, but we are capped at 1.
    
    # Let's try to improve the residue optimization.
    # The current local search is random. We can make it smarter:
    # When changing residue for prime p, try all residues 0..p-1 and pick the best one.
    # This is O(p) per step. For small p, this is fast.
    
    def optimize_residues_smart(primes, max_iters=5000, time_limit=100):
        import random
        residues = [random.randint(0, p-1) for p in primes]
        best_L = _coverage_length([[p, r] for p, r in zip(primes, residues)])
        best_residues = residues[:]
        
        start_opt = time.time()
        iters = 0
        no_improve_count = 0
        
        while iters < max_iters:
            if time.time() > start_opt + time_limit:
                break
            
            # Pick a random prime index
            idx = random.randint(0, len(primes)-1)
            p = primes[idx]
            
            # Try all residues for this prime
            best_r_for_p = residues[idx]
            best_L_for_p = best_L
            
            for r in range(p):
                if r == residues[idx]:
                    continue
                residues[idx] = r
                L = _coverage_length([[p, r] for p, r in zip(primes, residues)])
                if L > best_L_for_p:
                    best_L_for_p = L
                    best_r_for_p = r
            
            residues[idx] = best_r_for_p
            
            if best_L_for_p > best_L:
                best_L = best_L_for_p
                best_residues = residues[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    # Restart with random residues
                    residues = [random.randint(0, p-1) for p in primes]
                    best_L = _coverage_length([[p, r] for p, r in zip(primes, residues)])
                    best_residues = residues[:]
                    no_improve_count = 0
            
            iters += 1
        
        return best_residues, best_L

    # Run smart optimization
    res, L = optimize_residues_smart(primes_set_1, max_iters=5000, time_limit=60)
    if L > best_L:
        best_L = L
        best_classes = [[p, r] for p, r in zip(primes_set_1, res)]
    
    # Return the best classes found
    if best_classes:
        return best_classes
    else:
        # Fallback to seed
        primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]

# EVOLVE-BLOCK-END
