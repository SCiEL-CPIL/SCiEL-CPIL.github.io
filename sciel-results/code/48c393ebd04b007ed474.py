import time
from fractions import Fraction
import random

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _get_primes(1000)
    L_target = 100000 # Cap for evaluation
    
    # We want to cover 1..L. 
    # Initial budget is 1.
    # We will maintain a set of (p, r) classes.
    
    best_classes = []
    best_L = 0
    
    # Helper to calculate the longest prefix covered by a set of classes
    def calc_L(classes):
        if not classes:
            return 0
        # For each n, check if any class covers it.
        # To speed up, we can just check sequentially until we find a gap.
        # But checking 100k integers for each candidate is slow if done naively in Python loops.
        # Optimization: Represent coverage as a boolean array or use a set of covered numbers?
        # Better: Since we are doing local search, we can track covered numbers incrementally?
        # For simplicity and robustness within time limit, we'll use a fast check for the prefix.
        
        # Pre-compute nothing, just check 1..L_target.
        # To make it faster, we can break early if a number is not covered.
        # However, constructing a boolean array of size 100001 is fast.
        
        # Let's use a bytearray for speed.
        covered = bytearray(L_target + 1)
        
        for p, r in classes:
            # Mark n such that n % p == r
            # Start from r if r >= 1, else r+p
            start = r if r >= 1 else r + p
            # If r == 0, start is p. If r > 0, start is r.
            # Actually, range(r, L_target+1, p) works if we handle r=0 correctly?
            # If r=0, 0 is not in 1..L. So start at p.
            # If r>0, start at r.
            
            # Using slice assignment on bytearray is very fast in C.
            # But we need to OR them.
            # Let's stick to a loop for the marking part, but it's 1000 primes * 100000/p operations.
            # Sum(100000/p) for p<=1000 is approx 100000 * log(log(1000)) ~ 100000 * 2.2 = 220k operations.
            # This is very fast.
            
            for n in range(start, L_target + 1, p):
                covered[n] = 1
                
        # Find the longest prefix of 1s
        L = 0
        for i in range(1, L_target + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Greedy Construction
    # Start with empty set
    current_classes = []
    current_budget = Fraction(1)
    
    # Sort primes by size? Smaller primes have larger 1/p cost but cover more numbers per step?
    # Actually, small primes cover "dense" residues.
    # Let's try adding primes one by one, choosing the residue that maximizes coverage gain.
    
    # To keep it simple and effective, we will do a randomized greedy:
    # 1. Start with empty.
    # 2. While budget allows:
    #    a. Pick a prime not yet used.
    #    b. Try all residues for that prime? No, too slow.
    #    c. Instead, pick a random prime, then pick the residue that covers the most "new" numbers in the current prefix?
    #    
    # Alternative: 
    # We know the seed solution covers ~999. 
    # We can try to improve by swapping.
    
    # Let's implement a simple randomized local search starting from a good heuristic.
    # Heuristic: Use smallest primes to cover initial numbers?
    # Actually, the seed uses primes starting from 13. 
    # 2, 3, 5, 7 are very cheap in budget (1/2+1/3+1/5+1/7 = 0.5+0.33+0.2+0.14 = 1.17 > 1).
    # So we can't use 2 and 3 together easily.
    
    # Let's try a greedy approach that builds the set.
    # Candidate pool: all primes <= 1000.
    # We will iterate. At each step, we pick a prime p and residue r.
    # To avoid O(N^2) complexity in Python, we will limit the search.
    
    # Strategy: 
    # 1. Initialize with a random valid set (e.g., the seed one).
    # 2. Perform random swaps/additions/removals to improve L.
    
    # Initialize with seed-like structure but randomize residues?
    # Or just start with the seed's primes and random residues?
    # The seed's primes are [13, 17, 19, ...].
    
    # Let's start with the seed's exact solution as a baseline.
    seed_primes = [p for p in primes if p >= 13]
    # The seed maps i-th prime to i+1.
    # p_i covers i+1.
    # Let's just start with the seed solution.
    
    current_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    # Check budget and L for seed
    sum_recip = sum(Fraction(1, p) for p, _ in current_classes)
    if sum_recip > 1:
        # This shouldn't happen for seed, but just in case
        pass
        
    best_L = calc_L(current_classes)
    best_classes = current_classes[:]
    
    # Local Search
    # We will try to replace a prime with a different one, or change a residue.
    # Changing a residue is cheap.
    # Replacing a prime changes the budget.
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # 1. Try changing a random residue
        if current_classes:
            idx = random.randint(0, len(current_classes) - 1)
            p, old_r = current_classes[idx]
            new_r = random.randint(0, p - 1)
            
            # Temporarily apply
            old_class = current_classes[idx]
            current_classes[idx] = [p, new_r]
            
            new_L = calc_L(current_classes)
            
            if new_L > best_L:
                best_L = new_L
                best_classes = current_classes[:]
            else:
                # Revert if not better (or keep if equal with some probability for diversity? 
                # For now, strict improvement to climb the hill)
                current_classes[idx] = old_class
                
        # 2. Try swapping a prime
        # Pick a prime in current set, remove it, add a different prime.
        if current_classes:
            idx = random.randint(0, len(current_classes) - 1)
            p_old, r_old = current_classes[idx]
            
            # Find a prime to swap in
            # We need to ensure budget constraint.
            # Current budget sum S.
            # New budget S - 1/p_old + 1/p_new <= 1.
            
            # Pick a random candidate prime from all primes
            p_new = random.choice(primes)
            
            # Avoid same prime
            if p_new == p_old:
                continue
                
            # Check if p_new is already in the set?
            # Distinct primes required.
            if any(p == p_new for p, _ in current_classes):
                continue
            
            # Check budget
            current_sum = sum(Fraction(1, p) for p, _ in current_classes)
            new_sum = current_sum - Fraction(1, p_old) + Fraction(1, p_new)
            
            if new_sum <= 1:
                # Apply swap
                current_classes[idx] = [p_new, r_old] # Keep same residue initially?
                # Residue r_old might be >= p_new. Need to adjust.
                r_new = r_old % p_new
                
                current_classes[idx] = [p_new, r_new]
                
                new_L = calc_L(current_classes)
                
                if new_L > best_L:
                    best_L = new_L
                    best_classes = current_classes[:]
                else:
                    # Revert
                    current_classes[idx] = [p_old, r_old]
        
        # 3. Try adding a prime (if budget allows)
        current_sum = sum(Fraction(1, p) for p, _ in current_classes)
        if current_sum < 1:
            # Find a prime that fits
            # Sort remaining primes by 1/p ascending (largest primes first) to fit easily?
            # Or random.
            # Let's pick a random prime not in set.
            available = [p for p in primes if not any(p == cp for cp, _ in current_classes)]
            if available:
                p_add = random.choice(available)
                cost = Fraction(1, p_add)
                if current_sum + cost <= 1:
                    # Add it. What residue?
                    # Pick residue to cover the first uncovered number?
                    # Find first gap in best_L?
                    # Let's just pick random residue first, then optimize.
                    r_add = random.randint(0, p_add - 1)
                    
                    current_classes.append([p_add, r_add])
                    
                    new_L = calc_L(current_classes)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_classes = current_classes[:]
                    else:
                        current_classes.pop()

        # 4. Try removing a prime (to free up budget for something else? 
        # Usually removing reduces L, so only do if we plan to add something better immediately.
        # For now, skip removals unless budget is tight and we want to swap large prime for two small ones?
        # Swapping handles 1-for-1.
        
        # Periodically restart if stuck? 
        # If no improvement in last 1000 iterations, maybe randomize current state?
        # For now, keep it simple.
        
    return best_classes

# To ensure we don't return an empty list if something goes wrong, 
# we can wrap in a try-except in the actual construct if needed, 
# but the logic above is robust.
