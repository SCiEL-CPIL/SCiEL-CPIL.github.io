"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    prime_set = set(primes)
    
    # Target: try to cover as much as possible
    # Start with the seed solution as baseline
    # Seed: i-th prime from 13 upward covers integer i
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_sum = sum(Fraction(1, p) for p in seed_primes)
    
    # Compute seed coverage
    def compute_coverage(classes, max_L):
        covered = [False] * (max_L + 1)
        for p, r in classes:
            for n in range(1, max_L + 1):
                if n % p == r:
                    covered[n] = True
        # Find largest L such that 1..L all covered
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L, covered
    
    max_check = 5000
    seed_L, _ = compute_coverage(seed_classes, max_check)
    
    best_L = seed_L
    best_classes = seed_classes[:]
    
    # Greedy construction: build coverage from 1 upward
    # State: set of used primes, coverage bitset
    # For each uncovered position, try all (p, r) where r = pos % p
    # Cost: 1/p, benefit: number of newly covered positions
    
    def greedy_build(target_L):
        """Build a covering of 1..target_L using greedy set cover."""
        # Track which primes are used
        used_primes = set()
        classes = []
        # Coverage as a set of covered integers
        covered = set()
        
        # We need to cover 1, 2, 3, ..., target_L
        # Greedy: at each step, pick the (p,r) that covers the most uncovered integers
        # subject to p not in used_primes and sum(1/p) <= 1
        
        budget_remaining = Fraction(1, 1)
        uncovered = set(range(1, target_L + 1))
        
        # Precompute for each prime p, the residue classes and their coverage
        # For efficiency, process primes in order of 1/p (smallest first = largest p first? No, we want best coverage per cost)
        
        # Actually, let's think about this more carefully.
        # For a given prime p, residue r, it covers {r, r+p, r+2p, ...} within [1, target_L]
        # The number of covered integers is approximately target_L / p
        
        # Greedy: repeatedly pick the (p, r) that maximizes new_coverage / (1/p) = new_coverage * p
        # subject to p not used and budget
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_score = -1
            best_p = None
            best_r = None
            best_new_cover = set()
            
            # For each unused prime, try all residues
            for p in primes:
                if p in used_primes:
                    continue
                if Fraction(1, p) > budget_remaining:
                    continue
                
                # For each residue r in 0..p-1
                # But we only need to consider residues that cover at least one uncovered integer
                # The uncovered integers mod p give candidate residues
                candidate_residues = set()
                for n in list(uncovered)[:min(len(uncovered), 50)]:  # Limit for speed
                    candidate_residues.add(n % p)
                
                for r in candidate_residues:
                    # Compute which uncovered integers are covered by (p, r)
                    new_cover = set()
                    # Integers n in uncovered such that n % p == r
                    # n = r + k*p for k >= 0, n >= 1
                    start = r
                    if start < 1:
                        start = r + p * ((1 - r + p - 1) // p) if r > 0 else p
                    # More carefully: smallest n >= 1 with n % p == r
                    if r == 0:
                        start = p
                    else:
                        start = r
                    if start < 1:
                        start += p * ((1 - start + p - 1) // p)
                    
                    for n in range(start, target_L + 1, p):
                        if n in uncovered:
                            new_cover.add(n)
                    
                    if not new_cover:
                        continue
                    
                    # Score: new_coverage * p (since cost is 1/p, value per cost = new_cov * p)
                    score = len(new_cover) * p
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_r = r
                        best_new_cover = new_cover
            
            if best_p is None:
                break
            
            # Add this class
            used_primes.add(best_p)
            classes.append([best_p, best_r])
            budget_remaining -= Fraction(1, best_p)
            covered |= best_new_cover
            uncovered -= best_new_cover
        
        return classes, covered
    
    # Try greedy for various target L values
    for target in [best_L + 1, best_L + 5, best_L + 20, best_L + 50, 1500, 2000, 3000]:
        if time.time() > deadline:
            break
        if target <= best_L:
            continue
        
        classes, covered = greedy_build(target)
        if not classes:
            continue
        
        # Verify budget
        total = sum(Fraction(1, p) for p, r in classes)
        if total > 1:
            continue
        
        # Compute actual coverage
        L, _ = compute_coverage(classes, max(target, best_L + 100))
        if L > best_L:
            best_L = L
            best_classes = classes[:]
    
    # Local improvement: try swapping primes
    if time.time() < deadline:
        # For each class, try replacing with a different prime
        for i in range(len(best_classes)):
            if time.time() > deadline:
                break
            old_p, old_r = best_classes[i]
            # Try replacing with other primes
            for new_p in primes:
                if new_p == old_p:
                    continue
                # Check if new_p is already used
                if any(c[0] == new_p for c in best_classes):
                    continue
                
                # Try all residues for new_p
                # We want to maintain or improve coverage
                # The old class covered some integers; the new class should cover at least those
                # plus potentially more
                
                # Find which integers the old class covered in [1, best_L + 100]
                old_covered = set()
                for n in range(1, best_L + 101):
                    if n % old_p == old_r:
                        old_covered.add(n)
                
                best_new_r = None
                best_extra = -1
                
                for r in range(new_p):
                    new_covered = set()
                    for n in range(1, best_L + 101):
                        if n % new_p == r:
                            new_covered.add(n)
                    
                    # Check: does new_covered cover all of old_covered?
                    # Actually, we just need the overall coverage to be maintained
                    # This is complex; skip for now and just try to improve
                    
                    extra = len(new_covered - old_covered) - len(old_covered - new_covered)
                    if extra > best_extra:
                        best_extra = extra
                        best_new_r = r
                
                if best_new_r is not None and best_extra > 0:
                    # Check budget
                    old_budget = Fraction(1, old_p)
                    new_budget = Fraction(1, new_p)
                    if new_budget <= old_budget:  # Doesn't increase budget
                        # Try this swap
                        new_classes = best_classes[:]
                        new_classes[i] = [new_p, best_new_r]
                        total = sum(Fraction(1, p) for p, r in new_classes)
                        if total <= 1:
                            L, _ = compute_coverage(new_classes, best_L + 100)
                            if L > best_L:
                                best_L = L
                                best_classes = new_classes
    
    # Final verification
    final_L, _ = compute_coverage(best_classes, max(best_L + 100, 10000))
    return best_classes

# EVOLVE-BLOCK-END
