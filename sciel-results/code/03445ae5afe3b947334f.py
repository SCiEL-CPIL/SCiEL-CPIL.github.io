"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
import random
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
    
    # Seed solution: i-th prime from 13 upward covers integer i
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    def compute_coverage(classes, max_L):
        """Compute the largest L such that 1..L is fully covered."""
        covered = [False] * (max_L + 1)
        for p, r in classes:
            # Start from the first n >= 1 with n % p == r
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, max_L + 1, p):
                covered[n] = True
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    max_check = 20000
    seed_L = compute_coverage(seed_classes, max_check)
    
    best_L = seed_L
    best_classes = seed_classes[:]
    
    # Greedy construction targeting a specific L
    def greedy_build(target_L, seed_classes=None):
        """Build a covering of 1..target_L using greedy set cover."""
        if time.time() > deadline:
            return None
        
        # Start with seed if provided
        if seed_classes:
            classes = [c[:] for c in seed_classes]
        else:
            classes = []
        
        used_primes = set(p for p, r in classes)
        budget_remaining = Fraction(1, 1) - sum(Fraction(1, p) for p, r in classes)
        
        # Track coverage
        covered = [False] * (target_L + 1)
        for p, r in classes:
            start = p if r == 0 else r
            for n in range(start, target_L + 1, p):
                covered[n] = True
        
        # Find first uncovered
        def find_first_uncovered():
            for i in range(1, target_L + 1):
                if not covered[i]:
                    return i
            return None
        
        while True:
            if time.time() > deadline:
                break
            
            first_uncovered = find_first_uncovered()
            if first_uncovered is None:
                break  # All covered
            
            # Try to find the best (p, r) that covers first_uncovered
            best_score = -1
            best_p = None
            best_r = None
            best_new_count = 0
            
            for p in primes:
                if p in used_primes:
                    continue
                if Fraction(1, p) > budget_remaining:
                    continue
                
                # The residue that covers first_uncovered
                r = first_uncovered % p
                
                # Count how many currently uncovered integers this would cover
                new_count = 0
                start = p if r == 0 else r
                for n in range(start, target_L + 1, p):
                    if not covered[n]:
                        new_count += 1
                
                if new_count == 0:
                    continue
                
                # Score: new_coverage * p (value per unit cost)
                score = new_count * p
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
                    best_new_count = new_count
            
            if best_p is None:
                break
            
            # Add this class
            used_primes.add(best_p)
            classes.append([best_p, best_r])
            budget_remaining -= Fraction(1, best_p)
            
            # Update coverage
            start = best_p if best_r == 0 else best_r
            for n in range(start, target_L + 1, best_p):
                covered[n] = True
        
        return classes
    
    # Try greedy for various target L values
    for target in [best_L + 10, best_L + 50, best_L + 100, 1200, 1500, 2000, 3000, 5000]:
        if time.time() > deadline:
            break
        if target <= best_L:
            continue
        
        classes = greedy_build(target, seed_classes=best_classes)
        if classes is None:
            continue
        
        # Verify budget
        total = sum(Fraction(1, p) for p, r in classes)
        if total > 1:
            continue
        
        # Compute actual coverage
        L = compute_coverage(classes, max(target, best_L + 100))
        if L > best_L:
            best_L = L
            best_classes = classes[:]
    
    # Local improvement: try swapping or adding primes
    def try_local_improve(classes, current_L, max_iter=500):
        """Try to improve coverage by swapping primes."""
        if time.time() > deadline:
            return classes, current_L
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            
            # Try each class
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                old_p, old_r = classes[i]
                
                # Try replacing with a different prime
                for new_p in primes:
                    if new_p == old_p:
                        continue
                    if any(c[0] == new_p for c in classes):
                        continue
                    
                    # Check budget
                    old_cost = Fraction(1, old_p)
                    new_cost = Fraction(1, new_p)
                    if new_cost > old_cost:
                        total_new = sum(Fraction(1, c[0]) for c in classes) - old_cost + new_cost
                        if total_new > 1:
                            continue
                    
                    # Try all residues for new_p
                    best_new_r = None
                    best_L_new = current_L - 1  # Must be strictly better
                    
                    # Only try a subset of residues for speed
                    candidate_residues = list(range(new_p))
                    random.shuffle(candidate_residues)
                    for r in candidate_residues[:20]:  # Limit to 20 residues
                        new_classes = classes[:]
                        new_classes[i] = [new_p, r]
                        
                        L_new = compute_coverage(new_classes, max(current_L + 100, 2000))
                        if L_new > best_L_new:
                            best_L_new = L_new
                            best_new_r = r
                    
                    if best_new_r is not None and best_L_new > current_L:
                        classes[i] = [new_p, best_new_r]
                        current_L = best_L_new
                        improved = True
                        break  # Restart after improvement
        
        return classes, current_L
    
    # Run local improvement
    if time.time() < deadline - 10:
        best_classes, best_L = try_local_improve(best_classes[:], best_L, max_iter=200)
    
    # Try multiple random restarts
    if time.time() < deadline - 15:
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Create a random initial solution
            random_primes = primes[:]
            random.shuffle(random_primes)
            
            init_classes = []
            budget = Fraction(0)
            for p in random_primes:
                if budget + Fraction(1, p) <= 1:
                    r = random.randint(0, p - 1)
                    init_classes.append([p, r])
                    budget += Fraction(1, p)
                else:
                    break
            
            if not init_classes:
                continue
            
            init_L = compute_coverage(init_classes, 2000)
            
            # Run local search
            classes, L = try_local_improve(init_classes, init_L, max_iter=100)
            
            if L > best_L:
                best_L = L
                best_classes = classes
    
    # Final verification
    final_L = compute_coverage(best_classes, max(best_L + 100, 10000))
    
    # Verify budget
    total = sum(Fraction(1, p) for p, r in best_classes)
    if total > 1:
        # Fallback to seed
        return seed_classes
    
    return best_classes

# EVOLVE-BLOCK-END
