"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    primes = _sieve(1000)
    
    def compute_coverage(classes, max_check=200000):
        """Compute the largest L such that 1..L are all covered."""
        if not classes:
            return 0
        covered = [False] * (max_check + 1)
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            for n in range(start, max_check + 1, p):
                covered[n] = True
        
        L = 0
        for i in range(1, max_check + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def greedy_residues(prime_list, max_check=200000):
        """Assign residues greedily to maximize consecutive coverage from 1."""
        if not prime_list:
            return []
        
        covered = [False] * (max_check + 1)
        classes = []
        
        for p in prime_list:
            # Find the first uncovered number
            first_uncovered = None
            for i in range(1, max_check + 1):
                if not covered[i]:
                    first_uncovered = i
                    break
            
            if first_uncovered is None:
                break
            
            # Try all residues and pick the one that maximizes new coverage
            # near the first uncovered number
            best_r = first_uncovered % p
            best_new = -1
            best_covered_count = -1
            
            for r in range(p):
                # Compute which numbers in 1..max_check are covered by (p, r)
                new_covered_count = 0
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                for n in range(start, max_check + 1, p):
                    if not covered[n]:
                        new_covered_count += 1
                
                # Score: prioritize covering numbers near first_uncovered
                score = 0
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                for n in range(start, max_check + 1, p):
                    if not covered[n]:
                        if n == first_uncovered:
                            score += 10000
                        elif n <= first_uncovered + 100:
                            score += max(0, 100 - (n - first_uncovered))
                        else:
                            score += 1
                
                if score > best_new or (score == best_new and new_covered_count > best_covered_count):
                    best_new = score
                    best_r = r
                    best_covered_count = new_covered_count
            
            classes.append([p, best_r])
            # Update covered array
            if best_r == 0:
                start = p
            else:
                start = best_r
            if start < 1:
                start += p
            for n in range(start, max_check + 1, p):
                covered[n] = True
            
            if time.time() > deadline:
                break
        
        return classes

    def local_search(classes, max_check=200000):
        """Try to improve coverage by changing residues of existing classes."""
        if not classes:
            return classes, 0
        
        covered = [False] * (max_check + 1)
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            for n in range(start, max_check + 1, p):
                covered[n] = True
        
        current_L = compute_coverage(classes, max_check)
        best_L = current_L
        best_classes = [c[:] for c in classes]
        
        # Try changing each residue
        for idx, (p, r) in enumerate(classes):
            if time.time() > deadline:
                break
            
            # Remove this class from coverage
            old_covered = [False] * (max_check + 1)
            for pp, rr in classes:
                if pp == p and rr == r:
                    if rr == 0:
                        start = pp
                    else:
                        start = rr
                    if start < 1:
                        start += pp
                    for n in range(start, max_check + 1, pp):
                        old_covered[n] = True
            
            # Try all other residues
            for new_r in range(p):
                if new_r == r:
                    continue
                
                # Temporarily set to new_r
                classes[idx][1] = new_r
                
                # Recompute coverage
                new_covered = [False] * (max_check + 1)
                for pp, rr in classes:
                    if rr == 0:
                        start = pp
                    else:
                        start = rr
                    if start < 1:
                        start += pp
                    for n in range(start, max_check + 1, pp):
                        new_covered[n] = True
                
                new_L = compute_coverage(classes, max_check)
                
                if new_L > best_L:
                    best_L = new_L
                    best_classes = [c[:] for c in classes]
                
                # Restore
                classes[idx][1] = r
            
            if time.time() > deadline:
                break
        
        return best_classes, best_L

    # Strategy: select primes greedily by trying to maximize coverage
    best_classes = []
    best_L = 0
    
    # Try strategy 1: smallest primes first (densest)
    primes_asc = sorted(primes)
    budget = Fraction(1)
    selected = []
    for p in primes_asc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    
    if selected:
        classes = greedy_residues(selected, max_check=200000)
        L = compute_coverage(classes, max_check=200000)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Try strategy 2: largest primes first (cheapest, most primes)
    primes_desc = sorted(primes, reverse=True)
    budget = Fraction(1)
    selected = []
    for p in primes_desc:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected.append(p)
        else:
            break
    
    if selected:
        classes = greedy_residues(selected, max_check=200000)
        L = compute_coverage(classes, max_check=200000)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Try strategy 3: mixed - use small primes for density, fill with large
    if time.time() < deadline - 10:
        for num_small in range(1, 6):
            small_primes = primes_asc[:num_small]
            budget = Fraction(1)
            for p in small_primes:
                budget -= Fraction(1, p)
            if budget < 0:
                break
            
            selected = list(small_primes)
            for p in primes_desc:
                if p in small_primes:
                    continue
                cost = Fraction(1, p)
                if budget >= cost:
                    budget -= cost
                    selected.append(p)
            
            if selected:
                classes = greedy_residues(selected, max_check=200000)
                L = compute_coverage(classes, max_check=200000)
                if L > best_L:
                    best_L = L
                    best_classes = classes
            
            if time.time() > deadline:
                break
    
    # Local search on the best solution found
    if best_classes and time.time() < deadline - 5:
        improved_classes, improved_L = local_search(best_classes, max_check=200000)
        if improved_L > best_L:
            best_L = improved_L
            best_classes = improved_classes
    
    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
