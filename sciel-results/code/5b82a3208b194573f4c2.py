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
    
    def compute_coverage(classes, max_check=100000):
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

    def greedy_residues(prime_list, max_check=100000):
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
                # A simple heuristic: maximize the count of new numbers covered
                # that are in the range first_uncovered..first_uncovered+buffer
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

    # Strategy: select primes greedily by trying to maximize coverage
    # We'll use a priority-based approach: try adding primes one by one
    
    # First, let's find which primes we can afford
    # Try different subsets and evaluate
    
    # Greedy prime selection: start with empty, add prime that gives best coverage improvement
    # This is expensive, so let's use a simpler approach.
    
    # Approach: select primes in order of increasing p (smallest first gives densest coverage)
    # but constrained by budget.
    
    # Let's try: sort primes by p (ascending), greedily add if budget allows
    # But also try: sort by p descending (cheapest first)
    
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
        classes = greedy_residues(selected, max_check=100000)
        L = compute_coverage(classes, max_check=100000)
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
        classes = greedy_residues(selected, max_check=100000)
        L = compute_coverage(classes, max_check=100000)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Try strategy 3: mixed - use small primes for density, fill with large
    # Start with a few small primes, then add large ones
    if time.time() < deadline - 5:
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
                classes = greedy_residues(selected, max_check=100000)
                L = compute_coverage(classes, max_check=100000)
                if L > best_L:
                    best_L = L
                    best_classes = classes
            
            if time.time() > deadline:
                break
    
    # Try strategy 4: use the i-th prime from 13 upward covers the integer i
    # This is a simple constructive approach that guarantees coverage
    if time.time() < deadline - 5:
        primes_from_13 = [p for p in primes if p >= 13]
        # We can use as many as we can afford
        budget = Fraction(1)
        selected_from_13 = []
        for p in primes_from_13:
            cost = Fraction(1, p)
            if budget >= cost:
                budget -= cost
                selected_from_13.append(p)
        
        if selected_from_13:
            # Assign the i-th prime to cover integer i+1
            classes = []
            for i, p in enumerate(selected_from_13):
                r = (i + 1) % p
                classes.append([p, r])
            
            L = compute_coverage(classes, max_check=100000)
            if L > best_L:
                best_L = L
                best_classes = classes
    
    return best_classes if best_classes else [[2, 1]]

# EVOLVE-BLOCK-END
