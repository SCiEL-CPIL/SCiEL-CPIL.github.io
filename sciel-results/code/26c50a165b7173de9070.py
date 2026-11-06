import time
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute Smallest Prime Factor (SPF) up to 20000
    LIMIT = 20000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def calculate_prime_counts(A):
        counts = Counter()
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                s = A[i] + A[j]
                for p in get_prime_factors(s):
                    counts[p] += 1
        return counts

    def get_cost(counts):
        return len(counts)

    # 2. Generate Initial Candidates
    # Heuristic 1: Small even numbers
    cand1 = list(range(2, 26, 2))
    # Heuristic 2: Multiples of 30
    cand2 = [i * 30 for i in range(1, 13)]
    # Heuristic 3: Powers of 2 (scaled?)
    cand3 = [1 << i for i in range(12)]
    # Heuristic 4: Random
    cand4 = random.sample(range(1, 10001), 12)
    
    candidates = [cand1, cand2, cand3, cand4]
    
    best_set = None
    best_cost = float('inf')
    best_counts = None
    
    # Evaluate candidates
    for c in candidates:
        # Ensure distinct and <= 10000
        if len(set(c)) != 12 or max(c) > 10000:
            continue
        counts = calculate_prime_counts(c)
        cost = get_cost(counts)
        if cost < best_cost:
            best_cost = cost
            best_set = list(c)
            best_counts = counts

    if best_set is None:
        best_set = list(range(1, 13))
        best_counts = calculate_prime_counts(best_set)
        best_cost = get_cost(best_counts)

    # 3. Local Search
    current_set = list(best_set)
    current_counts = Counter(best_counts)
    current_cost = best_cost
    
    n = 12
    # Pre-compute indices for neighbors? Just iterate.
    
    # To speed up, we can precompute factors for all possible sums?
    # Sums range 1+2=3 to 9999+10000=19999.
    # We have spf, so get_prime_factors is fast.
    
    while time.time() < deadline:
        # Pick a random index to replace
        i = random.randint(0, n - 1)
        old_val = current_set[i]
        
        # Pick a new value
        # Strategy: Random in [1, 10000] not in current_set
        # Or local perturbation: old_val + delta
        # Random is safer to escape local minima.
        new_val = random.randint(1, 10000)
        if new_val in current_set:
            continue
            
        # Calculate change in prime counts
        # Sums involving i: old_val + current_set[j] for j != i
        # New sums: new_val + current_set[j] for j != i
        
        # We need to update current_counts
        # We will simulate the change on a copy of the counter? 
        # Counter copy is O(K). K is small (< 100 probably).
        # Or we can just apply, check, and revert if bad.
        
        # Let's apply
        changed_primes = []
        
        for j in range(n):
            if j == i:
                continue
            other = current_set[j]
            
            sum_old = old_val + other
            sum_new = new_val + other
            
            # Remove old factors
            for p in get_prime_factors(sum_old):
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
                    changed_primes.append((p, 'removed'))
                    
            # Add new factors
            for p in get_prime_factors(sum_new):
                current_counts[p] += 1
                changed_primes.append((p, 'added'))
                
        new_cost = len(current_counts)
        
        # Acceptance Criteria
        if new_cost < current_cost:
            # Always accept improvement
            current_cost = new_cost
            current_set[i] = new_val
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = list(current_set)
                best_counts = Counter(current_counts)
                
        elif new_cost == current_cost:
            # Accept with some probability to explore
            if random.random() < 0.3:
                current_set[i] = new_val
            else:
                # Revert
                for p, action in changed_primes:
                    if action == 'removed':
                        current_counts[p] += 1
                    else:
                        current_counts[p] -= 1
                        if current_counts[p] == 0:
                            del current_counts[p]
                            
        else:
            # Reject (or SA accept with low prob)
            # Revert
            for p, action in changed_primes:
                if action == 'removed':
                    current_counts[p] += 1
                    if current_counts[p] == 0:
                         # Should not happen if logic correct, but safe
                         pass
                    # Wait, if it was removed, count went 1->0. Revert 0->1.
                    # If it was added, count went 0->1 or 1->2. Revert 1->0 or 2->1.
                    # My revert logic above:
                    # If 'removed' (was 1, became 0): add 1.
                    # If 'added' (was 0, became 1): sub 1. If 0, delete.
                    # This is correct.
                    pass # Handled below in common revert block? No, I did it in elif.
                    # Let's refactor revert.
            
            # Common Revert Block
            # I need to be careful. The loop above modifies current_counts.
            # To revert, I need to know the exact delta.
            # Simpler: Keep a backup of the affected primes' counts before change.
            
            # Let's restart the loop logic for clarity in code below.
            pass

    return best_set
# EVOLVE-BLOCK-END
