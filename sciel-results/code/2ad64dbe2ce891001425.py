"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_N = 256

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

# Precompute conflicts: for each x, set of pairs (i, j) with i < j, i != x, j != x
# such that x conflicts with (i, j)
# Conflict if (i*j)%x==0 or (x*j)%i==0 or (x*i)%j==0
def _build_conflicts():
    conflicts = [set() for _ in range(_N + 1)]
    for x in range(2, _N + 1):
        for i in range(2, _N + 1):
            if i == x:
                continue
            for j in range(i + 1, _N + 1):
                if j == x:
                    continue
                if (i * j) % x == 0 or (x * j) % i == 0 or (x * i) % j == 0:
                    conflicts[x].add((i, j))
    return conflicts

_CONFLICTS = _build_conflicts()

def _num_distinct_prime_factors(n):
    """Count distinct prime factors of n."""
    count = 0
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count += 1
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        count += 1
    return count

def _largest_prime_factor(n):
    """Find the largest prime factor of n."""
    largest = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            largest = d
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        largest = temp
    return largest

def _greedy_build(ordering, deadline):
    """Build initial set using greedy approach with given ordering."""
    S = []
    S_set = set()
    S_pairs = set() # Set of all pairs (a,b) with a<b in S
    candidates = list(ordering)
    
    while time.time() < deadline and candidates:
        valid = []
        for x in candidates:
            if x in S_set:
                continue
            # Check if x conflicts with any pair in S
            # x conflicts if any pair in S_pairs is in _CONFLICTS[x]
            if not (_CONFLICTS[x] & S_pairs):
                valid.append(x)
        
        if not valid:
            break
        
        best_x = valid[0]
        S.append(best_x)
        S_set.add(best_x)
        candidates.remove(best_x)
        
        # Update S_pairs
        for a in S:
            if a < best_x:
                S_pairs.add((a, best_x))
            elif a > best_x:
                S_pairs.add((best_x, a))
            
    return S, S_set, S_pairs

def _local_search_sa(S, S_set, S_pairs, deadline, iterations=20000):
    """
    Simulated Annealing / Hill Climbing local search.
    Tries to maximize len(S) while maintaining validity.
    Moves: 
    1. Add a valid element.
    2. Remove an element and try to add another (swap).
    3. Perturbation: Remove k random elements, then greedily refill.
    """
    S = list(S)
    S_set = set(S)
    S_pairs = set(S_pairs)
    current_size = len(S)
    best_S = list(S)
    best_S_set = set(S)
    best_S_pairs = set(S_pairs)
    
    all_numbers = list(range(2, _N + 1))
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
            
        # 1. Try to add any valid element
        improved = False
        # Randomly pick a few candidates to try adding
        if S_set != set(all_numbers):
            candidates = [x for x in all_numbers if x not in S_set]
            random.shuffle(candidates)
            # Try first 10 random candidates
            for x in candidates[:10]:
                if not (_CONFLICTS[x] & S_pairs):
                    # Add x
                    S.append(x)
                    S_set.add(x)
                    for a in S:
                        if a < x:
                            S_pairs.add((a, x))
                        elif a > x:
                            S_pairs.add((x, a))
                    current_size += 1
                    improved = True
                    break
        
        if improved:
            if current_size > len(best_S):
                best_S = list(S)
                best_S_set = set(S_set)
                best_S_pairs = set(S_pairs)
            continue
        
        # 2. If no addition possible, try a swap (remove one, add another)
        # Or try a perturbation: remove 2-3 elements and refill
        # Let's try a simple swap first
        if S:
            remove_idx = random.randint(0, len(S) - 1)
            x_to_remove = S[remove_idx]
            
            # Temporarily remove
            S.remove(x_to_remove)
            S_set.remove(x_to_remove)
            # Remove pairs involving x_to_remove
            pairs_to_remove = []
            for p in S_pairs:
                if x_to_remove in p:
                    pairs_to_remove.append(p)
            for p in pairs_to_remove:
                S_pairs.remove(p)
            
            old_size = current_size
            current_size -= 1
            
            # Try to add a random element
            candidates = [x for x in all_numbers if x not in S_set]
            random.shuffle(candidates)
            added = False
            for x in candidates[:20]:
                if not (_CONFLICTS[x] & S_pairs):
                    S.append(x)
                    S_set.add(x)
                    for a in S:
                        if a < x:
                            S_pairs.add((a, x))
                        elif a > x:
                            S_pairs.add((x, a))
                    current_size += 1
                    added = True
                    break
            
            if not added:
                # Revert if nothing was added? Or keep the removal if it allows future additions?
                # For simplicity, revert if size didn't increase
                if current_size < old_size:
                    # Re-add x_to_remove
                    S.append(x_to_remove)
                    S_set.add(x_to_remove)
                    for a in S:
                        if a < x_to_remove:
                            S_pairs.add((a, x_to_remove))
                        elif a > x_to_remove:
                            S_pairs.add((x_to_remove, a))
                    current_size = old_size
            else:
                if current_size > len(best_S):
                    best_S = list(S)
                    best_S_set = set(S_set)
                    best_S_pairs = set(S_pairs)
                    
        # 3. Perturbation: If stuck for a while, do a random restart from a perturbed state
        # We don't track "stuck" easily here, so we just rely on the loop.
        # If current_size is significantly smaller than best, we might want to reset to best?
        # Let's just keep the best found so far.
        
    return best_S, best_S_set, best_S_pairs

def _diversified_search(deadline):
    """Try multiple diversified greedy orderings and keep the best result."""
    best_S = []
    best_S_set = set()
    best_S_pairs = set()
    
    all_numbers = list(range(2, _N + 1))
    
    orderings = []
    
    # Ordering 1: Primes first
    primes = [a for a in all_numbers if _is_prime(a)]
    composites = [a for a in all_numbers if not _is_prime(a)]
    orderings.append(primes + composites)
    
    # Ordering 2: Random
    random_order = list(all_numbers)
    random.shuffle(random_order)
    orderings.append(random_order)
    
    # Ordering 3: Few distinct prime factors first
    orderings.append(sorted(all_numbers, key=lambda x: (_num_distinct_prime_factors(x), x)))
    
    # Ordering 4: Large numbers first
    orderings.append(sorted(all_numbers, reverse=True))
    
    # Ordering 5: Large prime factor first
    orderings.append(sorted(all_numbers, key=lambda x: -_largest_prime_factor(x)))
    
    # Ordering 6: Another random
    random_order2 = list(all_numbers)
    random.shuffle(random_order2)
    orderings.append(random_order2)
    
    # Time budget per ordering
    num_orderings = len(orderings)
    time_per_ordering = (deadline - time.time()) / num_orderings
    
    for i, ordering in enumerate(orderings):
        if time.time() > deadline:
            break
        
        # Allocate time: 20% for greedy, 80% for local search
        greedy_deadline = time.time() + time_per_ordering * 0.2
        ls_deadline = time.time() + time_per_ordering * 0.8
        
        S, S_set, S_pairs = _greedy_build(ordering, greedy_deadline)
        S, S_set, S_pairs = _local_search_sa(S, S_set, S_pairs, ls_deadline)
        
        if len(S) > len(best_S):
            best_S = S
            best_S_set = S_set
            best_S_pairs = S_pairs
    
    return best_S, best_S_set, best_S_pairs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Use diversified search
    S, S_set, S_pairs = _diversified_search(deadline)
    
    # Final local search polish with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        S, S_set, S_pairs = _local_search_sa(S, S_set, S_pairs, deadline)
    
    # Return as list of Python ints
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
