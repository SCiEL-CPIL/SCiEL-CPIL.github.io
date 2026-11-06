import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 320.0  # seconds

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _is_admissible(points, primes):
    """Check if points form an admissible tuple w.r.t. given primes."""
    n = len(points)
    for p in primes:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _fast_local_search(points, primes, n_max=N, max_iters=20000, deadline=None):
    """
    Fast local search using dynamic residue counts.
    """
    points_set = set(points)
    best = list(points)
    best_size = len(best)
    
    # Precompute relevant primes
    relevant_primes = [p for p in primes if p <= n_max + 1]
    
    # Initialize counts for current set
    # counts[p] is a list of size p, counts[p][r] is count of x in S s.t. x % p == r
    # full_count[p] is number of r with counts[p][r] > 0
    counts = {}
    full_counts = {}
    for p in relevant_primes:
        c = [0] * p
        for v in points_set:
            c[v % p] += 1
        counts[p] = c
        full_counts[p] = sum(1 for x in c if x > 0)
        
    if deadline is not None:
        start_time = time.time()
    
    candidates = [v for v in range(n_max + 1) if v not in points_set]
    random.shuffle(candidates)
    
    for candidate in candidates:
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Try to add candidate
        # Check if adding candidate makes the set inadmissible
        # New size K' = len(points_set) + 1
        new_size = len(points_set) + 1
        is_valid = True
        
        # For p <= new_size, check if all classes are covered after adding
        # We only need to check primes p <= new_size
        # Note: if p > new_size, condition is not checked.
        
        # Optimization: 
        # For p < new_size: 
        #   If full_counts[p] == p, it's already bad (shouldn't happen if current is valid).
        #   If full_counts[p] == p-1, and candidate fills the missing class, it becomes p (bad).
        #   If full_counts[p] < p-1, adding one element can at most increase full_count by 1.
        #      So if full_counts[p] <= p-2, it stays <= p-1, so it's still "missed" at least one class. OK.
        # For p == new_size:
        #   We need full_counts[p] + (1 if counts[p][candidate%p]==0 else 0) < p.
        #   Since p = new_size = |S|+1, and |S|=p-1, max distinct residues in S is p-1.
        #   So full_counts[p] <= p-1.
        #   If full_counts[p] == p-1, and candidate fills the missing one, full_count becomes p (bad).
        #   If full_counts[p] < p-1, it stays < p (good).

        for p in relevant_primes:
            if p > new_size:
                break
            
            r = candidate % p
            current_full = full_counts[p]
            
            if p < new_size:
                # Current set is admissible, so current_full < p.
                # If current_full == p - 1, we are on the edge.
                # If counts[p][r] == 0, adding candidate makes full_count p. Bad.
                if current_full == p - 1 and counts[p][r] == 0:
                    is_valid = False
                    break
            else: # p == new_size
                # Current set size is p-1.
                # Max full_count is p-1.
                # If full_counts[p] == p - 1 and counts[p][r] == 0, adding makes it p. Bad.
                if full_counts[p] == p - 1 and counts[p][r] == 0:
                    is_valid = False
                    break
        
        if is_valid:
            # Commit addition
            points_set.add(candidate)
            for p in relevant_primes:
                if p > new_size:
                    break
                r = candidate % p
                if counts[p][r] == 0:
                    full_counts[p] += 1
                counts[p][r] += 1
            
            best = sorted(points_set)
            best_size = new_size
    
    return best


def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            min_count = min(class_counts.values())
            remove_class = min(class_counts.keys(), key=lambda r: class_counts[r])
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    An admissible tuple satisfies: for every prime p <= len(tuple),
    at least one residue class mod p is missed.
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    # First, get a baseline greedy solution
    baseline = _greedy_construct(primes, N)
    best = baseline
    best_size = len(best)
    
    # Use fast local search to improve
    deadline = start_time + DEADLINE
    if time.time() < deadline - 10:
        local_result = _fast_local_search(
            best, primes, N,
            max_iters=20000,
            deadline=deadline
        )
        if len(local_result) > best_size:
            best = local_result
            best_size = len(best)
            
    # Try a few random restarts with fast local search
    for i in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random perturbation of the greedy set
        perturbed = best[:]
        # Remove a few random elements
        for _ in range(random.randint(5, 20)):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Run fast local search on perturbed set
        local_result = _fast_local_search(
            perturbed, primes, N,
            max_iters=10000,
            deadline=deadline
        )
        
        if len(local_result) > best_size:
            best = local_result
            best_size = len(best)
            
    # Final verification
    n = len(best)
    small_primes = [p for p in primes if p <= n]
    for p in small_primes:
        residues = set(v % p for v in best)
        if len(residues) == p:
            # Not admissible! Remove from the smallest class
            class_counts = Counter(v % p for v in best)
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = min_classes[0]
            best = [v for v in best if v % p != remove_class]
            
    return best

# EVOLVE-BLOCK-END
