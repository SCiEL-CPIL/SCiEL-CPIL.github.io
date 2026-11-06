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


def _count_full_classes(points, primes, n):
    """Count how many primes p <= n have all p residue classes covered."""
    full_count = 0
    for p in primes:
        if p > n:
            break
        residues = set()
        for v in points:
            r = v % p
            residues.add(r)
            if len(residues) == p:
                full_count += 1
                break
        else:
            if len(residues) == p:
                full_count += 1
    return full_count


def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    """
    points = set(range(n_max + 1))
    # Sort primes by size so we handle small primes first (more constrained)
    small_primes = [p for p in primes if p <= n_max + 1]
    
    for p in small_primes:
        if p > len(points):
            break
        # Check if all p residue classes are covered
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            # Remove the least-populated residue class
            min_count = min(class_counts.values())
            # Find which class has min count
            remove_class = min(class_counts.keys(), key=lambda r: class_counts[r])
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def _local_search(points, primes, n_max=N, max_iters=5000, deadline=None):
    """
    Local search: try to add back removed elements or swap to increase size.
    Returns improved list if found.
    """
    points_set = set(points)
    best = list(points)
    best_size = len(best)
    small_primes = [p for p in primes if p <= n_max + 1]
    
    if deadline is not None:
        start_time = time.time()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try adding a random element from [0, n_max] not in points_set
        candidate = random.randint(0, n_max)
        if candidate in points_set:
            continue
        
        trial = points_set | {candidate}
        if _is_admissible(trial, small_primes):
            points_set = trial
            best = sorted(points_set)
            best_size = len(best)
    
    return best


def _multi_start_greedy(primes, n_max=N, deadline=None):
    """
    Multi-start greedy with different random removals to find larger admissible sets.
    """
    best = None
    best_size = 0
    small_primes = [p for p in primes if p <= n_max + 1]
    
    if deadline is not None:
        start_time = time.time()
    
    # Start with the standard greedy
    greedy_result = _greedy_construct(primes, n_max)
    best = greedy_result
    best_size = len(best)
    
    if deadline is not None and time.time() > deadline - 10:
        return best
    
    # Try multiple random initial removals
    for attempt in range(100):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Start with full set, but randomly remove a few elements first
        points = set(range(n_max + 1))
        # Randomly remove some elements to break symmetry
        for _ in range(random.randint(1, 5)):
            if points:
                to_remove = random.choice(list(points))
                points.discard(to_remove)
        
        # Now run greedy repair
        for p in small_primes:
            if p > len(points):
                break
            class_counts = Counter(v % p for v in points)
            if len(class_counts) == p:
                min_count = min(class_counts.values())
                # Randomly choose among min-count classes
                min_classes = [r for r, c in class_counts.items() if c == min_count]
                remove_class = random.choice(min_classes)
                points = {v for v in points if v % p != remove_class}
        
        if len(points) > best_size:
            best = sorted(points)
            best_size = len(points)
    
    return best


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
    
    # Try multi-start greedy with randomization
    deadline = start_time + DEADLINE
    if time.time() < deadline - 30:
        multi_result = _multi_start_greedy(primes, N, deadline=deadline)
        if len(multi_result) > best_size:
            best = multi_result
            best_size = len(best)
    
    # Local search to try to add elements back
    if time.time() < deadline - 20:
        local_result = _local_search(
            best, primes, N,
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
