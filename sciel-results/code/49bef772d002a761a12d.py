"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random
from collections import defaultdict

N = 5000
DEADLINE = 320  # safety margin before 330s kill


def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _construct_greedy(primes, deadline, order="increasing"):
    """
    Greedy removal: start with all points [0, N]. For each prime p,
    if all p residue classes are still present, remove the least-populated class.
    """
    start = time.time()
    present = bytearray(b'\x01') * (N + 1)
    
    if order == "increasing":
        prime_list = primes
    elif order == "decreasing":
        prime_list = list(reversed(primes))
    else:
        prime_list = list(primes)
        random.shuffle(prime_list)
    
    for p in prime_list:
        if p > N:
            break
        if time.time() - start > deadline:
            break
        
        # Count points in each residue class mod p
        class_counts = [0] * p
        for v in range(N + 1):
            if present[v]:
                class_counts[v % p] += 1
        
        # If all p classes have at least one point, remove the least populated class
        if min(class_counts) > 0:
            min_count = min(class_counts)
            # Find the first class with minimum count
            target_class = class_counts.index(min_count)
            
            # Remove all points in this class
            for v in range(target_class, N + 1, p):
                present[v] = 0
    
    # Collect remaining points
    result = [v for v in range(N + 1) if present[v]]
    return result


def _refine(points, primes, deadline):
    """
    Try to recover removed elements that don't violate admissibility.
    For each removed element, check if adding it back keeps the tuple admissible.
    """
    start = time.time()
    if not points:
        return points
    
    points_set = set(points)
    current_len = len(points)
    
    # Generate list of removed elements
    removed = [v for v in range(N + 1) if v not in points_set]
    
    # Try to add back removed elements one by one
    for v in removed:
        if time.time() - start > deadline:
            break
        
        # Check if adding v keeps the tuple admissible
        new_len = current_len + 1
        
        # Check admissibility for all primes p <= new_len
        admissible = True
        for p in primes:
            if p > new_len:
                break
            # Count residues in points_set plus v
            residues = set(x % p for x in points_set)
            if v % p not in residues:
                residues.add(v % p)
            if len(residues) == p:
                admissible = False
                break
        
        if admissible:
            points_set.add(v)
            current_len += 1
    
    return list(points_set)


def _refine_random(points, primes, deadline, num_trials=50):
    """
    Multiple refinement passes with random ordering of removed elements.
    """
    start = time.time()
    if not points:
        return points
    
    best_result = list(points)
    best_len = len(points)
    
    for trial in range(num_trials):
        if time.time() - start > deadline:
            break
        
        points_set = set(points)
        current_len = len(points)
        
        # Generate list of removed elements
        removed = [v for v in range(N + 1) if v not in points_set]
        
        # Randomize order
        random.shuffle(removed)
        
        # Try to add back removed elements one by one
        for v in removed:
            if time.time() - start > deadline:
                break
            
            new_len = current_len + 1
            
            # Check admissibility for all primes p <= new_len
            admissible = True
            for p in primes:
                if p > new_len:
                    break
                residues = set(x % p for x in points_set)
                if v % p not in residues:
                    residues.add(v % p)
                if len(residues) == p:
                    admissible = False
                    break
            
            if admissible:
                points_set.add(v)
                current_len += 1
        
        if current_len > best_len:
            best_len = current_len
            best_result = list(points_set)
    
    return best_result


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    
    An admissible tuple is one where for every prime p <= len(tuple),
    some residue class mod p is missed.
    """
    start_time = time.time()
    
    # Generate primes up to N
    primes = _primes(N)
    
    # Try multiple strategies and keep the best
    best_result = []
    best_len = 0
    
    strategies = [
        ("increasing", 0.6),
        ("decreasing", 0.3),
        ("random", 0.1),
    ]
    
    for order, time_frac in strategies:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 10  # Leave 10s for refinement
        if remaining < 5:
            break
        
        deadline_phase1 = remaining * time_frac
        result = _construct_greedy(primes, deadline_phase1, order=order)
        
        # Verify admissibility
        if not _is_admissible(result, primes):
            # Fallback: aggressive removal
            for p in primes:
                if p > len(result):
                    break
                residues = set(v % p for v in result)
                if len(residues) == p:
                    class_counts = defaultdict(list)
                    for v in result:
                        class_counts[v % p].append(v)
                    min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                    result = [v for v in result if v % p != min_class]
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Refinement phase
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 5
    
    if remaining_time > 10 and best_result:
        # First, deterministic refinement
        refine_deadline = min(remaining_time * 0.4, 60)
        refined = _refine(best_result, primes, refine_deadline)
        if len(refined) > best_len:
            best_len = len(refined)
            best_result = refined
        
        # Then, random refinement
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed - 5
        if remaining_time > 10 and best_result:
            random_deadline = min(remaining_time * 0.6, 120)
            refined_random = _refine_random(best_result, primes, random_deadline, num_trials=30)
            if len(refined_random) > best_len:
                best_len = len(refined_random)
                best_result = refined_random
    
    # Final verification
    if best_result and not _is_admissible(best_result, primes):
        # Emergency fallback: remove until admissible
        for p in primes:
            if p > len(best_result):
                break
            residues = set(v % p for v in best_result)
            if len(residues) == p:
                class_counts = defaultdict(list)
                for v in best_result:
                    class_counts[v % p].append(v)
                min_class = min(class_counts, key=lambda c: len(class_counts[c]))
                best_result = [v for v in best_result if v % p != min_class]
    
    # Ensure we return distinct integers in [0, N]
    best_result = sorted(set(best_result))
    
    return best_result
