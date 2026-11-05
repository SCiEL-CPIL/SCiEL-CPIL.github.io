"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000
DEADLINE = 320  # seconds; keep 10s safety margin under 330s kill

# EVOLVE-BLOCK-START

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_list):
    """Check if points is admissible with respect to primes_list."""
    point_set = set(points)
    for p in primes_list:
        residues = set(v % p for v in point_set)
        if len(residues) == p:
            return False
    return True


def _greedy_removal(points, primes_list):
    """
    Greedy removal: while not admissible, remove the element that is in the
    smallest residue class among the problematic primes.
    Returns the resulting admissible list.
    """
    point_set = set(points)
    # We need to handle primes p <= len(points). But len changes as we remove.
    # Actually the condition is: for every prime p <= len(tuple), some residue class mod p is missed.
    # As we remove elements, len decreases, so the set of relevant primes shrinks.
    # But it's simpler to just check all primes p <= initial_len, and as len shrinks,
    # the condition for larger p becomes vacuously easier (we need p <= len(tuple)).
    # Wait, the condition is: for every prime p <= len(tuple), the tuple does NOT cover all residues mod p.
    # So if len(tuple) < p, the condition is automatically satisfied.
    # Let's just iterate and check.
    
    # We'll maintain the list and a set for O(1) removal.
    # For efficiency, let's use a list and a set.
    # Since N is only 5000, even O(N^2) is fine.
    
    current = list(points)
    current_set = set(current)
    
    while True:
        # Find the smallest prime p <= len(current) such that all residues mod p are covered
        problem_prime = None
        for p in primes_list:
            if p > len(current):
                break
            residues = set(v % p for v in current)
            if len(residues) == p:
                problem_prime = p
                break
        
        if problem_prime is None:
            break
        
        # For this problem_prime, find the residue class with the fewest elements
        class_counts = {}
        for v in current:
            r = v % problem_prime
            class_counts[r] = class_counts.get(r, 0) + 1
        
        # Remove the smallest class
        min_class = min(class_counts, key=class_counts.get)
        # Remove all elements in that class
        to_remove = [v for v in current if v % problem_prime == min_class]
        for v in to_remove:
            current_set.remove(v)
        current = [v for v in current if v not in to_remove]
    
    return current


def _try_insertion(current, primes_list, candidates, deadline):
    """
    Try to insert candidates one by one (in random order) to maximize size.
    Returns the largest admissible set found.
    """
    best = list(current)
    best_set = set(best)
    
    # Shuffle candidates
    shuffled = list(candidates)
    random.shuffle(shuffled)
    
    for v in shuffled:
        if time.time() > deadline:
            break
        if v in best_set:
            continue
        # Try adding v
        new_set = best_set | {v}
        new_list = list(new_set)
        # Check admissibility
        admissible = True
        for p in primes_list:
            if p > len(new_list):
                break
            residues = set(x % p for x in new_list)
            if len(residues) == p:
                admissible = False
                break
        if admissible:
            best = new_list
            best_set = new_set
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Primes up to N (5000)
    all_primes = _primes(N)
    
    # Strategy:
    # 1. Start with full set [0, N]
    # 2. Greedy removal to get an admissible set
    # 3. Try to add back removed elements in random order
    
    # Step 1: Full set
    full_set = list(range(N + 1))
    
    # Step 2: Greedy removal
    # For greedy removal, we need primes up to len(full_set) = N+1
    # But we only have primes up to N. Let's get primes up to N+1.
    # Actually, primes up to N is fine since N+1 might not be prime.
    # Let's be safe and get primes up to N+1.
    primes_for_removal = _primes(N + 1)
    
    admissible = _greedy_removal(full_set, primes_for_removal)
    admissible_set = set(admissible)
    
    # Step 3: Try to add back removed elements
    removed = [v for v in range(N + 1) if v not in admissible_set]
    
    # We want to add back elements. The primes we need to check are up to len(current).
    # As we add elements, len increases, so we might need to check larger primes.
    # Let's use all_primes (up to N) and check p <= len(current).
    
    best = _try_insertion(admissible, all_primes, removed, deadline)
    
    # If we have time, try a few more random restarts
    # But the greedy removal is deterministic, so restarts would need different initial removals.
    # Let's try a few random permutations of the removal order.
    
    while time.time() < deadline - 10:  # Keep 10s buffer
        # Randomized greedy removal
        current = list(range(N + 1))
        current_set = set(current)
        
        while True:
            if time.time() > deadline:
                break
            problem_prime = None
            for p in primes_for_removal:
                if p > len(current):
                    break
                residues = set(v % p for v in current)
                if len(residues) == p:
                    problem_prime = p
                    break
            
            if problem_prime is None:
                break
            
            # For this problem_prime, find the residue class with the fewest elements
            class_counts = {}
            for v in current:
                r = v % problem_prime
                class_counts[r] = class_counts.get(r, 0) + 1
            
            # Randomly choose among the smallest classes
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            chosen_class = random.choice(min_classes)
            
            to_remove = [v for v in current if v % problem_prime == chosen_class]
            for v in to_remove:
                current_set.remove(v)
            current = [v for v in current if v not in to_remove]
        
        # Try to add back removed elements
        current_set_final = set(current)
        removed2 = [v for v in range(N + 1) if v not in current_set_final]
        
        candidate_best = _try_insertion(current, all_primes, removed2, deadline)
        
        if len(candidate_best) > len(best):
            best = candidate_best
    
    return best

# EVOLVE-BLOCK-END
