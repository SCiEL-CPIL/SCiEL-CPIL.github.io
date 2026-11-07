"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

CANDIDATES = list(range(2, P['n'] + 1))

def is_valid(S):
    """
    Check if all subset products of S are distinct.
    Returns True if valid, False otherwise.
    """
    if not S:
        return True
    
    prods = {1}
    for x in S:
        # We need to check if adding x to the current set of products
        # creates any duplicates.
        # The new products are {p * x for p in prods}.
        # If any p * x is already in prods, or if any p1 * x == p2 * x (impossible since x!=0),
        # we have a collision.
        # Actually, we just need to ensure {p * x for p in prods} is disjoint from prods.
        # And obviously {p * x for p in prods} are distinct from each other if prods are distinct and x != 0.
        
        new_prods = set()
        for p in prods:
            np = p * x
            if np in prods:
                return False
            if np in new_prods:
                # This can happen if x=0, but x>=2, so p1*x == p2*x => p1==p2.
                # So this check is technically redundant if prods is a set, but good for safety.
                return False
            new_prods.add(np)
        prods.update(new_prods)
    return True

def greedy_fill(S, deadline):
    """
    Try to add any remaining candidate to S if it keeps S valid.
    Returns the updated set S.
    """
    added = True
    while added and time.time() < deadline:
        added = False
        # Shuffle candidates for randomization
        candidates = [c for c in CANDIDATES if c not in S]
        random.shuffle(candidates)
        for c in candidates:
            # Check if adding c is valid
            # We can do a quick check: is_valid(S | {c})
            # To save time, we can just try it.
            new_S = S | {c}
            if is_valid(new_S):
                S = new_S
                added = True
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial best set: Primes
    # Primes are a known valid set.
    primes = [a for a in CANDIDATES if all(a % d for d in range(2, int(a**0.5) + 1))]
    best_set = set(primes)
    best_size = len(best_set)
    
    if best_size == P['maximum_size']:
        return list(best_set)

    # Local Search
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Start from current best or a random perturbation
        if restart == 0:
            S = set(best_set)
        elif restart % 5 == 0:
            S = set(best_set)
        else:
            # Random restart: start from a random subset of best_set or just primes
            # To ensure diversity, mix primes and composites
            S = set(random.sample(CANDIDATES, min(len(CANDIDATES), best_size + random.randint(-2, 2))))
            # Ensure S is valid, otherwise shrink it
            # A random set is likely invalid. We should start from a valid set.
            # Better: Start from primes, then randomize.
            S = set(primes)
            # Remove some random primes
            for _ in range(random.randint(0, 3)):
                if S:
                    S.discard(random.choice(list(S)))
            # Add some random composites
            composites = [c for c in CANDIDATES if c not in primes]
            for _ in range(random.randint(0, 2)):
                if composites:
                    S.add(random.choice(composites))
            
            # Repair S to be valid
            # Simple repair: remove elements that cause collisions
            # This is complex. Instead, let's just start from a valid set.
            # The previous logic of "random subset of best_set" is safer if we ensure validity.
            # Let's stick to starting from best_set and perturbing.
            
            # Perturbation: remove 1-2 random elements from best_set
            S = set(best_set)
            for _ in range(random.randint(1, 2)):
                if S:
                    S.discard(random.choice(list(S)))
            
            # Add 1-2 random elements from candidates not in S
            candidates = [c for c in CANDIDATES if c not in S]
            for _ in range(random.randint(1, 2)):
                if candidates:
                    S.add(random.choice(candidates))
            
            # Now S might be invalid. We need to make it valid before greedy filling.
            # Strategy: try to remove elements until valid, or just restart if invalid.
            # Given the time, it's better to ensure we start with a valid set.
            # Let's try a different approach: Hill Climbing on the set of valid sets.
            
            # If S is not valid, discard this restart and try a simpler perturbation
            if not is_valid(S):
                # Fallback: just use best_set
                S = set(best_set)
                # Remove one random element to allow a different path
                if S:
                    S.discard(random.choice(list(S)))
                # Add one random element
                candidates = [c for c in CANDIDATES if c not in S]
                if candidates:
                    S.add(random.choice(candidates))
                if not is_valid(S):
                    # If still invalid, revert to best_set
                    S = set(best_set)

        # Greedy Fill
        S = greedy_fill(S, deadline)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(best_size) if False else len(S) # Fix typo in thought, code below
            if best_size == P['maximum_size']:
                break

        # Swap Phase
        if len(S) < P['maximum_size'] and time.time() < deadline - 5:
            for _ in range(50):
                if time.time() > deadline:
                    break
                if not S:
                    break
                
                # Remove one random element
                to_remove = random.choice(list(S))
                S.discard(to_remove)
                
                # Try to add a random element
                candidates = [c for c in CANDIDATES if c not in S]
                if not candidates:
                    S.add(to_remove)
                    break
                
                to_add = random.choice(candidates)
                S.add(to_add)
                
                if is_valid(S):
                    # Valid swap, try to fill
                    S = greedy_fill(S, deadline)
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
                        if best_size == P['maximum_size']:
                            break
                else:
                    # Invalid swap, revert
                    S.discard(to_add)
                    S.add(to_remove)
                    
        if best_size == P['maximum_size']:
            break

    # Convert to list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
