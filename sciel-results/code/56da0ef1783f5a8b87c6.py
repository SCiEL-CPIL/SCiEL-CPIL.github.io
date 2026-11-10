import time
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Initial set
    S = set(range(N + 1))
    removed = []
    
    # Precompute primes up to N
    primes = _primes(N)
    
    # Helper to get current size
    def get_size():
        return len(S)
    
    # Helper to check if a prime is "bad" (all residues present)
    # We need to track residue counts for each prime.
    # Updating this dynamically for all primes is expensive.
    # Instead, we can recompute for relevant primes when needed.
    
    # Relevant primes are p <= len(S).
    
    while True:
        if time.time() > deadline:
            break
            
        current_size = len(S)
        
        # Find bad primes
        bad_primes = []
        for p in primes:
            if p > current_size:
                break
            # Check if all residues mod p are present in S
            # We can do this by checking if the set of residues has size p
            # Optimization: If we just removed elements, we can cache?
            # For now, recompute.
            residues = set()
            for x in S:
                residues.add(x % p)
                if len(residues) == p:
                    break
            if len(residues) == p:
                bad_primes.append(p)
                
        if not bad_primes:
            break
            
        # For each bad prime, find the minimum residue class
        # We want to remove the union of these minimum classes.
        elements_to_remove = set()
        
        for p in bad_primes:
            # Count elements in each residue class
            class_counts = [0] * p
            for x in S:
                class_counts[x % p] += 1
                
            # Find min count
            min_count = min(class_counts)
            
            # Find all classes with this min count?
            # Or just one?
            # If multiple classes have the same min count, removing any one satisfies the constraint.
            # However, removing the union of *all* min classes might be too aggressive.
            # We should pick ONE class to empty.
            # Which one?
            # Heuristic: Pick the class that contains elements that are also in min classes of other bad primes?
            # That's complex.
            # Let's stick to picking the class with the minimum count.
            # If tie, pick the one that appears most frequently in other bad primes' min classes?
            # For simplicity, let's pick the first one, or randomize?
            # Let's pick the one that minimizes the "cost" in terms of other constraints.
            
            # Simple heuristic: Pick the class with min count.
            min_class_idx = class_counts.index(min_count)
            
            # Add all elements in this class to removal set
            for x in S:
                if x % p == min_class_idx:
                    elements_to_remove.add(x)
                    
        # Remove elements
        S -= elements_to_remove
        removed.extend(elements_to_remove)
        
    # Polish step: Try to add back removed elements
    # We iterate through removed elements and try to add them back if valid.
    # To be efficient, we can just try to add them back in a loop.
    # Since we might add one, the size increases, so more primes might become relevant.
    
    added_back = True
    while added_back and time.time() < deadline:
        added_back = False
        # Sort removed by some heuristic? Smaller numbers first? Random?
        # Let's just iterate through a copy of removed
        for x in list(removed):
            if time.time() > deadline:
                break
            # Check if adding x back is valid
            new_size = len(S) + 1
            valid = True
            for p in primes:
                if p > new_size:
                    break
                # Check if x fills the unique empty class for p
                # We need to know the current residues mod p in S
                # And check if (x % p) is the only missing one
                residues = set()
                for y in S:
                    residues.add(y % p)
                    if len(residues) == p:
                        # Already full, so adding x is fine?
                        # Wait, if it's already full, it's a bad prime.
                        # We are adding x.
                        # If it's already full, adding x doesn't change the "full" status.
                        # So it's valid?
                        # Yes, if it's already bad, it stays bad.
                        # We only care if it transitions from Good to Bad.
                        # Good means some class is empty.
                        # If x fills the *only* empty class, it becomes Bad.
                        pass
                if len(residues) < p:
                    # There is at least one empty class
                    missing = set(range(p)) - residues
                    if x % p in missing and len(missing) == 1:
                        # x fills the only missing class
                        valid = False
                        break
            
            if valid:
                S.add(x)
                removed.remove(x)
                added_back = True
                
    return list(S)

# EVOLVE-BLOCK-END
