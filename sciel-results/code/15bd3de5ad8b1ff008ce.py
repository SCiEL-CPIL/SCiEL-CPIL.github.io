import time
import math
import random
import numpy as np

N = 5000

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

MAX_PRIME_FOR_PRECOMP = 1000
ALL_PRIMES = _primes(MAX_PRIME_FOR_PRECOMP)

# Precompute residue class indices
prime_indices = {}
for p in ALL_PRIMES:
    indices = [[] for _ in range(p)]
    for x in range(N + 1):
        r = x % p
        indices[r].append(x)
    prime_indices[p] = indices

def _is_admissible(S_active):
    """Check if the active set is admissible."""
    size = int(S_active.sum())
    for p in ALL_PRIMES:
        if p > size:
            break
        all_present = True
        for r in range(p):
            found = False
            for x in prime_indices[p][r]:
                if S_active[x]:
                    found = True
                    break
            if not found:
                all_present = False
                break
        if all_present:
            return False
    return True

def _compute_conflict_degrees(S_active):
    """Compute conflict degree for each element.
    
    For each prime p, find which residue classes are 'full' (all elements active).
    The elements in those full residue classes have higher conflict degree.
    We approximate: for each prime p, if all residues are present, each element
    contributes to the 'pressure' to remove one residue. Elements in the most
    crowded residues have higher degree.
    """
    degrees = np.zeros(N + 1, dtype=np.float64)
    size = int(S_active.sum())
    
    for p in ALL_PRIMES:
        if p > size:
            break
        
        # Count active elements per residue
        counts = np.zeros(p, dtype=np.int32)
        for r in range(p):
            for x in prime_indices[p][r]:
                if S_active[x]:
                    counts[r] += 1
        
        # Check if all residues are present
        if np.all(counts > 0):
            # All residues present - we need to remove one
            # Elements in residues with highest count have highest "cost"
            # to remove (we lose more elements), so they should be removed first
            # Actually, we want to remove the residue with MIN count to lose fewest
            # So elements in min-count residues are "cheapest" to remove
            # Elements in max-count residues are most "valuable" to keep
            
            # Conflict degree: elements in residues that are "critical" (all present)
            # get a boost. The more crowded a residue, the more its elements contribute
            # to the violation.
            max_count = counts.max()
            min_count = counts.min()
            
            for r in range(p):
                # Elements in this residue contribute to the violation
                # Weight by how "critical" this residue is
                # If we remove this residue, we lose counts[r] elements
                # So removing a low-count residue is better
                # Elements in high-count residues are more "expensive" to sacrifice
                # Give higher degree to elements in residues that are harder to remove
                # (i.e., high count) because they're more valuable
                weight = counts[r] / max_count if max_count > 0 else 0
                for x in prime_indices[p][r]:
                    if S_active[x]:
                        degrees[x] += weight
    
    return degrees

def _greedy_remove(S_active, degrees, deadline):
    """Greedily remove elements with highest conflict degree until admissible."""
    current_size = int(S_active.sum())
    
    # Get list of active elements sorted by degree (highest first)
    active_elements = [x for x in range(N + 1) if S_active[x]]
    active_elements.sort(key=lambda x: -degrees[x])
    
    for x in active_elements:
        if time.time() > deadline:
            break
        if not S_active[x]:
            continue
        
        # Try removing x
        S_active[x] = False
        new_size = current_size - 1
        
        # Check if admissible
        if _is_admissible(S_active):
            current_size = new_size
            # Recompute degrees for remaining elements
            degrees = _compute_conflict_degrees(S_active)
            # Re-sort remaining active elements
            active_elements = [x for x in range(N + 1) if S_active[x]]
            active_elements.sort(key=lambda x: -degrees[x])
        else:
            S_active[x] = True
    
    return S_active, current_size

def _local_search_add(S_active, current_size, deadline):
    """Try to add back removed elements to increase size."""
    inactive_elements = [x for x in range(N + 1) if not S_active[x]]
    random.shuffle(inactive_elements)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in inactive_elements:
            if time.time() > deadline:
                break
            
            S_active[x] = True
            new_size = current_size + 1
            
            if _is_admissible(S_active):
                current_size = new_size
                improved = True
                inactive_elements.remove(x)
            else:
                S_active[x] = False
    
    return S_active, current_size

def _local_search_swap(S_active, current_size, deadline):
    """Try swapping: remove one element, add another to maintain admissibility."""
    active_elements = [x for x in range(N + 1) if S_active[x]]
    inactive_elements = [x for x in range(N + 1) if not S_active[x]]
    random.shuffle(active_elements)
    random.shuffle(inactive_elements)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x_remove in active_elements:
            if time.time() > deadline:
                break
            if not S_active[x_remove]:
                continue
            
            # Try removing x_remove
            S_active[x_remove] = False
            
            # Try adding each inactive element
            for x_add in inactive_elements:
                if time.time() > deadline:
                    break
                if S_active[x_add]:
                    continue
                
                S_active[x_add] = True
                new_size = current_size  # same size (swap)
                
                if _is_admissible(S_active):
                    # Accept the swap
                    current_size = new_size
                    improved = True
                    active_elements.remove(x_remove)
                    if x_remove in inactive_elements:
                        inactive_elements.remove(x_remove)
                    inactive_elements.append(x_remove)
                    active_elements.append(x_add)
                    if x_add in inactive_elements:
                        inactive_elements.remove(x_add)
                    break
                else:
                    S_active[x_add] = False
            
            # If no swap worked, put x_remove back
            if not S_active[x_remove]:
                S_active[x_remove] = True
    
    return S_active, current_size

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = 0
    best_set = []
    
    # Multi-start: try different random initializations
    num_starts = 15
    
    for trial in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        # Randomize: start with a random subset or full set
        if trial % 3 == 0:
            # Start with full set
            S_active = np.ones(N + 1, dtype=bool)
        elif trial % 3 == 1:
            # Start with random 80% subset
            S_active = np.random.random(N + 1) < 0.8
        else:
            # Start with random 90% subset
            S_active = np.random.random(N + 1) < 0.9
        
        current_size = int(S_active.sum())
        
        # Phase 1: Greedy removal based on conflict degrees
        degrees = _compute_conflict_degrees(S_active)
        S_active, current_size = _greedy_remove(S_active, degrees, deadline)
        
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
        
        # Phase 2: Local search - add back elements
        S_active, current_size = _local_search_add(S_active, current_size, deadline)
        
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
        
        # Phase 3: Local search - swaps to improve structure
        S_active, current_size = _local_search_swap(S_active, current_size, deadline)
        
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
        
        # Phase 4: Another round of add-back after swaps
        S_active, current_size = _local_search_add(S_active, current_size, deadline)
        
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
    
    return sorted(best_set)
