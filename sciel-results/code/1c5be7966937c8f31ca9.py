import time
import random
import math
import sys
import numpy as np
from functools import lru_cache

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

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

PRIMES = _primes(N)

def _construct_dp():
    """
    Dynamic programming over the CRT hierarchy.
    
    State: (k, assignments_tuple) where k is the number of primes processed,
    and assignments_tuple is a tuple of (prime, residue) pairs for the first k primes.
    
    For each state, we track the survivor set S.
    Transition: add a new prime p_{k+1} and choose a residue r to remove.
    
    We use beam search to keep the state space manageable.
    """
    start_time = time.time()
    
    # We'll do a beam search over residue assignments for the first K primes.
    # K will be determined by time budget.
    
    # Initialize: no primes processed, S = [0, N]
    # But S as a full set is expensive. Instead, we can use inclusion-exclusion
    # or just track the set for small K.
    
    # For K up to ~8 primes, the product of primes is manageable.
    # Let's use a dictionary: key = tuple of residues for first K primes,
    # value = the survivor set (as a frozenset or list).
    
    # Actually, for efficiency, let's use numpy arrays and track membership.
    
    # Beam search approach:
    # Level 0: one state, empty assignment, S = all numbers
    # Level 1: for prime 2, try residue 0 or 1 -> 2 states
    # Level 2: for prime 3, try residues 0,1,2 -> up to 6 states
    # ...
    # At each level, keep only the top B states by |S|.
    
    B = 50  # Beam width
    K_max = 10  # Max primes to consider in DP
    
    # State: (assignment_dict, survivor_set)
    # We'll represent survivor_set as a numpy boolean array for speed.
    
    # Initialize
    all_ones = np.ones(N + 1, dtype=bool)
    states = [({}, all_ones.copy())]
    
    for k, p in enumerate(PRIMES[:K_max]):
        if time.time() - start_time > MAX_TIME - 30:
            break
            
        new_states = []
        
        for assignment, S in states:
            # For this state, try each residue r mod p
            for r in range(p):
                # Create new survivor set
                new_S = S.copy()
                # Remove numbers where x % p == r
                # Efficient way: create mask
                mask = np.arange(N + 1) % p != r
                new_S = new_S & mask
                
                size = new_S.sum()
                
                # Check if we're running out of time
                if time.time() - start_time > MAX_TIME - 20:
                    break
                
                new_assignment = dict(assignment)
                new_assignment[p] = r
                
                new_states.append((new_assignment, new_S, size))
        
        # Keep only top B states by size
        new_states.sort(key=lambda x: x[2], reverse=True)
        states = [(a, s) for a, s, _ in new_states[:B]]
        
        # Prune states that are too small
        min_size = states[-1][1].sum() if states else 0
        if min_size < 100:
            # If we're getting very small sets, stop expanding
            break
    
    # Now, for each final state, complete the assignment greedily
    # and check admissibility.
    
    best_S = []
    best_size = 0
    
    rng = random.Random(42)
    
    for assignment, S in states:
        if time.time() - start_time > MAX_TIME - 10:
            break
        
        # Convert S to a list for the greedy finisher
        S_list = list(np.where(S)[0])
        
        # Greedily complete the assignment
        committed = dict(assignment)
        
        while True:
            len_S = len(S_list)
            if len_S == 0:
                break
            
            # Find a violating prime
            viol_p = None
            for p in PRIMES:
                if p > len_S:
                    break
                if p in committed:
                    continue
                # Check if all residues mod p are present
                residues = set(x % p for x in S_list)
                if len(residues) == p:
                    viol_p = p
                    break
            
            if viol_p is None:
                break
            
            p = viol_p
            # Choose residue with min count
            counts = [0] * p
            for x in S_list:
                counts[x % p] += 1
            
            min_count = min(counts)
            candidates = [r for r in range(p) if counts[r] == min_count]
            r_to_remove = rng.choice(candidates)
            
            committed[p] = r_to_remove
            S_list = [x for x in S_list if x % p != r_to_remove]
        
        if len(S_list) > best_size:
            best_S = S_list
            best_size = len(S_list)
    
    return best_S

def _construct_greedy(assignments, rng):
    """Original greedy construction for comparison."""
    S = set(range(N + 1))
    
    for p, r in assignments.items():
        S = {x for x in S if x % p != r}
        
    if not S:
        return []
        
    committed = dict(assignments)
    
    while True:
        len_S = len(S)
        if len_S == 0:
            break
            
        viol_p = None
        for p in PRIMES:
            if p > len_S:
                break
            if p in committed:
                continue
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
            
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
            
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        r_to_remove = rng.choice(candidates)
        
        committed[p] = r_to_remove
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _construct():
    start_time = time.time()
    best_S = []
    best_size = 0
    
    # Phase 1: DP-based construction
    dp_result = _construct_dp()
    if len(dp_result) > best_size:
        best_S = dp_result
        best_size = len(dp_result)
    
    # Phase 2: Greedy with various fixed assignments
    rng = random.Random(42)
    
    # Try empty assignments
    S = _construct_greedy({}, rng)
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
    
    # Try fixing prime 2
    for r2 in [0, 1]:
        if time.time() - start_time > MAX_TIME - 40:
            break
        S = _construct_greedy({2: r2}, rng)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Try fixing primes 2 and 3
    for r2 in [0, 1]:
        for r3 in [0, 1, 2]:
            if time.time() - start_time > MAX_TIME - 30:
                break
            S = _construct_greedy({2: r2, 3: r3}, rng)
            if len(S) > best_size:
                best_S = S
                best_size = len(S)
                
    # Try fixing primes 2, 3, 5
    for r2 in [0, 1]:
        for r3 in [0, 1, 2]:
            for r5 in [0, 1, 2, 3, 4]:
                if time.time() - start_time > MAX_TIME - 20:
                    break
                S = _construct_greedy({2: r2, 3: r3, 5: r5}, rng)
                if len(S) > best_size:
                    best_S = S
                    best_size = len(S)
                    
    # Phase 3: Randomized restarts
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 10:
        small_primes = [2, 3, 5, 7, 11]
        while time.time() - start_time < MAX_TIME - 5:
            k = rng.randint(1, 5)
            chosen_primes = rng.sample(small_primes, k)
            assignments = {}
            for p in chosen_primes:
                assignments[p] = rng.randint(0, p - 1)
                
            S = _construct_greedy(assignments, rng)
            if len(S) > best_size:
                best_S = S
                best_size = len(S)
                
    return best_S

def construct():
    return _construct()

if __name__ == "__main__":
    result = construct()
    print(f"Best size: {len(result)}")
