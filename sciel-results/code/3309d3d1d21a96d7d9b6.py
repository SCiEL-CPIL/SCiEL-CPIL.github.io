"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _compute_max_rep(A, n=128):
    """Compute the maximum representation multiplicity for sums in [n, 2n].
    Pairs are unordered and permit equal summands.
    """
    A = list(A)
    if not A:
        return float('inf')
    A_set = set(A)
    # For each target t in [n, 2n], count pairs (a,b) with a <= b, a+b=t, a,b in A
    # We can do this efficiently:
    max_rep = 0
    for t in range(n, 2*n + 1):
        count = 0
        for a in A:
            b = t - a
            if b < a:
                continue
            if b in A_set:
                count += 1
        max_rep = max(max_rep, count)
    return max_rep

def _is_valid(A, n=128):
    """Check if every integer in [n, 2n] is a sum of two elements of A."""
    A = list(A)
    if not A:
        return False
    A_set = set(A)
    for t in range(n, 2*n + 1):
        found = False
        for a in A:
            b = t - a
            if b >= a and b in A_set:
                found = True
                break
        if not found:
            return False
    return True

def _greedy_peel(initial_A, n=128, max_time=45.0):
    """Greedy peeling: start with initial_A, try to remove elements while maintaining validity.
    Then do local search swaps.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    A = list(initial_A)
    A_set = set(A)
    
    # First, ensure validity
    if not _is_valid(A, n):
        # Try to fix by adding elements
        for x in range(n+1):
            if x not in A_set and _is_valid(A + [x], n):
                A.append(x)
                A_set.add(x)
        if not _is_valid(A, n):
            return list(range(n//2, n+1))  # fallback
    
    current_max = _compute_max_rep(A, n)
    
    # Greedy peeling: try to remove elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(A)):
            if time.time() > deadline:
                break
            candidate = A[:i] + A[i+1:]
            if not candidate:
                continue
            if _is_valid(candidate, n):
                new_max = _compute_max_rep(candidate, n)
                if new_max <= current_max:
                    A = candidate
                    current_max = new_max
                    improved = True
                    break  # restart the loop
    
    # Local search: swaps (remove one, add another)
    for _ in range(5000):
        if time.time() > deadline:
            break
        if len(A) <= 2:
            break
        i = random.randint(0, len(A)-1)
        # Try removing A[i] and adding a random element
        candidate = [x for x in A if x != A[i]]
        # Try adding a random element
        candidates_to_add = [x for x in range(n+1) if x not in set(candidate)]
        if not candidates_to_add:
            continue
        new_elem = random.choice(candidates_to_add)
        new_A = candidate + [new_elem]
        if _is_valid(new_A, n):
            new_max = _compute_max_rep(new_A, n)
            if new_max < current_max:
                A = new_A
                current_max = new_max
    
    # Also try single swaps: remove A[i], add x
    for _ in range(3000):
        if time.time() > deadline:
            break
        if len(A) <= 2:
            break
        i = random.randint(0, len(A)-1)
        candidate = [x for x in A if x != A[i]]
        candidates_to_add = [x for x in range(n+1) if x not in set(candidate)]
        if not candidates_to_add:
            continue
        new_elem = random.choice(candidates_to_add)
        new_A = candidate + [new_elem]
        if _is_valid(new_A, n):
            new_max = _compute_max_rep(new_A, n)
            if new_max < current_max:
                A = new_A
                current_max = new_max
    
    return A

def _local_search(A, n=128, max_time=30.0):
    """Local search with swaps and small perturbations."""
    start_time = time.time()
    deadline = start_time + max_time
    
    A = list(A)
    if not _is_valid(A, n):
        return A
    
    current_max = _compute_max_rep(A, n)
    A_set = set(A)
    
    # Precompute contribution of each element to each target
    # For element a, it contributes to targets t where t-a is in A and t-a >= a
    # i.e., t in [2a, 2n] and t-a in A
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Try removing an element
        if len(A) > 3:
            i = random.randint(0, len(A)-1)
            candidate = A[:i] + A[i+1:]
            if _is_valid(candidate, n):
                new_max = _compute_max_rep(candidate, n)
                if new_max <= current_max:
                    A = candidate
                    current_max = new_max
                    continue
        
        # Try swapping: remove one, add one
        if len(A) >= 3:
            i = random.randint(0, len(A)-1)
            candidate = [x for x in A if x != A[i]]
            cand_set = set(candidate)
            # Pick a random element to add
            possible_adds = [x for x in range(n+1) if x not in cand_set]
            if possible_adds:
                new_elem = random.choice(possible_adds)
                new_A = candidate + [new_elem]
                if _is_valid(new_A, n):
                    new_max = _compute_max_rep(new_A, n)
                    if new_max < current_max:
                        A = new_A
                        current_max = new_max
    
    return A

def construct():
    """Construct a subset A of [0,128] such that every integer in [128,256] is a sum of two elements of A,
    minimizing the largest representation multiplicity."""
    start_time = time.time()
    hard_deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    
    # Start with a good initial set
    # The seed uses range(64, 129) which has 65 elements
    # Let's start with that and try to improve
    initial_A = list(range(64, 129))
    
    # Verify initial validity
    if not _is_valid(initial_A, n):
        # Fallback: use full range
        initial_A = list(range(0, 129))
    
    best_A = initial_A
    best_max = _compute_max_rep(best_A, n)
    
    # Phase 1: Greedy peeling
    if time.time() < hard_deadline - 15:
        peeled = _greedy_peel(initial_A, n, max_time=min(30.0, hard_deadline - time.time() - 15))
        if _is_valid(peeled, n):
            peeled_max = _compute_max_rep(peeled, n)
            if peeled_max < best_max:
                best_A = peeled
                best_max = peeled_max
    
    # Phase 2: Local search from best
    if time.time() < hard_deadline - 5:
        remaining = min(30.0, hard_deadline - time.time() - 5)
        local = _local_search(best_A, n, max_time=remaining)
        if _is_valid(local, n):
            local_max = _compute_max_rep(local, n)
            if local_max < best_max:
                best_A = local
                best_max = local_max
    
    # Phase 3: Try different initializations
    if time.time() < hard_deadline - 5:
        # Try starting from a symmetric set
        sym_A = list(range(0, 65))  # [0, 64]
        # Check if valid: need sums in [128, 256]
        # Max sum from [0,64] is 128, so we can only cover 128. Not valid alone.
        # Try [32, 128]
        sym_A2 = list(range(32, 129))
        if _is_valid(sym_A2, n):
            sym_max = _compute_max_rep(sym_A2, n)
            if sym_max < best_max:
                best_A = sym_A2
                best_max = sym_max
                if time.time() < hard_deadline - 10:
                    local2 = _local_search(best_A, n, max_time=min(20.0, hard_deadline - time.time() - 5))
                    if _is_valid(local2, n):
                        local2_max = _compute_max_rep(local2, n)
                        if local2_max < best_max:
                            best_A = local2
                            best_max = local2_max
        
        # Try [0, 128] full set and peel
        full_A = list(range(0, 129))
        if _is_valid(full_A, n):
            full_max = _compute_max_rep(full_A, n)
            if full_max < best_max:
                best_A = full_A
                best_max = full_max
                if time.time() < hard_deadline - 10:
                    peeled2 = _greedy_peel(best_A, n, max_time=min(20.0, hard_deadline - time.time() - 5))
                    if _is_valid(peeled2, n):
                        peeled2_max = _compute_max_rep(peeled2, n)
                        if peeled2_max < best_max:
                            best_A = peeled2
                            best_max = peeled2_max
    
    # Final check
    if not _is_valid(best_A, n):
        # Emergency fallback
        best_A = list(range(64, 129))
    
    return best_A
# EVOLVE-BLOCK-END
