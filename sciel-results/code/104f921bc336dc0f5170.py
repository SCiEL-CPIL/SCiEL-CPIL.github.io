"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

import time
import random

def _precompute():
    """Precompute all forbidden triples and helper structures."""
    N = 128
    # A triple (a,b,c) with a<b<c is forbidden if a+b, a+c, b+c are all in 1..128
    # For a set S, we must not contain all three of any such triple.
    # 
    # Key structural insight: use bitmasks for fast membership testing.
    # Represent S as a 128-bit integer where bit i means (i+1) in S.
    
    # Precompute for each pair (i,j) with i<j, the bitmask of k values
    # such that {min(i,j,k), mid(i,j,k), max(i,j,k)} forms a forbidden triple
    # Actually, better: for each element x, precompute which pairs (a,b) in S
    # would make {x,a,b} a forbidden triple.
    
    # Most efficient: for each x, precompute a list of (a,b) pairs (a<b, a!=x, b!=x)
    # such that {x,a,b} is a forbidden triple. Then x can be added to S iff
    # no such (a,b) has both a and b in S.
    
    forbid_pairs = [[] for _ in range(N+1)]  # 1-indexed
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            if a+b > N:
                break
            for c in range(b+1, N+1):
                if a+c > N:
                    break
                if b+c <= N:
                    # (a,b,c) is a forbidden triple
                    forbid_pairs[a].append((b,c))
                    forbid_pairs[b].append((a,c))
                    forbid_pairs[c].append((a,b))
    
    return forbid_pairs

_FORBID_PAIRS = _precompute()

def _can_add(x, S_mask):
    """Check if x can be added to set represented by S_mask (bit i = element i+1 present)."""
    fp = _FORBID_PAIRS[x]
    for (a, b) in fp:
        # Check if both a and b are in S_mask
        if (S_mask >> (a-1)) & 1 and (S_mask >> (b-1)) & 1:
            return False
    return True

def _local_search(S_mask, deadline, rng):
    """Local search: try to add elements, and if stuck, try 1-remove+1-add or 2-remove+1-add swaps."""
    current = S_mask
    current_size = bin(current).count('1')
    
    # First, greedily try to add all possible elements
    added = True
    while added and time.time() < deadline:
        added = False
        for x in range(1, 129):
            if not (current >> (x-1)) & 1:
                if _can_add(x, current):
                    current |= (1 << (x-1))
                    current_size += 1
                    added = True
    
    # Now try local search: remove one element, try to add two others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elems = [i+1 for i in range(128) if (current >> i) & 1]
        rng.shuffle(elems)
        
        for rem in elems:
            if time.time() > deadline:
                break
            if (current_size - 1) >= best_size_global:
                # Try removing rem and adding others
                trial = current & ~(1 << (rem-1))
                trial_size = current_size - 1
                # Try to add elements
                for x in range(1, 129):
                    if x == rem:
                        continue
                    if not (trial >> (x-1)) & 1:
                        if _can_add(x, trial):
                            trial |= (1 << (x-1))
                            trial_size += 1
                if trial_size > current_size:
                    current = trial
                    current_size = trial_size
                    improved = True
                    break
                # Try adding just one element
                for x in range(1, 129):
                    if x == rem:
                        continue
                    if not (trial >> (x-1)) & 1:
                        if _can_add(x, trial):
                            trial2 = trial | (1 << (x-1))
                            if trial_size + 1 > current_size:
                                current = trial2
                                current_size = trial_size + 1
                                improved = True
                            break
    
    return current, current_size

def _validate(S_list):
    """Validate that S_list is a valid set."""
    Sset = set(S_list)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            for k in range(j+1, n):
                c = S_list[k]
                if (a+b) in Sset and (a+c) in Sset and (b+c) in Sset:
                    return False
    return True

best_size_global = 0

def construct():
    global best_size_global
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with a good initial solution: all elements 64..128 plus some from 1..63
    # Elements 64..128: their pairwise sums are all > 128, so they can't be "sums"
    # They can only participate as addends.
    base = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]
    
    # Try to add elements from 1..63
    initial_mask = 0
    for x in base:
        initial_mask |= (1 << (x-1))
    
    # Greedily add from 1..63
    for x in range(1, 64):
        if _can_add(x, initial_mask):
            initial_mask |= (1 << (x-1))
    
    initial_list = [i+1 for i in range(128) if (initial_mask >> i) & 1]
    best_size_global = len(initial_list)
    best_mask = initial_mask
    
    # Try multiple randomized local searches
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        rng = random.Random(trial * 7919 + 13)
        
        # Start from the initial solution or a perturbation
        if trial == 0:
            start_mask = initial_mask
        else:
            # Perturb: remove a random few elements
            start_mask = initial_mask
            elems = [i+1 for i in range(128) if (start_mask >> i) & 1]
            rng.shuffle(elems)
            for _ in range(rng.randint(1, 5)):
                start_mask &= ~(1 << (elems[0]-1))
                elems.pop(0)
        
        new_mask, new_size = _local_search(start_mask, deadline - 10.0, rng)
        
        if new_size > best_size_global:
            best_size_global = new_size
            best_mask = new_mask
    
    # Also try pure greedy with different orderings
    for trial in range(100):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(trial * 104729 + 37)
        candidates = list(range(1, 129))
        rng.shuffle(candidates)
        
        mask = 0
        for x in candidates:
            if _can_add(x, mask):
                mask |= (1 << (x-1))
        
        size = bin(mask).count('1')
        if size > best_size_global:
            best_size_global = size
            best_mask = mask
    
    # Convert best mask to list
    best_list = [i+1 for i in range(128) if (best_mask >> i) & 1]
    
    # Validate
    if not _validate(best_list):
        # Fallback
        best_list = [1, 62] + list(range(64, 129))
    
    return [int(x) for x in sorted(best_list)]

# EVOLVE-BLOCK-END
