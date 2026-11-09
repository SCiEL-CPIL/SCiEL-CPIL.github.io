import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 55

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Verify all unordered sums a+b (a<=b) are distinct."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _get_blocking_set(selected, sums):
    """
    Calculate the set of all x in [1, N] that are blocked by selected.
    x is blocked if:
    1. 2x in sums
    2. exists a in selected such that x+a in sums => x = sum - a
    """
    blocked = set()
    # 1. Diagonal blocks
    for s in sums:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                blocked.add(x)
    
    # 2. Off-diagonal blocks
    # For each sum s in sums, and each a in selected, x = s - a is blocked
    # Optimization: iterate sums, then selected
    for s in sums:
        for a in selected:
            x = s - a
            if 1 <= x <= N:
                blocked.add(x)
                
    return blocked

def _is_maximal(selected, sums):
    """
    Check if the set is inclusion-maximal.
    S is maximal if every x in [1, N] \ S is in the blocking set of S.
    """
    blocked = _get_blocking_set(selected, sums)
    sset = set(selected)
    # If any x not in S and not in blocked, it can be added
    for x in range(1, N + 1):
        if x not in sset and x not in blocked:
            return False
    return True

def _greedy_randomize(perm):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in perm:
        # Check if x can be added
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if conflict:
            continue
            
        # Safe to add
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]: # selected[:-1] are the ones present before x
            sums.add(x + a)
            
    return selected, sums

def _local_search_sa(selected, sums, rng, time_limit):
    """
    Simulated Annealing to minimize size.
    Moves:
    1. Remove element (if valid and maximal) -> Accept if size decreases.
    2. Swap element (replace with random other) -> Accept if valid (Sidon) and 
       (maximal OR size same with probability based on T).
       Actually, we only care about minimal size. 
       Standard SA minimizes an energy function. 
       Energy = size + penalty(not maximal).
       We want to find a state with minimal size that is maximal.
       
       Strategy:
       State: (selected, sums)
       Valid: Sidon property holds.
       Maximal: Inclusion maximal.
       
       If we are at a valid, maximal state with size K:
       - Try removing one element. If new state is valid (always true for subset) and maximal, accept (size K-1).
       - If no removal works, try a "swap" to change the structure.
         Swap element e with x (x not in S).
         New set S' = S - {e} + {x}.
         Check if S' is Sidon.
         If Sidon, check if Maximal.
         If Maximal, size is same (K). We accept this with some probability to explore.
         If not Maximal, we might still accept with low probability? No, non-maximal is invalid for the final answer, 
         but can be an intermediate state? 
         The task requires the FINAL result to be maximal.
         If we allow intermediate non-maximal states, we might escape local minima.
         However, "adding ANY missing integer must destroy property" is the definition.
         If we are not maximal, we can just add elements to become maximal? 
         No, adding elements increases size, which is bad.
         
       So, we must stay in the space of Valid Sidon sets.
       We prefer Maximal sets.
       
       Let's define:
       - Move 1: Remove e. If Valid & Maximal -> Accept (Size - 1).
       - Move 2: Swap e with x. If Valid (Sidon) -> 
           If Maximal -> Accept (Size same, but different structure).
           If Not Maximal -> Reject? Or Accept with penalty?
           If we reject non-maximal, we are stuck if no swap leads to maximal.
           
       Better: 
       We only accept moves that result in a VALID Sidon set.
       We track `is_maximal`.
       Objective: Minimize Size, then Maximize Maximality (i.e., be maximal).
       
       Actually, if we start with a maximal set, and only do swaps that result in Sidon sets:
       Some swaps will result in non-maximal sets.
       If we accept them, we can later remove elements?
       If a set is not maximal, it means there is an x that can be added.
       Adding x increases size.
       Removing e decreases size.
       
       If we have a non-maximal set S, it is "smaller" in some sense? No, size is the same.
       But if S is not maximal, it is "closer" to being extendable.
       
       Let's stick to: 
       1. Try to remove. If successful, size drops.
       2. If stuck (no removals possible), try swaps to find a new configuration that *allows* a removal.
       
       Acceptance for Swap:
       If New Set is Sidon:
          If New Set is Maximal: Accept (we found another maximal set of same size, might allow future removal).
          If New Set is NOT Maximal: Accept with probability exp(-1/T)? 
             Why? Because a non-maximal set might be "looser" and allow a removal that leads to a smaller maximal set?
             No, if it's not maximal, it has "holes" that can be filled.
             If we remove an element from a non-maximal set, it might become maximal?
             Unlikely.
             
       Let's try: Only accept swaps if the new set is Sidon. 
       We accept if:
       - New Set is Maximal (always accept, as it's a valid candidate for future removals).
       - New Set is Not Maximal (accept with low probability to explore, hoping to find a path to a smaller maximal set).
       
       However, checking maximality is expensive.
       
       Let's simplify:
       We will perform a "Shake" phase.
       When stuck, we perform random swaps (maintaining Sidon) for a few steps, then check for removals.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.95
    
    while time.time() - start < time_limit and time.time() < DEADLINE:
        if T < T_min:
            T = 1.0 # Restart cooling if stuck too long? Or just keep going.
            
        current_sel = list(selected)
        current_sums = set(sums)
        current_maximal = _is_maximal(current_sel, current_sums)
        
        improved = False
        
        # 1. Try Single Removals
        for i in range(len(current_sel)):
            if time.time() > DEADLINE:
                return best_size, best_sel, best_sums
            
            e = current_sel[i]
            new_sel = current_sel[:i] + current_sel[i+1:]
            
            # Recalculate sums (expensive, but K is small)
            # Optimization: Remove sums involving e
            removed_sums = {2 * e}
            for a in new_sel:
                removed_sums.add(e + a)
            
            new_sums = current_sums - removed_sums
            
            # Check Maximality
            if _is_maximal(new_sel, new_sums):
                # Accept
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break
        
        if improved:
            T *= cooling
            continue
            
        # 2. If stuck, try Swaps
        # Pick a random element to remove
        i = rng.randint(0, len(current_sel) - 1)
        e = current_sel[i]
        
        # Pick a random element to add from [1, N] \ S
        candidates = [x for x in range(1, N + 1) if x not in set(current_sel)]
        if not candidates:
            break # Should not happen for small sets
            
        x = rng.choice(candidates)
        
        # Try swap
        new_sel = [a for a in current_sel if a != e] + [x]
        
        # Check if new_sel is Sidon
        # Build sums
        new_sums = set()
        valid = True
        for a in range(len(new_sel)):
            for b in range(a, len(new_sel)):
                s = new_sel[a] + new_sel[b]
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                break
        
        if not valid:
            T *= cooling
            continue
            
        # If valid, check maximality
        new_maximal = _is_maximal(new_sel, new_sums)
        
        # Acceptance
        if new_maximal:
            # Always accept maximal swaps
            selected = new_sel
            sums = new_sums
            T *= cooling
        else:
            # Accept non-maximal with probability exp(-1/T)
            # This allows exploring "loose" states
            if rng.random() < np.exp(-1.0 / T):
                selected = new_sel
                sums = new_sums
                T *= cooling
            else:
                T *= cooling

    return best_size, best_sel, best_sums

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    
    start = time.time()
    
    # Phase 1: Find a good initial maximal Sidon set
    iterations = 0
    while time.time() < DEADLINE - 15:
        iterations += 1
        
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        selected, sums = _greedy_randomize(perm)
        
        # It is automatically maximal
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)
            best_sums = set(sums)
            
        if best_size <= 20:
             break

    # Phase 2: Local Search (SA) on the best found
    if best_result is not None:
        remaining = DEADLINE - time.time()
        if remaining > 5:
            t_limit = remaining - 5
            new_size, new_sel, new_sums = _local_search_sa(best_result, best_sums, rng, t_limit)
            if new_size < best_size:
                best_size = new_size
                best_result = new_sel
                best_sums = new_sums

    # Phase 3: More restarts if time remains
    if time.time() < DEADLINE - 10 and best_size > 18:
        while time.time() < DEADLINE - 5:
            perm = list(range(1, N + 1))
            rng.shuffle(perm)
            selected, sums = _greedy_randomize(perm)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
                
            if len(selected) <= best_size + 1:
                t_rem = DEADLINE - time.time()
                if t_rem > 5:
                    new_size, new_sel, new_sums = _local_search_sa(selected, sums, rng, min(3.0, t_rem - 5))
                    if new_size < best_size:
                        best_size = new_size
                        best_result = new_sel
                        best_sums = new_sums

    if best_result is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            if 2 * x in sums: continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict: continue
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
