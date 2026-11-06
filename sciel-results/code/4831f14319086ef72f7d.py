import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Candidate pools, ordered by likely feasibility
    # Larger numbers = fewer constraints (fewer multiples to check)
    pools = [
        list(range(160, 193)),   # 33 numbers - too small for 48, skip
        list(range(144, 193)),   # 49 numbers
        list(range(128, 193)),   # 65 numbers
        list(range(112, 193)),   # 81 numbers
        list(range(96, 193)),    # 97 numbers
        list(range(80, 193)),    # 113 numbers
        list(range(1, 193))      # Fallback
    ]
    
    best_set = []
    best_size = 0
    
    # Fast validity check using precomputed triple sums
    def _is_valid(S):
        """Check if S is valid: no element divides sum of any 3 distinct others."""
        n = len(S)
        if n < 4:
            return True
        
        # Build map: sum -> list of indices of triples
        sums_map = {}
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    s = S[i] + S[j] + S[k]
                    if s not in sums_map:
                        sums_map[s] = []
                    sums_map[s].append((i, j, k))
        
        # For each element a=S[a_idx], check if a divides any triple sum
        # where the triple does NOT contain a_idx
        for a_idx, a in enumerate(S):
            # Check multiples of a
            # Min possible sum of 3 others >= 3*1 = 3
            # Max possible sum = 3*192 = 576
            max_multiple = 576 // a
            for k in range(1, max_multiple + 1):
                target = k * a
                if target in sums_map:
                    for (i, j, kk) in sums_map[target]:
                        # Check if a_idx is NOT in this triple
                        if a_idx != i and a_idx != j and a_idx != kk:
                            return False
        return True
    
    def _find_conflict(S):
        """Find a conflict: returns (a_idx, b_idx, c_idx, d_idx) or None."""
        n = len(S)
        if n < 4:
            return None
        
        sums_map = {}
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    s = S[i] + S[j] + S[k]
                    if s not in sums_map:
                        sums_map[s] = []
                    sums_map[s].append((i, j, k))
        
        for a_idx, a in enumerate(S):
            max_multiple = 576 // a
            for k in range(1, max_multiple + 1):
                target = k * a
                if target in sums_map:
                    for (i, j, kk) in sums_map[target]:
                        if a_idx != i and a_idx != j and a_idx != kk:
                            return (a_idx, i, j, kk)
        return None
    
    # Try each pool with random sampling + local repair
    for pool in pools:
        if time.time() > deadline:
            break
        
        if len(pool) < 48:
            continue
            
        pool_set = set(pool)
        
        # Multiple random restarts with local search
        max_attempts = 3000
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random sample of 48
            S = random.sample(pool, 48)
            
            if _is_valid(S):
                return S
            
            # Local repair: find conflicts and try to fix them
            repair_rounds = 0
            max_repair_rounds = 50
            S_valid = False
            
            while repair_rounds < max_repair_rounds and time.time() < deadline:
                repair_rounds += 1
                
                conflict = _find_conflict(S)
                if conflict is None:
                    S_valid = True
                    break
                    
                a_idx, i, j, kk = conflict
                # Conflict: S[a_idx] divides S[i]+S[j]+S[kk]
                # Try swapping one of the summands (i, j, kk) with an unused element
                # Or swap a_idx itself
                
                unused = list(pool_set - set(S))
                if not unused:
                    break
                    
                # Try several swap strategies
                swapped = False
                for _ in range(20):
                    if time.time() > deadline:
                        break
                        
                    # Pick which element to swap out
                    # Prefer swapping one of the summands to break the divisibility
                    candidates = [i, j, kk]
                    if random.random() < 0.3:
                        candidates.append(a_idx)
                        
                    elem_idx = random.choice(candidates)
                    new_elem = random.choice(unused)
                    
                    if new_elem in S:
                        continue
                        
                    new_S = S.copy()
                    new_S[elem_idx] = new_elem
                    
                    if _is_valid(new_S):
                        S = new_S
                        swapped = True
                        break
                
                if not swapped:
                    # Try a different approach: remove the problematic element and add a new one
                    # This keeps size at 48
                    elem_idx = a_idx
                    new_elem = random.choice(unused)
                    if new_elem in S:
                        continue
                    new_S = S.copy()
                    new_S[elem_idx] = new_elem
                    if _is_valid(new_S):
                        S = new_S
                        swapped = True
                    else:
                        # Try another random swap
                        for _ in range(10):
                            if time.time() > deadline:
                                break
                            elem_idx2 = random.choice(range(len(S)))
                            new_elem2 = random.choice(unused)
                            if new_elem2 in S:
                                continue
                            new_S2 = S.copy()
                            new_S2[elem_idx2] = new_elem2
                            if _is_valid(new_S2):
                                S = new_S2
                                swapped = True
                                break
                        if not swapped:
                            break  # Give up on this repair round
            
            if S_valid:
                return S
        
        # If we got a partially valid set, track it
        # (We don't easily track partial validity here, so skip)
    
    # Fallback: Greedy construction from largest pool
    if time.time() < deadline:
        pool = list(range(1, 193))
        S = []
        # Try adding elements in descending order (larger = fewer constraints)
        for x in sorted(pool, reverse=True):
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
            candidate = S + [x]
            if _is_valid(candidate):
                S.append(x)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Second fallback: Greedy with randomization
    if time.time() < deadline and best_size < 48:
        for trial in range(20):
            if time.time() > deadline:
                break
            pool = list(range(1, 193))
            random.shuffle(pool)
            S = []
            for x in pool:
                if time.time() > deadline:
                    break
                if len(S) >= 48:
                    break
                candidate = S + [x]
                if _is_valid(candidate):
                    S.append(x)
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
    
    return best_set

# Keep the helper functions for compatibility (they're not used in the new implementation)
def _is_valid(S, pool_min=1):
    """Legacy validity check."""
    return _is_valid_internal(S)

def _find_conflict(S, pool_min=1):
    """Legacy conflict finder."""
    return _find_conflict_internal(S)

def _is_valid_internal(S):
    n = len(S)
    if n < 4:
        return True
    sums_map = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = S[i] + S[j] + S[k]
                if s not in sums_map:
                    sums_map[s] = []
                sums_map[s].append((i, j, k))
    for a_idx, a in enumerate(S):
        max_multiple = 576 // a
        for k in range(1, max_multiple + 1):
            target = k * a
            if target in sums_map:
                for (i, j, kk) in sums_map[target]:
                    if a_idx != i and a_idx != j and a_idx != kk:
                        return False
    return True

def _find_conflict_internal(S):
    n = len(S)
    if n < 4:
        return None
    sums_map = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = S[i] + S[j] + S[k]
                if s not in sums_map:
                    sums_map[s] = []
                sums_map[s].append((i, j, k))
    for a_idx, a in enumerate(S):
        max_multiple = 576 // a
        for k in range(1, max_multiple + 1):
            target = k * a
            if target in sums_map:
                for (i, j, kk) in sums_map[target]:
                    if a_idx != i and a_idx != j and a_idx != kk:
                        return (a_idx, i, j, kk)
    return None
