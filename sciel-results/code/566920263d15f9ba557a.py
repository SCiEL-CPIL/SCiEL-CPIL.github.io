import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 256
    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            s = i + j
            p = i * j
            if p % s == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Initialize set with all odds
    S_mask = 0
    for i in range(1, N + 1, 2):
        S_mask |= (1 << i)
    
    # Helper: count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Helper: check if v can be added given current S_mask
    # v is addable if no element in S_mask conflicts with v
    def can_add_mask(v, smask):
        return (conflict_mask[v] & smask) == 0
    
    # Greedy add: repeatedly add any element that has no conflicts
    changed = True
    while changed:
        changed = False
        for v in range(1, N + 1):
            if (S_mask >> v) & 1:
                continue
            if can_add_mask(v, S_mask):
                S_mask |= (1 << v)
                changed = True
    
    best_size = popcount(S_mask)
    best_mask = S_mask
    
    # Local search with bitmask operations
    # For each u in S, find v not in S where u is the only conflict
    # i.e., (conflict_mask[v] & S_mask) == (1 << u)
    
    while time.time() < deadline:
        improved = False
        
        # Get list of elements in S for iteration
        S_list = [v for v in range(1, N + 1) if (S_mask >> v) & 1]
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline:
                break
            
            # Find all v not in S where u is the ONLY conflict
            # (conflict_mask[v] & S_mask) == (1 << u)
            cands = []
            for v in range(1, N + 1):
                if (S_mask >> v) & 1:
                    continue
                if (conflict_mask[v] & S_mask) == (1 << u):
                    cands.append(v)
            
            if len(cands) >= 2:
                # Try to find 2 non-conflicting candidates
                found = False
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        v1, v2 = cands[i], cands[j]
                        if not (conflict_mask[v1] >> v2) & 1:
                            # Valid 1-for-2 move
                            S_mask &= ~(1 << u)
                            S_mask |= (1 << v1) | (1 << v2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    if popcount(S_mask) > best_size:
                        best_size = popcount(S_mask)
                        best_mask = S_mask
                    break  # Restart with new S
            
            # Also try 2-for-1: remove u and some w from S, add one v
            # This is harder; skip for now to save time
        
        if not improved:
            # Try random 1-for-1 swaps to escape local minima
            if time.time() < deadline - 10:
                # Pick random u in S, random v not in S with exactly 1 conflict
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    u = random.choice(S_list)
                    # Find v with exactly one conflict = u
                    cands = [v for v in range(1, N + 1) 
                             if not ((S_mask >> v) & 1) 
                             and (conflict_mask[v] & S_mask) == (1 << u)]
                    if cands:
                        v = random.choice(cands)
                        S_mask &= ~(1 << u)
                        S_mask |= (1 << v)
                        # Check if this enables a 1-for-2
                        # Quick check: for each w in new S, see if any v2 has only w as conflict
                        # This is expensive; just accept the swap and let next iteration find improvements
                        break
    
    # Return the best mask found
    result = [v for v in range(1, N + 1) if (best_mask >> v) & 1]
    if not result:
        result = [1]
    return result

if __name__ == "__main__":
    print(construct())
