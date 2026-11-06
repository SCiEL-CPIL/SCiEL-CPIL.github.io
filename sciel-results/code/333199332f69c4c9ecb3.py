"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(S, cand):
    """Check if cand can be added to S without violating constraints."""
    # Case 1: cand is 'a' (smallest)
    larger = [x for x in S if x > cand]
    for i in range(len(larger)):
        b = larger[i]
        for j in range(i, len(larger)):
            c = larger[j]
            if (b + c) % cand == 0:
                return False

    # Case 2: cand is 'b' (middle)
    smaller = [x for x in S if x < cand]
    geq = [x for x in S if x >= cand]
    for a in smaller:
        for c in geq:
            if (cand + c) % a == 0:
                return False

    # Case 3: cand is 'c' (largest)
    for a in S:
        if a >= cand:
            continue
        for b in S:
            if a < b <= cand and (b + cand) % a == 0:
                return False

    return True

def construct():
    deadline = time.time() + 85
    n = 256
    
    best_S = set()
    best_size = 0
    
    # Run multiple greedy passes with different orders
    # 1. Descending
    # 2. Ascending
    # 3. Random
    # 4. Random
    
    orders = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
        list(range(1, n + 1)),
        list(range(1, n + 1))
    ]
    
    for i in range(2, 4):
        random.shuffle(orders[i])
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S = set()
        for cand in order:
            if time.time() > deadline:
                break
            if _can_add(S, cand):
                S.add(cand)
                
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Local search: try to add any remaining
        added = True
        while added and time.time() < deadline:
            added = False
            for cand in range(1, n + 1):
                if time.time() > deadline:
                    break
                if cand not in S and _can_add(S, cand):
                    S.add(cand)
                    added = True
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Try a few random restarts with perturbation
    # Start from best_S, remove k random elements, re-greedy
    for _ in range(20):
        if time.time() > deadline:
            break
        
        S = set(best_S)
        # Remove 5 random elements
        if len(S) >= 5:
            to_remove = random.sample(list(S), 5)
            for x in to_remove:
                S.discard(x)
        else:
            continue
            
        # Re-greedy with random order
        remaining = [x for x in range(1, n + 1) if x not in S]
        random.shuffle(remaining)
        
        for cand in remaining:
            if time.time() > deadline:
                break
            if _can_add(S, cand):
                S.add(cand)
                
        # Final add pass
        added = True
        while added and time.time() < deadline:
            added = False
            for cand in range(1, n + 1):
                if time.time() > deadline:
                    break
                if cand not in S and _can_add(S, cand):
                    S.add(cand)
                    added = True
                    
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    return sorted(best_S)
# EVOLVE-BLOCK-END
