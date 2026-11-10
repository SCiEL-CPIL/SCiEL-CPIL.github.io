import time
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    def valid_set(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if lcm(S[i], S[j]) <= 256:
                    return False
        return True

    def score(S):
        return sum(1.0/x for x in S)

    best_score = 0.0
    best_set = []
    
    # Try anchors
    anchors = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for p in anchors:
        if time.time() > deadline:
            break
            
        # Build candidate set compatible with p
        candidates = []
        for b in range(2, 257):
            if b == p:
                continue
            if lcm(p, b) > 256:
                candidates.append(b)
        
        # Add p to the set
        # We need to find a subset of candidates that is mutually compatible
        # and also compatible with p (already ensured)
        
        # Greedy initial solution
        current_set = [p]
        # Sort candidates by weight (1/b) descending, i.e., b ascending
        candidates.sort()
        
        for b in candidates:
            if all(lcm(b, y) > 256 for y in current_set):
                current_set.append(b)
        
        if valid_set(current_set):
            s = score(current_set)
            if s > best_score:
                best_score = s
                best_set = current_set[:]

        # Local search (1-swap)
        # Try to swap out a low weight element for a high weight one?
        # Or just try to add/remove?
        # Since we want to maximize sum, and 1/b is decreasing,
        # we want to include small b's if possible.
        
        # Simple hill climbing: try to add any candidate not in set
        # then try to swap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            for b in candidates:
                if b not in current_set:
                    if all(lcm(b, y) > 256 for y in current_set):
                        current_set.append(b)
                        improved = True
                        break
            if improved:
                continue
                
            # Try to swap: remove x, add y
            # This is expensive.
            # Instead, try to remove the largest element (smallest weight)
            # and see if we can add two smaller ones?
            # Or just restart with different order?
            
            # For now, just break and try next anchor
            break

    # Also try a generic greedy without anchor
    generic_set = []
    for x in range(2, 257):
        if all(lcm(x, y) > 256 for y in generic_set):
            generic_set.append(x)
    if valid_set(generic_set):
        s = score(generic_set)
        if s > best_score:
            best_score = s
            best_set = generic_set[:]

    return best_set
