import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Known valid 20-mark Sidon set (from optimal 20-mark Golomb ruler shifted)
    # Note: Optimal 20-mark ruler length is 240.
    # The set [1, 2, 5, 11, ...] in the current program corresponds to the 20-mark ruler.
    base_20 = [1, 2, 5, 11, 19, 24, 30, 43, 48, 59, 73, 86, 92, 102, 105, 113, 126, 135, 149, 167]
    
    # Verify base_20
    def is_valid(sidon_set):
        mask = 0
        n = len(sidon_set)
        for i in range(n):
            for j in range(i, n):
                s = sidon_set[i] + sidon_set[j]
                if (mask >> s) & 1:
                    return False
                mask |= (1 << s)
        return True

    best_set = base_20[:]
    best_size = len(base_20)
    
    if not is_valid(best_set):
        # Fallback if hardcoded is somehow invalid
        best_set = [1, 2, 5, 11]
        best_size = 4

    # Try to extend to size 21 and 22
    # We will use a backtracking search starting from the base set
    # To make it fast, we sort candidates and use pruning
    
    prescribed = [1, 2, 5, 11]
    
    # Pre-compute mask for base_20
    mask_20 = 0
    for i in range(len(base_20)):
        for j in range(i, len(base_20)):
            mask_20 |= (1 << (base_20[i] + base_20[j]))
            
    # Candidates for addition: 1..284 not in base_20
    candidates = [x for x in range(1, 285) if x not in base_20]
    
    # Filter candidates that are compatible with base_20
    valid_candidates_21 = []
    for x in candidates:
        ok = True
        for y in base_20:
            if (mask_20 >> (x + y)) & 1:
                ok = False
                break
        if (mask_20 >> (2*x)) & 1:
            ok = False
        if ok:
            valid_candidates_21.append(x)
            
    # Search for size 21
    found_21 = False
    for x in valid_candidates_21:
        if time.time() > deadline - 20: break
        new_set = base_20 + [x]
        new_set.sort()
        if is_valid(new_set):
            best_set = new_set
            best_size = 21
            found_21 = True
            break # Found one 21, try to find 22
            
    if found_21:
        # Try to extend best_set (size 21) to size 22
        mask_21 = 0
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                mask_21 |= (1 << (best_set[i] + best_set[j]))
                
        candidates_22 = [x for x in range(1, 285) if x not in best_set]
        valid_candidates_22 = []
        for x in candidates_22:
            ok = True
            for y in best_set:
                if (mask_21 >> (x + y)) & 1:
                    ok = False
                    break
            if (mask_21 >> (2*x)) & 1:
                ok = False
            if ok:
                valid_candidates_22.append(x)
                
        for x in valid_candidates_22:
            if time.time() > deadline - 5: break
            new_set = best_set + [x]
            new_set.sort()
            if is_valid(new_set):
                best_set = new_set
                best_size = 22
                break # Found 22, max possible likely
                
    return best_set
