import time
import sys

def construct():
    start = time.time()
    DEADLINE = 140.0  # Safety margin
    
    TARGET_DIFFS = set(range(1, 31))
    MAX_SIZE = 32
    FULL_MASK = (1 << 30) - 1
    
    def validate(S):
        if not S or 0 not in S:
            return False
        if len(S) > MAX_SIZE:
            return False
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if not TARGET_DIFFS.issubset(diffs):
            return False
        return True

    def dfs_limited(S, last, used_diffs, covered_mask, max_val):
        if time.time() > DEADLINE:
            return None
        
        if covered_mask == FULL_MASK:
            return list(S)
        
        if len(S) >= MAX_SIZE:
            return None
        
        if last >= max_val:
            return None
        
        # Heuristic: Try x from last+1 to max_val
        # To improve speed, we can check if it's possible to cover remaining diffs.
        # Remaining diffs count:
        missing_count = 30 - bin(covered_mask).count('1')
        # Max new diffs we can get from adding one element x to set of size k is k.
        # If missing_count > (max_val - last) * len(S) / 2? No, simple bound:
        # Each new element adds at most len(S) new differences.
        # If missing_count > (MAX_SIZE - len(S)) * len(S):
        #    return None
        
        for x in range(last + 1, max_val + 1):
            # Check Sidon property for adding x
            valid = True
            for a in S:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate new mask
            new_mask = covered_mask
            for a in S:
                d = x - a
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
            
            # Recurse
            S.append(x)
            for a in S[:-1]:
                used_diffs.add(x - a)
            
            res = dfs_limited(S, x, used_diffs, new_mask, max_val)
            
            if res is not None:
                return res
            
            # Backtrack
            S.pop()
            for a in S:
                used_diffs.discard(x - a)
        
        return None

    # Iterative deepening on the maximum element D
    # Lower bound: We need 30 differences. 
    # A Sidon set of size k has at most k(k-1)/2 differences.
    # 30 differences requires k(k-1)/2 >= 30 => k(k-1) >= 60 => k >= 9.
    # So we need at least 9 elements.
    # Diameter for size 9 Sidon set?
    # Start D from 30.
    
    for D in range(30, 300):
        if time.time() > DEADLINE - 5:
            break
        
        res = dfs_limited([0], 0, set(), 0, D)
        if res is not None:
            if validate(res):
                return res
            else:
                # Should not happen
                continue
                
    # If no solution found in iterative deepening, try a wider search
    # This should rarely happen if D goes up to 300.
    res = dfs_limited([0], 0, set(), 0, 1000)
    if res is not None:
        if validate(res):
            return res
            
    # Absolute fallback: Return a known valid set if we can find one quickly
    # Or return [0] which is invalid.
    # To avoid 0 score, we must return a valid set.
    # Let's try a greedy construction that doesn't limit D, but stops when full.
    
    def greedy_search():
        S = [0]
        used_diffs = set()
        covered_mask = 0
        last = 0
        while covered_mask != FULL_MASK and len(S) < MAX_SIZE:
            found = False
            # Try x from last+1 to last+100
            for x in range(last + 1, last + 100):
                valid = True
                for a in S:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_mask = covered_mask
                for a in S:
                    d = x - a
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d-1))
                
                # Accept if it adds new coverage or just to keep going?
                # We want to minimize diameter, so we pick the smallest valid x.
                # But we must ensure we *can* finish.
                # For now, just pick smallest valid x.
                
                S.append(x)
                for a in S[:-1]:
                    used_diffs.add(x - a)
                last = x
                covered_mask = new_mask
                found = True
                break
            
            if not found:
                # Stuck. Try a larger jump?
                # For fallback, we just return what we have if it's valid? No, must cover 1..30.
                # If stuck, we fail.
                return None
                
        if covered_mask == FULL_MASK:
            return S
        return None

    res = greedy_search()
    if res and validate(res):
        return res
        
    # If all fails, return a hardcoded valid set.
    # I will generate one now.
    # Using a known Sidon set from OEIS A005282 or similar.
    # {0, 1, 4, 10, 18, 23, 25, 29} was invalid.
    # Let's use a larger one.
    # {0, 1, 4, 10, 18, 23, 28, 34, 41, 49}
    # Diffs:
    # 1
    # 4,3
    # 10,9,6
    # 18,17,14,8
    # 23,22,19,13,5
    # 28,27,24,18(DUP 18 from 18-0), ...
    
    # Let's just return the result of the last dfs call if it exists, even if not optimal?
    # No, must be valid.
    
    # If we get here, we failed.
    return [0]
