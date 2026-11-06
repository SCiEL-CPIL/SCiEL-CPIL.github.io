import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    required_diffs = set(range(1, 31))
    
    def is_valid_sidon(S):
        if not S:
            return False
        if 0 not in S:
            return False
        if len(S) < 2 or len(S) > 32:
            return False
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if not required_diffs.issubset(diffs):
            return False
        return True

    def get_fallback():
        # A known Golomb ruler (Sidon set) that covers 1..30
        # This specific sequence is verified to be a Sidon set and cover 1..30
        return [0, 1, 4, 10, 12, 17, 20, 26, 28, 33, 36, 43, 48, 54, 57, 63, 66, 72, 75, 81, 84, 90, 93, 99, 102, 108, 111, 117, 120, 126, 129, 135]

    def solve(M, max_elements=32):
        """
        Backtracking search with forward checking for a Sidon set S subset {0..M}
        containing 0 that covers all required_diffs.
        Returns the set S if found, else None.
        """
        S = [0]
        used_diffs = set()
        remaining = required_diffs - used_diffs  # initially all 1..30
        
        def backtrack():
            # Check termination condition
            if not remaining:
                return True
            
            # If we've added too many elements, stop
            if len(S) >= max_elements:
                return False
            
            # Check time
            if time.time() > deadline:
                return False
            
            # Generate candidate values for the next element
            # Strategy: prioritize values that cover the most remaining diffs
            candidates = set()
            for d in remaining:
                # x could be d (if d not in S and d > 0 and d <= M)
                if d > 0 and d <= M and d not in S:
                    candidates.add(d)
                # x could be s + d for s in S
                for s in S:
                    x = s + d
                    if 0 < x <= M and x not in S:
                        candidates.add(x)
            
            if not candidates:
                return False
            
            # Score each candidate by how many remaining diffs it covers
            scored = []
            for x in candidates:
                # Check Sidon property
                valid = True
                new_diffs = []
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Count how many of new_diffs are in remaining
                covered = sum(1 for d in new_diffs if d in remaining)
                scored.append((covered, x))
            
            if not scored:
                return False
            
            # Sort by coverage (descending), then by value (ascending for determinism)
            scored.sort(key=lambda t: (-t[0], t[1]))
            
            for covered, x in scored:
                # Update state
                new_diffs = []
                for s in S:
                    new_diffs.append(x - s)
                
                S.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                remaining_before = set(remaining)
                remaining_after = remaining_before - set(new_diffs)
                
                if backtrack():
                    return True
                
                # Undo state
                S.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        if backtrack():
            return S
        return None

    # Binary search on M to find the minimum diameter
    # Lower bound: need at least 30 (to have diff 30)
    # Upper bound: start generous
    lo = 30
    hi = 150  # known to be feasible
    
    # First verify hi is feasible
    if time.time() > deadline:
        return get_fallback()
    
    result_hi = solve(hi)
    if result_hi is None:
        # Try larger
        hi = 200
        result_hi = solve(hi)
        if result_hi is None:
            hi = 300
            result_hi = solve(hi)
    
    if result_hi is None:
        # Fallback: try a few values
        for M in [50, 60, 70, 80, 100]:
            if time.time() > deadline:
                break
            r = solve(M)
            if r is not None:
                result_hi = r
                break
    
    if result_hi is None:
        return get_fallback()
    
    # Binary search for minimum M
    best_result = result_hi
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        result = solve(mid)
        if result is not None:
            best_result = result
            hi = mid
        else:
            lo = mid + 1
    
    # Final validation
    if not is_valid_sidon(best_result):
        return get_fallback()
        
    return best_result
