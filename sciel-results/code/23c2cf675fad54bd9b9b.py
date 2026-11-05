import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0

    NEED = set(range(1, 31))
    MAXN = 32

    def is_valid(a):
        if 0 not in a:
            return False
        a = sorted(set(a))
        if len(a) > MAXN:
            return False
        diffs = set()
        for i in range(len(a)):
            ai = a[i]
            for j in range(i + 1, len(a)):
                d = a[j] - ai
                if d in diffs:
                    return False
                diffs.add(d)
        return NEED.issubset(diffs)

    def find_solution(M):
        """
        Find a Sidon set A subset of [0, M] containing 0, size <= 32,
        such that all differences 1..30 are present.
        Returns the set A or None.
        """
        A = [0]
        diffs = set()
        missing = set(NEED)
        
        # We will build A in increasing order.
        # next_val: the minimum value for the next element.
        # We start with 0. Next element must be > 0.
        
        def backtrack(start_val):
            # Check if we are done
            if not missing:
                return list(A)
            
            # Check if we have too many elements
            if len(A) > MAXN:
                return None
            
            # Pruning: Can we cover the largest missing difference?
            # The max possible difference involving future elements is M - min(A).
            # Since min(A) is 0, max diff is M.
            # If max(missing) > M, return None (but M is fixed, so this is constant check)
            
            # Heuristic: Try to pick the next element x.
            # x must be >= start_val and <= M.
            # To be efficient, we can restrict x to values that help cover missing diffs.
            # A value x is "useful" if x - a is in missing for some a in A, 
            # OR if it allows covering a diff later.
            # However, simply iterating x from M down to start_val might be slow.
            # Iterating x from start_val to M is also slow.
            
            # Let's try candidates that directly cover a missing difference.
            # For each missing d, and each a in A:
            #   candidate x1 = a + d
            #   candidate x2 = a - d (must be > 0 and not in A)
            # Also, we might need to add "filler" elements?
            # If we only add elements that cover a missing diff, we might get stuck
            # if the remaining missing diffs cannot be covered by the remaining slots.
            
            # A safer approach: Iterate x from M down to start_val.
            # This tends to find solutions with large elements first, 
            # but since we fixed M, any solution is valid for this M.
            # However, we want to find *a* solution quickly.
            
            # Optimization: 
            # If we have k elements left, we can generate at most k*(k-1)/2 + k*len(A) new differences.
            # This is an upper bound on new diffs.
            
            # Let's try a smart ordering: 
            # Candidates are x in [start_val, M].
            # Score candidates by how many missing diffs they cover.
            # x covers d if d = x - a or d = a - x.
            
            candidates = []
            for x in range(start_val, M + 1):
                if x in A:
                    continue
                # Check Sidon property with current A
                ok = True
                new_diffs = set()
                for a in A:
                    d = x - a
                    if d <= 0: continue # x > a since x >= start_val > a? 
                    # Wait, A is sorted. x >= start_val. 
                    # If start_val > max(A), then x > all a.
                    # If start_val <= max(A), we must ensure x is not in A.
                    # The differences are x - a.
                    if d in diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                # Count how many missing diffs are covered
                covered = len(missing.intersection(new_diffs))
                
                # If covered == 0, we might still add it to enable future diffs,
                # but it's less efficient. We'll prioritize covered > 0.
                candidates.append((covered, x))
            
            # Sort candidates by coverage (descending), then by value (descending? ascending?)
            # Descending value might be better to "use up" the range?
            # Or ascending to keep numbers small?
            # Since M is fixed, it doesn't matter for validity, but might for speed.
            # Let's try descending coverage, then descending x.
            candidates.sort(key=lambda t: (-t[0], -t[1]))
            
            for covered, x in candidates:
                if covered == 0 and len(candidates) > 10:
                    # If we have many candidates with 0 coverage, 
                    # maybe skip them? Or just try a few?
                    # If we don't cover any missing diff now, we rely on future pairs.
                    # This is risky. Let's only try 0-coverage candidates if no other option.
                    pass
                
                # Add x
                A.append(x)
                for a in A[:-1]: # all except x
                    d = x - a
                    if d > 0:
                        diffs.add(d)
                        if d in missing:
                            missing.discard(d)
                
                res = backtrack(x + 1)
                if res is not None:
                    return res
                
                # Undo
                A.pop()
                # Remove diffs added by x
                for a in A:
                    d = x - a
                    if d > 0:
                        diffs.discard(d)
                        # Note: if d was in missing, it was discarded. 
                        # We need to restore it if it's in NEED and not in diffs anymore.
                        # This is tricky. 
                        # Better to pass diffs and missing as parameters or recreate them?
                        # Recreating is O(N^2). N is small (<=32). 
                        # Let's just copy them.
                        
            return None

        # To avoid complex undo logic, we'll pass copies or use a simpler state.
        # Given N <= 32, copying sets is cheap.
        
        def backtrack_v2(start_val, A, diffs, missing):
            if not missing:
                return list(A)
            if len(A) > MAXN:
                return None
            
            # Pruning: Max possible diff is M. If max(missing) > M, fail.
            if missing and max(missing) > M:
                return None
            
            # Generate candidates
            candidates = []
            for x in range(start_val, M + 1):
                if x in A:
                    continue
                ok = True
                new_diffs = set()
                for a in A:
                    d = x - a
                    if d <= 0:
                        continue
                    if d in diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                covered = len(missing.intersection(new_diffs))
                candidates.append((covered, x))
            
            if not candidates:
                return None
            
            # Sort: High coverage first.
            candidates.sort(key=lambda t: (-t[0], -t[1]))
            
            # Limit candidates to avoid explosion if many 0-coverage
            # If we have candidates with coverage > 0, prefer them.
            # If only 0-coverage, try a few.
            
            for covered, x in candidates:
                new_A = A + [x]
                new_diffs = set(diffs)
                new_missing = set(missing)
                
                for a in A:
                    d = x - a
                    if d > 0:
                        new_diffs.add(d)
                        if d in new_missing:
                            new_missing.discard(d)
                
                res = backtrack_v2(x + 1, new_A, new_diffs, new_missing)
                if res is not None:
                    return res
                    
            return None

        return backtrack_v2(1, [0], set(), set(NEED))

    # Binary search for minimal M
    # Lower bound: 30
    # Upper bound: Start with 200, increase if needed
    lo = 30
    hi = 200
    
    # First, ensure hi is valid
    while True:
        if time.time() > deadline - 5:
            break
        sol = find_solution(hi)
        if sol is not None:
            break
        hi *= 2
        if hi > 10000:
            # Fallback to greedy
            return greedy_fallback()
            
    # Now find minimal M in [lo, hi]
    best_sol = None
    best_max = hi
    
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        sol = find_solution(mid)
        if sol is not None:
            best_sol = sol
            best_max = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_sol is not None:
        # Optional: Local search to reduce max further? 
        # The binary search finds the minimal M such that a solution exists in [0, M].
        # This implies the minimal possible max element is <= M.
        # Actually, if a solution exists in [0, M], its max is <= M.
        # If no solution exists in [0, M-1], then any valid set must have max >= M.
        # So the minimal max is exactly M (or possibly less if the set found in [0,M] has max < M, 
        # but that would imply a solution in [0, max] < M, contradicting the binary search).
        # Wait. If find_solution(M) returns a set with max 50, and M=60.
        # Then find_solution(50) should have found it? 
        # Not necessarily, because find_solution(50) restricts elements to [0, 50].
        # If the set found for M=60 has max 50, it is a valid set in [0, 50].
        # So find_solution(50) should return it (or another).
        # So the binary search will converge to the true minimal max.
        
        return best_sol

    # Fallback
    return greedy_fallback()

def greedy_fallback():
    NEED = set(range(1, 31))
    A = [0]
    diffs = set()
    for x in range(1, 10000):
        ok = True
        new_diffs = set()
        for a in A:
            d = x - a
            if d in diffs:
                ok = False
                break
            new_diffs.add(d)
        if not ok:
            continue
        A.append(x)
        diffs = diffs.union(new_diffs)
        if NEED.issubset(diffs):
            return A
    return A
