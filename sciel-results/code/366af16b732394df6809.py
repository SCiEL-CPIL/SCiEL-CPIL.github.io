import time

def construct():
    start_time = time.time()
    limit = 85.0  # Safety margin before 90s limit
    
    N = 12
    period = 27720  # lcm(1..12)
    
    # Precompute residues for speed?
    # For a given m and L, we need S_i.
    # d in 1..L such that (m+d) % i == 0
    # d % i == (-m) % i
    # Let r_i = (-m) % i.
    # Valid d are r_i, r_i+i, ... <= L. Note: if r_i == 0, d starts at i? 
    # Wait. d must be >= 1.
    # If r_i == 0, d must be a multiple of i. Smallest positive is i.
    # If r_i != 0, d must be congruent to r_i mod i. Smallest positive is r_i.
    
    def get_candidates(m, L):
        # Returns list of lists: cands[i] contains valid d for index i+1
        cands = []
        for i in range(1, N+1):
            r = (-m) % i
            start = r if r != 0 else i
            # Generate d = start, start+i, ... <= L
            # Using range
            cands.append(list(range(start, L+1, i)))
        return cands

    def can_match(m, L):
        cands = get_candidates(m, L)
        
        # Check if any candidate list is empty
        for c in cands:
            if not c:
                return False
        
        # Sort indices by length of candidate list (MRV)
        indices = list(range(N))
        indices.sort(key=lambda idx: len(cands[idx]))
        
        used = [False] * (L + 1) # 1-indexed for d
        
        def backtrack(k):
            if k == N:
                return True
            i = indices[k]
            for d in cands[i]:
                if not used[d]:
                    used[d] = True
                    if backtrack(k + 1):
                        return True
                    used[d] = False
            return False
        
        return backtrack(0)

    best_m = 0
    best_f = 0
    
    # Binary search for minimal L for each m
    # f(m) is in [1, 144]
    
    for m in range(period):
        if m % 1000 == 0:
            if time.time() - start_time > limit:
                break
        
        # Binary search for minimal L
        lo = 1
        hi = 144
        # We want the smallest L such that can_match(m, L) is True
        # can_match is monotonic: if True for L, True for L+1
        
        # Optimization: If we already have a best_f, we can start lo from max(1, best_f) ?
        # No, we need the exact minimal L to compare.
        # But if minimal L <= best_f, we don't care about the exact value, we just know it's not better.
        # So:
        # 1. Check if can_match(m, best_f) is True.
        #    If True, then minimal L <= best_f. We cannot improve. Skip.
        # 2. If False, then minimal L > best_f.
        #    Binary search in [best_f+1, 144].
        
        if best_f > 0:
            if can_match(m, best_f):
                # Minimal L is <= best_f, so no improvement
                continue
        
        # Now we know minimal L > best_f (or best_f is 0)
        lo = best_f + 1 if best_f > 0 else 1
        hi = 144
        
        # If lo > hi, no solution? But problem says max 144.
        # If can_match(m, 144) is False, then m is invalid? 
        # Problem states "Maximize this independently computed L... at most 144".
        # It implies a solution always exists within 144.
        
        # Binary search
        min_L = 145 # Default
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L < 145:
            if min_L > best_f:
                best_f = min_L
                best_m = m
                # If we reached the theoretical max, we can stop
                if best_f == 144:
                    break

    return best_m
