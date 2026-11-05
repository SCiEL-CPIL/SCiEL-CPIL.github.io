def construct():
    from math import isqrt
    import numpy as np

    # Generate all primes <= 10000
    primes = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            primes.append(p)
    
    n = len(primes)
    if n < 2:
        return primes

    max_gap = 10000
    
    # dp[i] is a dictionary: {gap: length}
    # parent[i] is a dictionary: {gap: (j, prev_gap)}
    dp = [{} for _ in range(n)]
    parent = [{} for _ in range(n)]
    
    # pref_val[i] is an array of size max_gap+1
    # pref_val[i][g] = max length of subseq ending at i with last gap <= g
    # pref_idx[i] is an array of size max_gap+1
    # pref_idx[i][g] = the specific gap <= g that achieved pref_val[i][g]
    
    # For i=0, no gaps
    pref_val = [None] * n
    pref_idx = [None] * n
    
    # Initialize i=0
    # No sequences ending at 0 with a "last gap" in the context of extending, 
    # but when extending TO i=1 from j=0, we treat j=0 as having length 1 and "gap" 0?
    # No, the base case is length 2.
    # If we pick j=0, i=1, gap g.
    # We look for best seq ending at 0 with last gap <= g.
    # There are no such sequences (length 0 or 1 doesn't have a "last gap" constraint in the same way, 
    # or rather, a single element has no gap).
    # So we treat "best ending at j" as 1 if we just start there?
    # Let's refine:
    # A subsequence of length 1 has no gap.
    # A subsequence of length 2 (p_j, p_i) has gap g.
    # If we extend a subsequence of length L ending at p_j with last gap g_prev, 
    # and add p_i with gap g_new, we need g_prev <= g_new.
    # New length is L + 1.
    
    # So, for j=0, there are no subsequences of length >= 2 ending at 0.
    # So pref_val[0] should be all -1 (or 0 if we consider length 1 as base?).
    # If pref_val[0][g] is 0, then candidate = 0 + 1 = 1? No, we want length 2.
    # Let's define:
    # dp[i][g] = max length of subseq ending at i with last gap exactly g.
    # Base: If we just pick p_j and p_i, length is 2.
    # This corresponds to extending a "virtual" sequence of length 1 at p_j with gap 0?
    # If we say virtual gap is 0, and 0 <= g_new, then 1 + 1 = 2.
    # So, let's set pref_val[j][g] = 1 for all g if j is the start?
    # No, that's messy.
    
    # Better:
    # If no previous gap exists, best is 1 (the element itself).
    # So, initialize pref_val[j] with 1s?
    # If pref_val[j][g] = 1, then candidate = 1 + 1 = 2. Correct.
    # And the "prev_gap" for the base case can be 0.
    
    # So:
    # For all j, pref_val[j][g] starts at 1?
    # But we only update dp[j] for gaps that actually exist.
    # If dp[j] is empty, it means no subsequence of length >= 2 ends at j.
    # But we can always start a new subsequence at j.
    # So effectively, for any j, the "best length with gap <= g" is at least 1.
    # And the "prev gap" associated with that length 1 is 0.
    
    # Let's implement this:
    # pref_val[j][g] = max(1, max_{g'<=g} dp[j][g'])
    # pref_idx[j][g] = 0 if the max is 1, else the g' that achieved the max.
    
    # Initialize pref_val and pref_idx for all j?
    # We can compute them on the fly or pre-initialize.
    # Since we iterate i from 1 to n-1, and need pref[j] for j < i,
    # we can compute pref[j] right after computing dp[j].
    
    # For j=0:
    # dp[0] is empty.
    # pref_val[0][g] = 1 for all g.
    # pref_idx[0][g] = 0 for all g.
    
    pref_val[0] = np.ones(max_gap + 1, dtype=np.int32)
    pref_idx[0] = np.zeros(max_gap + 1, dtype=np.int32)
    
    for i in range(1, n):
        pi = primes[i]
        for j in range(i):
            g = pi - primes[j]
            if g > max_gap:
                continue
            
            # Query pref[j]
            # pref_val[j][g] is the max length of subseq ending at j with last gap <= g
            # pref_idx[j][g] is the specific gap that achieved it
            
            best_len = pref_val[j][g]
            prev_gap = pref_idx[j][g]
            
            candidate_len = best_len + 1
            
            # Check if this candidate is better for dp[i][g]
            if g not in dp[i] or candidate_len > dp[i][g]:
                dp[i][g] = candidate_len
                parent[i][g] = (j, prev_gap)
        
        # Build pref_val[i] and pref_idx[i] from dp[i]
        # Initialize with base case: length 1, gap 0
        pref_val[i] = np.ones(max_gap + 1, dtype=np.int32)
        pref_idx[i] = np.zeros(max_gap + 1, dtype=np.int32)
        
        # Update with actual dp entries
        for g, length in dp[i].items():
            if g <= max_gap:
                if length > pref_val[i][g]:
                    pref_val[i][g] = length
                    pref_idx[i][g] = g
        
        # Prefix max pass
        # We need to propagate the max and the corresponding index
        # Note: if multiple gaps have the same max value, any is fine, 
        # but we must ensure consistency for backtracking.
        # Standard prefix max:
        # if A[g] > A[g-1], A[g] stays, idx[g] stays.
        # if A[g] <= A[g-1], A[g] = A[g-1], idx[g] = idx[g-1].
        
        for g in range(1, max_gap + 1):
            if pref_val[i][g-1] > pref_val[i][g]:
                pref_val[i][g] = pref_val[i][g-1]
                pref_idx[i][g] = pref_idx[i][g-1]
            # If equal, we keep the current one (larger gap? smaller gap? doesn't matter for validity, 
            # but for backtracking, we need to be sure the parent exists. 
            # Since both are valid, keeping the one from the array (which might be 0 if not set) 
            # or the previous one is fine. 
            # Actually, if equal, the previous one (smaller gap) is "safer" in terms of constraints? 
            # No, the constraint is <=. So any gap <= g is valid.
            # Let's just keep the one that achieved the max.
            # If pref_val[i][g] == pref_val[i][g-1], we can keep pref_idx[i][g] if it was set, 
            # or take from g-1. 
            # My code above: if >, update. If <=, do nothing? 
            # Wait, if pref_val[i][g] is 1 (base) and pref_val[i][g-1] is 5, 
            # then 5 > 1, so we update. 
            # If pref_val[i][g] is 5 and pref_val[i][g-1] is 5, 
            # we keep g. This is fine.
            
    # Find global max
    max_len = 0
    best_i = -1
    best_g = -1
    
    for i in range(n):
        for g, length in dp[i].items():
            if length > max_len:
                max_len = length
                best_i = i
                best_g = g
                
    # Reconstruct
    if best_i == -1:
        return primes[:2] # Fallback
        
    seq = []
    cur_i = best_i
    cur_g = best_g
    
    while cur_i != -1:
        seq.append(primes[cur_i])
        if cur_i == 0:
            break
        j, prev_g = parent[cur_i][cur_g]
        cur_i = j
        cur_g = prev_g
        
    seq.reverse()
    return seq
