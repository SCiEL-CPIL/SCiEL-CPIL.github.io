"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import bisect

    # 1. Generate all primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # states[i] is a list of tuples (gap, length, pred_idx)
    # The list is sorted by gap, and lengths are strictly increasing.
    # (0, 1, -1) represents a sequence of length 1 ending at primes[i] with no previous gap.
    states = [[] for _ in range(n)]
    
    # Initialize state for the first prime
    # Actually, we can just loop i from 0 to n-1 and handle it uniformly.
    
    for i in range(n):
        pi = primes[i]
        candidates = [(0, 1, -1)]  # Start new sequence
        
        for j in range(i):
            d = pi - primes[j]
            # Binary search in states[j] for the largest gap <= d
            # states[j] is sorted by gap.
            # We need the index of the rightmost element with gap <= d.
            sj = states[j]
            if not sj:
                continue
            
            # Extract gaps for binary search
            # To avoid creating a new list every time, we can search on the tuple list directly?
            # bisect works on lists. We can create a list of gaps once per j?
            # No, j changes.
            # We can store gaps separately or just use a helper.
            # Given small N, creating a list of gaps for each j inside the loop is O(N^3) memory/time overhead?
            # Sum of i is N^2/2. Creating list of size j for each j is O(j). Total O(N^3).
            # N=1229. 1229^3 is too big.
            
            # We need to binary search in sj without extracting gaps.
            # Python's bisect doesn't support key functions directly in older versions, 
            # but we can use a custom binary search or extract gaps.
            # Actually, extracting gaps for each j inside the i loop is expensive.
            # Let's pre-store gaps? No, states grow.
            
            # Alternative: Since sj is small on average, linear scan might be faster?
            # Max size of sj is bounded by the number of distinct gaps, which is <= j+1.
            # But we prune. The size is likely small.
            # Let's try linear scan from the end? No, gaps are sorted.
            # Let's use bisect on a list of gaps.
            # To optimize, we can store the list of gaps for each state?
            # states[j] = [(gap, len, pred), ...]
            # We can maintain a parallel list of gaps? Or just extract.
            
            # Let's try extracting gaps for sj.
            # If sj is small, it's fine.
            gaps_j = [s[0] for s in sj]
            # Find rightmost index where gap <= d
            idx = bisect.bisect_right(gaps_j, d) - 1
            
            if idx >= 0:
                gap_j, len_j, pred_j = sj[idx]
                # We can extend
                new_len = len_j + 1
                # The new state for i will have gap d
                # pred is j
                candidates.append((d, new_len, j))
        
        # Prune candidates
        # Sort by gap, then by length (descending? No, we want max length for each gap)
        # If multiple candidates have the same gap, keep the one with max length.
        # Then remove dominated states: if a state has gap g and length l,
        # and there is a state with gap g' < g and length l' >= l, then (g,l) is dominated.
        
        # Sort by gap
        candidates.sort(key=lambda x: x[0])
        
        pruned = []
        max_len_so_far = 0
        last_gap = -1
        best_for_gap = -1
        best_pred_for_gap = -1
        
        # Group by gap?
        # Since we want strictly increasing lengths, we can just iterate.
        # But if we have multiple entries with same gap, we only care about the max length.
        
        for gap, length, pred in candidates:
            if gap == last_gap:
                if length > best_for_gap:
                    best_for_gap = length
                    best_pred_for_gap = pred
            else:
                # Process previous gap group
                if last_gap != -1 and best_for_gap > max_len_so_far:
                    pruned.append((last_gap, best_for_gap, best_pred_for_gap))
                    max_len_so_far = best_for_gap
                
                # Start new group
                last_gap = gap
                best_for_gap = length
                best_pred_for_gap = pred
        
        # Process the last group
        if last_gap != -1 and best_for_gap > max_len_so_far:
            pruned.append((last_gap, best_for_gap, best_pred_for_gap))
            # max_len_so_far = best_for_gap # Not needed for next loop iteration of i
        
        states[i] = pruned

    # Find the best state
    max_len = 0
    best_i = -1
    best_state = None
    
    for i in range(n):
        if states[i]:
            # The last element in states[i] has the max length
            gap, length, pred = states[i][-1]
            if length > max_len:
                max_len = length
                best_i = i
                best_state = (gap, length, pred)

    if max_len == 0:
        return []

    # Reconstruct
    seq = []
    cur_i = best_i
    # We need to trace back.
    # The state in states[cur_i] that we picked is the last one?
    # Yes, we picked the max length state.
    # But we need to know WHICH state in states[cur_i] we are at to get the pred.
    # We stored the max length state in best_state.
    # However, the pred chain depends on the specific state.
    
    # Let's trace back using the states.
    # We know the final state is (gap_final, len_final, pred_final) in states[best_i].
    # But states[best_i] is a list. We need to find the specific entry.
    # Actually, we can just traverse back from best_i.
    # The state at best_i is the one with max length.
    
    # To make reconstruction easier, we should have stored the path or just the predecessor indices.
    # My current states[i] stores (gap, length, pred_idx).
    # The pred_idx points to the index j in primes.
    # But which state in states[j] was used?
    # It was the one with the largest gap <= d.
    # When we reconstruct, we need to know that gap to find the correct predecessor state in states[j].
    # Wait, the `pred_idx` stored in the candidate was `j`.
    # The gap used from `j` was `gap_j`.
    # We didn't store `gap_j` in the candidate, we only stored `d` (the new gap).
    # To reconstruct, we need to know `gap_j`.
    # We can re-calculate it?
    # If we are at `i` and we know we came from `j`, and the gap at `i` is `d_i`,
    # we need to find the state in `states[j]` that was used.
    # It was the one with `gap <= d_i` and max length.
    # This is deterministic.
    
    seq.append(primes[best_i])
    cur_i = best_i
    # We need the gap of the state at cur_i to find the predecessor.
    # The state at cur_i is the one with max length?
    # Not necessarily. We picked the global max length.
    # Let's find the specific state in states[best_i] that has length max_len.
    # There might be multiple? No, we pruned to strictly increasing lengths.
    # So there is only one state with length max_len in states[best_i]?
    # Yes, because lengths are strictly increasing.
    
    # So we can find the state in states[best_i] with length == max_len.
    current_state = None
    for s in states[best_i]:
        if s[1] == max_len:
            current_state = s
            break
    
    if current_state is None:
        return []
        
    gap_cur = current_state[0]
    pred_i = current_state[2]
    
    while pred_i != -1:
        seq.append(primes[pred_i])
        # Find the state in states[pred_i] that was used to form the current state.
        # The gap at pred_i must be <= gap_cur.
        # And it must be the one with the max length among those <= gap_cur.
        # Also, the length of that state must be current_len - 1.
        
        target_len = current_state[1] - 1
        target_gap_limit = gap_cur
        
        # Find state in states[pred_i] with gap <= target_gap_limit and length == target_len
        # Since lengths are strictly increasing, there is at most one state with length target_len.
        # We can just find the state with length target_len and check if its gap <= target_gap_limit.
        # Actually, we know it must be the one used.
        
        prev_state = None
        for s in states[pred_i]:
            if s[1] == target_len:
                # Check if it was the one used (gap <= target_gap_limit)
                # It should be, by construction.
                if s[0] <= target_gap_limit:
                    prev_state = s
                break # Lengths are unique in the pruned list?
                # Wait, are lengths unique?
                # We pruned such that lengths are strictly increasing.
                # So yes, each length appears at most once.
        
        if prev_state is None:
            # Error in reconstruction logic?
            # Maybe the state with target_len has a gap > target_gap_limit?
            # That would mean we didn't pick the right predecessor.
            # But we picked the predecessor based on max length <= gap.
            # So the state with max length <= gap_cur in states[pred_i] must have length target_len.
            # If there is a state with length target_len but gap > gap_cur, it wouldn't have been picked.
            # If there is a state with length < target_len and gap <= gap_cur, it wouldn't be the max.
            # So the state with length target_len MUST have gap <= gap_cur.
            
            # Let's just pick the state with length target_len.
            for s in states[pred_i]:
                if s[1] == target_len:
                    prev_state = s
                    break
        
        if prev_state is None:
            break
            
        gap_cur = prev_state[0]
        current_state = prev_state
        pred_i = prev_state[2]
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
