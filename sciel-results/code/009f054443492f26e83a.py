import time
import sys
import random

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _compute_sums(values):
    """Compute all consecutive block sums for a sequence."""
    n = len(values)
    sums = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

def _check_extension(current, x):
    """
    Check if appending x to current keeps all sums distinct.
    Returns True if valid, False otherwise.
    """
    if not current:
        return True
    
    # New sums introduced by x:
    # 1. x itself
    # 2. current[-1] + x
    # 3. current[-2] + current[-1] + x
    # ...
    # k. sum(current) + x
    
    # We need to check if any of these new sums are already in the existing sums.
    # Also, we don't need to check if the new sums collide with each other?
    # Let's verify:
    # New sums are S_k = sum(current[-k:]) + x for k=1..len(current)
    # S_k - S_{k-1} = current[-k] > 0 (since strictly increasing)
    # So S_1 < S_2 < ... < S_n. They are distinct from each other.
    # We only need to check against existing sums.
    
    existing_sums = _compute_sums(current)
    
    # Optimization: We can just compute the new tails and check intersection.
    tails = [x]
    total = x
    for a in reversed(current):
        total += a
        tails.append(total)
    
    return not existing_sums.intersection(tails)

def _construct_cp_search(max_len, max_val=256, deadline=None):
    """
    Constraint Propagation Search.
    We model the sequence as a list of length max_len.
    We fill positions one by one.
    Domain for position i is [prev_val + 1, max_val - (max_len - i - 1)]
    (ensuring we can still fit remaining elements strictly increasing).
    
    Heuristic: Choose the next position (always the next empty one, since sequence order matters)
    and try values from the domain.
    
    To make it "CP" style, we can pre-check feasibility.
    """
    best_seq = []
    
    # Pre-compute valid extensions for small prefixes to speed up?
    # No, let's just do DFS with strong pruning.
    
    # Since we must strictly increase, and max_val is 256, the maximum possible length is small.
    # Let's estimate max length.
    # If we pick 1, 2, 4, 8... length is ~8.
    # If we pick 1, 3, 6... length is larger.
    # The current best is around 10-12.
    
    # We will try to build sequences of length L from high to low.
    # Start with a reasonable upper bound.
    # Max possible length:
    # Min sum of k distinct integers from 1..256 is 1+2+...+k = k(k+1)/2.
    # But we have the distinct block sum constraint.
    # Let's just try L from 15 down to 1.
    
    for L in range(15, 0, -1):
        if deadline and time.time() > deadline:
            break
            
        found = _dfs_cp(L, max_val, deadline)
        if found:
            return found
            
    return best_seq

def _dfs_cp(L, max_val, deadline):
    """
    DFS to find a sequence of length L.
    """
    seq = []
    sums = set()
    
    def backtrack(idx, min_val):
        if deadline and time.time() > deadline:
            return False
            
        if idx == L:
            return True
            
        # Determine upper bound for this position
        # We need L - idx - 1 more elements, each at least 1 greater than previous.
        # So max_val must be at least min_val + (L - idx - 1)
        max_possible = max_val - (L - idx - 1)
        
        if min_val > max_possible:
            return False
            
        # Heuristic: Try values that leave more room? Or try middle?
        # Let's try from min_val upwards, but maybe skip some if too tight.
        # A good heuristic for "distinct sums" is often to have larger gaps.
        # Let's try a mix: some small increments, some large.
        
        # To avoid getting stuck in local minima, we can randomize the order slightly
        # or use a specific pattern.
        
        # Let's try candidates in a specific order:
        # 1. min_val
        # 2. min_val + 1
        # ...
        # But we want to find *a* solution.
        
        # Optimization: If idx is 0, we can start from 1.
        # If idx > 0, we start from min_val.
        
        # Let's iterate candidates.
        # To speed up, we can check if the candidate is "promising".
        
        for x in range(min_val, max_possible + 1):
            if deadline and time.time() > deadline:
                return False
                
            # Check validity
            if not seq:
                # First element
                if x in sums: # sums is empty, so always false
                    continue
                # Add x
                seq.append(x)
                sums.add(x)
                
                if backtrack(idx + 1, x + 1):
                    return True
                
                seq.pop()
                sums.remove(x)
            else:
                # Check extension
                # Compute tails
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                if sums.intersection(tails):
                    continue
                
                # Valid
                seq.append(x)
                sums.update(tails)
                
                if backtrack(idx + 1, x + 1):
                    return True
                
                seq.pop()
                # Remove tails from sums
                for t in tails:
                    sums.remove(t)
                    
        return False

    # Start backtracking
    # We can try different starting strategies.
    # 1. Start with 1.
    # 2. Start with 2.
    # etc.
    
    # Actually, the backtrack function handles min_val.
    # We call backtrack(0, 1).
    
    # To avoid symmetry or slow starts, we can try a few specific first moves.
    # But backtrack(0,1) will try 1, then 2, etc.
    
    # Let's add a timeout check inside the loop to be safe.
    
    if backtrack(0, 1):
        return list(seq)
        
    return None

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
            
    for x in range((values[-1] if values else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s buffer
    
    best = []
    
    # 1. Quick greedy to get a baseline
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > len(best):
        best = list(greedy_seq)
        
    # 2. CP Search for longer sequences
    # Try to find sequences of length > len(best)
    target_len = len(best) + 1
    
    # We can run CP search for specific lengths.
    # If we find one of length L, we can try to extend it or find a better one.
    
    # Let's try lengths from high to low, but stop if we find one longer than best.
    # Max possible length is likely < 20.
    
    # To make it faster, we can use the greedy result as a hint?
    # No, CP search is independent.
    
    # We'll run CP search for L = target_len up to some max.
    # If it finds a sequence of length L, we update best.
    # We can try to find the MAXIMUM length by binary search or linear scan.
    
    # Let's try L from 15 down to target_len.
    # If we find a valid sequence of length 15, that's great.
    # If not, try 14, etc.
    
    # Optimization: If we find a sequence of length L, we can try to extend it
    # using greedy to see if we can get L+1.
    
    for L in range(15, target_len - 1, -1):
        if time.time() > deadline:
            break
            
        # Run CP search for length L
        # To avoid spending too much time on one L, we can limit the search?
        # The deadline check inside backtrack handles this.
        
        found_seq = _dfs_cp(L, 256, deadline)
        if found_seq:
            if len(found_seq) > len(best):
                best = found_seq
                
                # Try to extend with greedy
                extended = _greedy_forward(best)
                if len(extended) > len(best):
                    best = extended
                    
                # If we found a long one, maybe we can stop?
                # Or try to find an even longer one?
                # The loop continues to lower L, so it won't find longer.
                # We should break if we found a very long one?
                # Or just let it run out of time.
                
                # Actually, if we found L, we should check if L+1 is possible?
                # The loop goes down, so it won't check L+1 if we started at 15.
                # But if we started at 15 and failed, we try 14.
                # If we found 14, we should have tried 15 first.
                # So if we found 14, 15 was impossible (or timeout).
                # So we can stop?
                # Not necessarily, maybe 15 was just slow.
                # But for safety, let's continue to try shorter ones only if we want to ensure validity?
                # No, we want MAX length.
                # If we found length 14, and 15 failed, is 15 really impossible?
                # Maybe it just timed out.
                # So we should probably try 15 again with more time?
                # But we have a fixed deadline.
                
                # Let's just return the best found so far if we run out of time.
                
                # If we found a sequence, we can break the loop if we think we've hit the max?
                # Hard to know. Let's just continue to lower L to ensure we have a valid
                # sequence if higher ones fail?
                # Actually, if we found L, we don't need to find L-1.
                # We want the MAX.
                # So if we found L, we should try L+1?
                # But we started from 15.
                
                # Let's change strategy:
                # Try L = 15, 14, 13...
                # If we find one, store it.
                # If we find one of length L, we can stop trying smaller lengths?
                # Yes, because we want max length.
                # But we must be sure L is the max.
                # If 15 failed due to timeout, it might be possible.
                # So we shouldn't stop immediately.
                
                # However, given the 90s limit, we can't spend all time on 15.
                # Let's allocate time per L.
                
                # Simple approach:
                # Try 15 with 20s.
                # Try 14 with 20s.
                # etc.
                
                # But _dfs_cp uses global deadline.
                # Let's modify _dfs_cp to take a local deadline?
                # Or just let it run until global deadline.
                
                # If we find a sequence of length 10, and we have 10s left,
                # should we try 11?
                # Yes.
                
                # So the loop should go UP?
                # No, we don't know the max.
                
                # Let's stick to: Try 15, 14, ...
                # If we find one, we keep it as best.
                # We continue to lower L only if we want to guarantee a valid output
                # in case higher L's fail?
                # But if we already have a valid output, we don't *need* to find a shorter one.
                # We only care if we can find a LONGER one.
                
                # So if we found L=12, we should try L=13, 14, 15.
                # My loop is 15 down to target.
                # If I find 12, I have already tried 15, 14, 13.
                # So I can stop?
                # Yes, because I've tried all higher lengths.
                
                break # Found a valid sequence of length L. Since we tried higher L's first, this is the max found.
                
    # 3. If CP search didn't find anything better, use greedy result.
    # 4. Final polish: Try to extend best with greedy again.
    if time.time() < deadline - 2:
        extended = _greedy_forward(best)
        if len(extended) > len(best):
            best = extended
            
    # 5. Random search for quick improvements?
    if time.time() < deadline - 5:
        # Simple random restart
        rng = random.Random()
        for _ in range(10):
            if time.time() > deadline:
                break
            # Start with random small number
            start = rng.randint(1, 5)
            seq = [start]
            sums = {start}
            while seq[-1] < 256:
                if time.time() > deadline:
                    break
                # Try random next
                found = False
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    cand = rng.randint(seq[-1] + 1, min(256, seq[-1] + 100))
                    tails = [cand]
                    total = cand
                    for a in reversed(seq):
                        total += a
                        tails.append(total)
                    if not sums.intersection(tails):
                        seq.append(cand)
                        sums.update(tails)
                        found = True
                        break
                if not found:
                    break
            if len(seq) > len(best):
                best = seq
                
    # Ensure validity
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
