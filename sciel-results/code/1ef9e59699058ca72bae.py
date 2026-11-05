import time
import random

def _check_extension(seq, sums, x):
    """
    Check if adding x to the end of seq is valid.
    Returns the new sums to add if valid, or None if invalid.
    """
    if not seq:
        if x in sums:
            return None
        return [x]
    
    tails = []
    total = x
    tails.append(total)
    for a in reversed(seq):
        total += a
        tails.append(total)
        
    for s in tails:
        if s in sums:
            return None
            
    return tails

def _greedy_randomized(k=1, seed=None):
    """
    Greedy construction with randomization.
    If k=1, it's standard greedy (smallest valid).
    If k>1, it picks randomly from the last k valid candidates seen.
    """
    if seed is not None:
        random.seed(seed)
        
    seq = []
    sums = set()
    candidates_buffer = []
    
    for x in range(1, 257):
        new_tails = _check_extension(seq, sums, x)
        
        if new_tails is not None:
            candidates_buffer.append((x, new_tails))
            
            if len(candidates_buffer) >= k:
                idx = random.randint(0, len(candidates_buffer) - 1)
                x_val, tails = candidates_buffer[idx]
                
                seq.append(x_val)
                sums.update(tails)
                
                candidates_buffer = []
                
    if candidates_buffer:
        idx = random.randint(0, len(candidates_buffer) - 1)
        x_val, tails = candidates_buffer[idx]
        seq.append(x_val)
        sums.update(tails)

    return seq

def _backtrack_search(seq, sums, start_val, max_depth, deadline, rng):
    """
    Depth-first search with backtracking.
    Tries to extend the current sequence by choosing a value >= start_val.
    """
    if time.time() > deadline:
        return seq[:]
    
    if len(seq) >= max_depth:
        return seq[:]
        
    best = seq[:]
    
    # Try values in increasing order, but with some randomization/shuffling
    # We want to find a valid extension. If we find one, we recurse.
    # If we don't find one, we return the current best (which is seq).
    # To allow for "skipping", we just iterate and if one fails, we try the next.
    # But standard DFS backtracking usually implies trying all options.
    # Here, we are constrained by time.
    
    # Optimization: Only try a limited number of candidates to avoid deep, fruitless searches
    # that waste time.
    tried = 0
    max_tries = 20
    
    for x in range(start_val, 257):
        if tried >= max_tries:
            break
        if time.time() > deadline:
            break
            
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            tried += 1
            # Valid extension
            new_seq = seq + [x]
            new_sums = sums.union(new_tails)
            
            # Recurse
            res = _backtrack_search(new_seq, new_sums, x + 1, max_depth, deadline, rng)
            if len(res) > len(best):
                best = res
                
            # If we found a sequence of max_depth, we can stop? 
            # Or maybe there's a longer one? No, max_depth is the limit.
            # But we might be able to find a *better* sequence of the same length?
            # The objective is just length. So any sequence of max_depth is equally good.
            # However, a different sequence of max_depth might allow for... wait, we are at max_depth.
            # So if we hit max_depth, we can return immediately?
            # Actually, if we hit max_depth, we can't extend further.
            # So if len(res) == max_depth, we can return it.
            if len(res) == max_depth:
                return res
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_seq = []
    best_len = 0
    
    # 1. Deterministic Greedy (k=1)
    seq_greedy = _greedy_randomized(k=1, seed=0)
    if len(seq_greedy) > best_len:
        best_len = len(seq_greedy)
        best_seq = seq_greedy[:]
        
    # 2. Randomized Greedy with various k and seeds
    for k in [2, 3, 5, 10, 20]:
        for seed in range(20):
            if time.time() > deadline:
                break
            seq = _greedy_randomized(k=k, seed=seed)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # 3. Backtracking Search
    # Start from the greedy solution and try to extend it or replace elements?
    # The _backtrack_search above starts from a prefix.
    # We can try to find a better sequence by starting from empty and using backtracking.
    # But that's expensive.
    # Alternative: Use the greedy solution as a guide.
    # Or, run backtracking from scratch with a depth limit slightly higher than best_len.
    
    # Let's try to improve upon the best greedy solution found so far.
    # We can't easily "improve" a fixed length sequence to a longer one without restarting.
    # So, let's run a global backtracking search.
    # To make it fast, we limit the depth and the branching factor.
    
    # We know best_len. We want to find best_len + 1.
    # So we set max_depth = best_len + 1.
    # If we find one, we update best_len and try again for best_len + 2.
    
    target_len = best_len + 1
    max_search_time = 10.0
    
    while time.time() < deadline - 5.0:
        if target_len <= best_len:
            break
            
        # Try to find a sequence of length target_len
        # We use random restarts with backtracking
        found = False
        rng = random.Random(int(time.time() * 1000))
        
        # Limit attempts per target_len
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts and time.time() < deadline - 5.0 and not found:
            attempts += 1
            # Start with a random small prefix or empty?
            # Empty is best for global search, but slow.
            # Let's try starting with the first few elements of the best greedy sequence?
            # That biases the search.
            # Let's just start empty.
            
            # To speed up, we can use the greedy solution to order candidates?
            # For now, simple DFS.
            res = _backtrack_search([], set(), 1, target_len, deadline - 5.0, rng)
            
            if len(res) >= target_len:
                best_seq = res
                best_len = len(res)
                target_len = best_len + 1
                found = True
                
    return best_seq
