import time
import random

def _all_block_sums(seq):
    """Compute all nonempty consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _check_valid(seq):
    """Check that seq is strictly increasing, all elements in [1,256], and all consecutive block sums distinct."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    sums = _all_block_sums(seq)
    n = len(seq)
    expected = n * (n + 1) // 2
    if len(sums) != expected:
        return False
    return True

def _greedy_build(start_value, max_time=2.0, strategy="largest"):
    """
    Build a sequence starting with start_value using a greedy strategy.
    strategy: "smallest", "largest"
    """
    start_time = time.time()
    seq = [start_value]
    sums_set = {start_value}
    last = start_value
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        n = len(seq)
        # Compute suffix sums for conflict check
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        valid_next = []
        for x in range(last + 1, 257):
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            if not conflict:
                valid_next.append(x)
        
        if not valid_next:
            break
        
        if strategy == "smallest":
            best_x = valid_next[0]
        elif strategy == "largest":
            best_x = valid_next[-1]
        else:
            best_x = random.choice(valid_next)
        
        seq.append(best_x)
        # Update sums
        for k in range(n + 1):
            sums_set.add(suffix_sums[k] + best_x)
        last = best_x
        
    return seq

def _flexibility_score(x, seq, sums_set, suffix_sums):
    """
    Score how flexible choosing x is: count how many future values (x+1 to 256)
    would still be valid after adding x. Higher score = more flexible.
    """
    n = len(seq)
    new_sums = sums_set.copy()
    for k in range(n + 1):
        new_sums.add(suffix_sums[k] + x)
    
    # Count how many values > x are still valid
    count = 0
    for y in range(x + 1, 257):
        conflict = False
        for k in range(n + 2):
            # After adding x, suffix sums change
            if k == n + 1:
                # This is the new suffix: just x
                if x + y in new_sums:
                    conflict = True
                    break
            else:
                if suffix_sums[k] + x + y in new_sums:
                    conflict = True
                    break
        if not conflict:
            count += 1
    return count

def _greedy_with_flexibility(start_value, max_time=2.0):
    """
    Build sequence using greedy with flexibility lookahead.
    At each step, among valid choices, pick the one that leaves the most
    future flexibility (most valid next values).
    """
    start_time = time.time()
    seq = [start_value]
    sums_set = {start_value}
    last = start_value
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        n = len(seq)
        # Compute suffix sums for conflict check
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        valid_next = []
        for x in range(last + 1, 257):
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            if not conflict:
                valid_next.append(x)
        
        if not valid_next:
            break
        
        # Score each valid choice by flexibility
        if len(valid_next) <= 5:
            # Few options, just pick the one with max flexibility
            best_x = max(valid_next, key=lambda x: _flexibility_score(x, seq, sums_set, suffix_sums))
        else:
            # Many options: sample a subset for speed, pick best flexibility
            sample_size = min(20, len(valid_next))
            sample = random.sample(valid_next, sample_size)
            best_x = max(sample, key=lambda x: _flexibility_score(x, seq, sums_set, suffix_sums))
        
        seq.append(best_x)
        # Update sums
        for k in range(n + 1):
            sums_set.add(suffix_sums[k] + best_x)
        last = best_x
        
    return seq

def _local_search_improve(seq, sums_set, max_time=5.0):
    """
    Try to extend the sequence by backtracking one or two steps
    and trying different choices.
    Returns improved sequence or original if no improvement.
    """
    start_time = time.time()
    best = seq[:]
    best_len = len(seq)
    
    for backtrack_depth in [1, 2, 3]:
        if time.time() - start_time > max_time:
            break
        
        if backtrack_depth > len(seq):
            continue
        
        prefix = seq[:len(seq) - backtrack_depth]
        if not prefix:
            continue
            
        # Try to extend from prefix with different choices
        prefix_sums = set()
        for i in range(len(prefix)):
            s = 0
            for j in range(i, len(prefix)):
                s += prefix[j]
                prefix_sums.add(s)
        
        # Get all valid next values after prefix
        n = len(prefix)
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += prefix[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        valid_next = []
        for x in range(prefix[-1] + 1, 257):
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in prefix_sums:
                    conflict = True
                    break
            if not conflict:
                valid_next.append(x)
        
        if not valid_next:
            continue
        
        # Try each valid next value, then greedily extend
        for x in valid_next:
            if time.time() - start_time > max_time:
                break
            
            new_seq = prefix + [x]
            new_sums = prefix_sums.copy()
            for k in range(n + 1):
                new_sums.add(suffix_sums[k] + x)
            
            # Greedily extend from here
            last = x
            while True:
                if time.time() - start_time > max_time:
                    break
                
                m = len(new_seq)
                # Compute suffix sums
                ss = [0] * (m + 1)
                s2 = 0
                for k in range(m - 1, -1, -1):
                    s2 += new_seq[k]
                    ss[k] = s2
                ss[m] = 0
                
                valid_next2 = []
                for y in range(last + 1, 257):
                    conflict = False
                    for k in range(m + 1):
                        if ss[k] + y in new_sums:
                            conflict = True
                            break
                    if not conflict:
                        valid_next2.append(y)
                
                if not valid_next2:
                    break
                
                # Pick with flexibility
                if len(valid_next2) <= 5:
                    best_y = max(valid_next2, key=lambda y: _flexibility_score(y, new_seq, new_sums, ss))
                else:
                    sample_size = min(10, len(valid_next2))
                    sample = random.sample(valid_next2, sample_size)
                    best_y = max(sample, key=lambda y: _flexibility_score(y, new_seq, new_sums, ss))
                
                new_seq.append(best_y)
                for k in range(m + 1):
                    new_sums.add(ss[k] + best_y)
                last = best_y
            
            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best = new_seq[:]
    
    return best

def construct():
    """
    Main entry point. Returns a strictly increasing sequence in [1,256] with distinct consecutive-block sums.
    Uses greedy with flexibility lookahead and local search improvement.
    """
    total_start = time.time()
    deadline = total_start + 80.0  # Safety margin for 90s limit
    
    best_seq = []
    best_len = 0
    
    # 1. Initial Greedy Passes with different strategies
    for start_val in range(1, 15):
        if time.time() > deadline - 10:
            break
        for strategy in ["largest", "smallest"]:
            seq = _greedy_build(start_val, max_time=0.5, strategy=strategy)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
    
    # 2. Greedy with flexibility lookahead
    for start_val in range(1, 15):
        if time.time() > deadline - 15:
            break
        seq = _greedy_with_flexibility(start_val, max_time=1.0)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # 3. Local search improvement on best sequence found
    if time.time() < deadline - 10:
        # Rebuild sums_set for best_seq
        sums_set = _all_block_sums(best_seq)
        improved = _local_search_improve(best_seq, sums_set, max_time=min(20.0, deadline - time.time() - 5))
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved
    
    # 4. More local search iterations if time permits
    while time.time() < deadline - 10:
        sums_set = _all_block_sums(best_seq)
        improved = _local_search_improve(best_seq, sums_set, max_time=min(10.0, deadline - time.time() - 5))
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved
        else:
            # No improvement, try different random seeds by perturbing
            if time.time() > deadline - 15:
                break
    
    # Verify result
    if not _check_valid(best_seq):
        # Fallback
        best_seq = [1, 2, 5]
        
    return best_seq
