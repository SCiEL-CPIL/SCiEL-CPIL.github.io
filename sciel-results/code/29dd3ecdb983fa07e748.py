import time
import sys

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

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _dfs_with_pruning(max_val=256, time_limit=45.0):
    """
    DFS with backtracking and pruning.
    State: (current_seq, current_sums, last_val)
    Heuristic: Try smaller values first to keep last_val low (more room).
    Pruning: 
      1. Time limit.
      2. Upper bound on possible length: (max_val - last_val) / 1. 
         Actually, since values are strictly increasing integers, max additional elements is max_val - last_val.
         If current_len + (max_val - last_val) <= best_len, prune.
      3. More sophisticated: The sum of the next k elements must be at least (last+1) + ... + (last+k).
         Also, block sums grow.
    """
    best_seq = []
    best_len = 0
    start_time = time.time()
    
    # Initial state
    seq = []
    sums = set()
    last_val = 0
    
    # We want to maximize length.
    # A simple upper bound on how many more elements we can add:
    # If we pick the smallest possible next values, we use the least range.
    # Min next value is last_val + 1.
    # So max possible length is last_val + (max_val - last_val) = max_val? No.
    # The number of elements we can add is bounded by the available integers.
    # If current length is L, last is V. We can add at most max_val - V more elements.
    # So total length <= L + max_val - V.
    
    def dfs(seq, sums, last_val, current_len):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)
        
        # Pruning: Upper bound on potential length
        # Max additional elements is max_val - last_val
        if current_len + (max_val - last_val) <= best_len:
            return
            
        # Try to extend
        # Heuristic: Try smaller values first. This tends to keep the "sums" smaller and distinct for longer?
        # Or does it? Smaller values mean smaller block sums, which might collide more easily with existing small sums?
        # Actually, if we pick small values, the new block sums are small. If we pick large values, new block sums are large.
        # Existing sums are a mix.
        # Let's stick to ascending order.
        
        for x in range(last_val + 1, max_val + 1):
            # Quick check: does x create a conflict?
            # New block sums involving x:
            # x
            # x + seq[-1]
            # x + seq[-1] + seq[-2]
            # ...
            # x + sum(seq)
            
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
                
            # Check intersection
            conflict = False
            for t in tails:
                if t in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Recurse
            new_sums = sums.union(tails)
            seq.append(x)
            dfs(seq, new_sums, x, current_len + 1)
            seq.pop()
            
            # Optional: If we found a sequence of length best_len, we can stop this branch?
            # No, we want to find even longer ones.
            
            # Optimization: If we just extended to a new best, we might want to continue exploring this path deeply first?
            # The recursive call handles that.
            
            # Pruning inside loop:
            # If current_len + 1 + (max_val - x) <= best_len, then even if we pick all remaining smallest integers, we can't beat best.
            # So we can break out of the loop for this level? 
            # Wait, x is increasing. max_val - x is decreasing.
            # So if for some x, the upper bound is <= best_len, then for all larger x, the upper bound is even smaller.
            # So we can break.
            if current_len + 1 + (max_val - x) <= best_len:
                break

    dfs([], set(), 0, 0)
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best = []
    best_len = 0
    
    # 1. Greedy for initial lower bound (fast)
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = list(greedy_seq)
        best_len = len(best)
        
    # 2. DFS with pruning (main strategy)
    # We give it most of the time.
    dfs_time_limit = min(deadline - start_time - 5, 45.0) # Leave 5s for fallback/validation
    if dfs_time_limit > 5:
        dfs_seq = _dfs_with_pruning(max_val=256, time_limit=dfs_time_limit)
        if len(dfs_seq) > best_len:
            best = list(dfs_seq)
            best_len = len(best)
            
    # 3. If time remains and we haven't hit a theoretical max, try a second DFS run with different heuristic?
    # Or just return. The DFS is usually good.
    # Let's add a simple check: if we have time, run greedy again? No, it's deterministic.
    # Maybe run DFS with a "larger first" heuristic?
    # The current DFS tries smaller first. This is generally good for maximizing count.
    
    # Final validation
    if not _is_valid_full(best):
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
