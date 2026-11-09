import time
import random
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

def _dfs_csp(start_seq, max_val=256, deadline=None, best_len=0):
    """
    Depth-First Search with Backtracking.
    State: (current_seq, cur_sums_set)
    Goal: Maximize length.
    Pruning: 
    1. If time exceeded, return.
    2. If len(current) + max_possible_additional <= best_len, prune.
       max_possible_additional is roughly (max_val - last_val) / min_gap.
       Since values are strictly increasing integers, min gap is 1.
       So max additional is max_val - last_val.
    """
    if deadline and time.time() > deadline:
        return start_seq, best_len

    current = list(start_seq)
    if not current:
        return current, best_len
        
    last_val = current[-1]
    
    # Calculate current sums
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    # If we already have a good length, we can prune if we can't beat it
    # Max possible length is len(current) + (max_val - last_val)
    if len(current) + (max_val - last_val) <= best_len:
        return current, best_len

    # Recurse: try adding next elements
    # To improve on greedy, we should try candidates in a smart order.
    # Greedy tries smallest. Sometimes larger gaps are better for future.
    # Let's try a mix: smallest first, but if it fails deep, backtrack.
    
    # Optimization: We can use a set of valid candidates to speed up checks
    # But checking intersection is O(n) where n is len(cur_sums) ~ O(k^2).
    # For k=10, k^2=100. 256 candidates * 100 checks = 25600 ops per node.
    # DFS depth ~10. Total ops ~ 256k. Very fast.
    
    best_found = list(current)
    
    # Candidate generation: 
    # Instead of all 1..256, we can be smarter.
    # But for correctness and simplicity in a 60s budget, we can iterate.
    # To avoid getting stuck in a local minimum of "greedy smallest", 
    # we can randomize the order of candidates slightly or use multiple passes.
    
    # Let's do a standard DFS but with a time limit.
    # We will try to extend the current sequence.
    
    def dfs(seq, sums_set):
        nonlocal best_found, best_len
        if deadline and time.time() > deadline:
            return
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_found = list(seq)
            
        last = seq[-1]
        if last >= max_val:
            return
            
        # Pruning bound
        # Even if we add 1, 2, 3... we can add at most max_val - last elements.
        if len(seq) + (max_val - last) <= best_len:
            return
            
        # Try candidates
        # We want to explore broadly. 
        # Sorting candidates by "promisingness" is hard.
        # Let's just iterate from last+1 to max_val.
        # To avoid deep recursion stack issues (though depth is < 20), 
        # Python default limit is 1000, so we are safe.
        
        # Optimization: Pre-compute tails for each candidate?
        # Tails depend on the whole sequence.
        
        for x in range(last + 1, max_val + 1):
            if deadline and time.time() > deadline:
                return
            
            # Check validity of x
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            # Check intersection
            valid = True
            for t in tails:
                if t in sums_set:
                    valid = False
                    break
            
            if valid:
                # Recurse
                new_sums = set(sums_set)
                new_sums.update(tails)
                dfs(seq + [x], new_sums)
                
                # Backtracking is implicit in the recursion stack
                # We don't need to manually remove from sums_set if we pass a copy
                # But copying a set of size O(k^2) is O(k^2).
                # For k=10, 100 items. This is fine.
                
    dfs(current, cur_sums)
    
    return best_found, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run DFS/CSP search starting from empty and various small prefixes
    # DFS from empty might be too slow if it explores bad branches.
    # Better to start from the greedy result? No, DFS needs to explore alternatives.
    # Let's run DFS from a few different starting points to diversify.
    
    # Strategy: Run DFS from empty, but with a time limit.
    # If DFS from empty is too slow, we can use the greedy result as a "seed" 
    # and try to improve it by backtracking from the end?
    # Actually, standard DFS from empty with pruning is robust.
    
    # To ensure we don't waste time on very short sequences, 
    # we can force the first few elements to be from the greedy solution?
    # No, that restricts search.
    
    # Let's try DFS from empty.
    found_seq, found_len = _dfs_csp([], max_val=256, deadline=deadline, best_len=best_len)
    if found_len > best_len:
        best = found_seq
        best_len = found_len
        
    # 3. If we have time, try DFS from different small prefixes to escape local optima
    # The DFS from empty should theoretically cover all, but if it gets stuck in a 
    # "bad" region of the search tree (e.g., starts with 1,2,4... which might block),
    # starting with 1,3,7... might be better.
    # However, DFS explores ALL valid paths. It will find the global optimum 
    # within the time limit if the search space is small enough or pruning is effective.
    
    # Given max_val=256, the branching factor is up to 256.
    # Depth is ~10-15.
    # 256^10 is huge. Pruning is essential.
    # The pruning "len + (max_val - last) <= best_len" is weak initially.
    
    # Let's add a heuristic: Randomized DFS to diversify if time remains.
    if time.time() < deadline - 5:
        # Try a few randomized restarts or different ordering
        # Actually, simple DFS is deterministic. 
        # We can try to "shake" the search by starting with a random valid prefix.
        
        rng = random.Random(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            # Generate a random valid prefix of length 3
            prefix = []
            cur_sums = set()
            last = 0
            try:
                for _ in range(3):
                    # Pick random next
                    found = False
                    for _ in range(100):
                        x = rng.randint(last + 1, 256)
                        tails = [x]
                        total = x
                        for a in reversed(prefix):
                            total += a
                            tails.append(total)
                        if not cur_sums.intersection(tails):
                            prefix.append(x)
                            cur_sums.update(tails)
                            last = x
                            found = True
                            break
                    if not found:
                        break
                if len(prefix) == 3:
                    found_seq, found_len = _dfs_csp(prefix, max_val=256, deadline=deadline, best_len=best_len)
                    if found_len > best_len:
                        best = found_seq
                        best_len = found_len
            except Exception:
                pass

    # Final validation
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
