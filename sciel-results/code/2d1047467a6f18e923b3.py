import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Greedy initialization to get a baseline
    def greedy_construct():
        values = []
        prefix = [0]
        diffs = set()
        last_a = 0
        for a in range(1, 257):
            if a <= last_a:
                continue
            # Check if adding 'a' is valid
            new_p = prefix[-1] + a
            new_diffs = []
            valid = True
            for i in range(len(prefix)):
                d = new_p - prefix[i]
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                values.append(a)
                prefix.append(new_p)
                diffs.update(new_diffs)
                last_a = a
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # DFS to try to extend the sequence
    # We will try to find a sequence of length best_len + 1, then +2, etc.
    # Or just do a DFS that maximizes length.
    
    def dfs(prefix, diffs, last_a, depth, current_seq):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning: if we can't beat best_len, return
        # Max possible additional steps:
        # Smallest next a is last_a + 1.
        # If we add m more terms, the last a will be at least last_a + m.
        # So last_a + m <= 256 => m <= 256 - last_a.
        # So max potential length is depth + (256 - last_a).
        if depth + (256 - last_a) <= best_len:
            return
            
        if depth > best_len:
            best_len = depth
            best_seq = current_seq[:]
        
        # Try next a
        # Heuristic: Try smaller 'a' first? Or larger?
        # Smaller 'a' keeps prefix sum small, might reduce collisions?
        # But larger 'a' separates sums more.
        # Let's try a mix: randomize order slightly or just iterate.
        # To avoid deep recursion on dead ends, we can limit the number of branches explored.
        
        # Optimization: If we are close to the limit, be more selective.
        # For now, try all valid a.
        
        for a in range(last_a + 1, 257):
            new_p = prefix[-1] + a
            new_diffs = []
            valid = True
            for i in range(len(prefix)):
                d = new_p - prefix[i]
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Recurse
            prefix.append(new_p)
            diffs.update(new_diffs)
            current_seq.append(a)
            
            dfs(prefix, diffs, a, depth + 1, current_seq)
            
            # Backtrack
            prefix.pop()
            diffs.difference_update(new_diffs)
            current_seq.pop()
            
            if time.time() > deadline:
                return

    # Start DFS from the greedy result to try to extend it?
    # Or start from scratch with DFS?
    # Starting from scratch with DFS might find a better structure.
    # But greedy is fast. Let's start DFS from the greedy state to see if we can extend.
    # Also, we can try different greedy strategies.
    
    # Let's run a few random greedy variants to get diverse starting points
    random.seed(42)
    
    # Run DFS from empty sequence
    # To make it faster, we can use a heuristic:
    # Sort candidate 'a' values by some metric?
    
    # Let's just run the DFS from the greedy result first.
    prefix = [0]
    diffs = set()
    last_a = 0
    for a in best_seq:
        new_p = prefix[-1] + a
        for i in range(len(prefix)):
            diffs.add(new_p - prefix[i])
        prefix.append(new_p)
        last_a = a
        
    dfs(prefix, diffs, last_a, len(best_seq), best_seq[:])
    
    # If time remains, try a fresh DFS with different heuristics
    if time.time() < deadline - 10:
        # Try starting from scratch with a different heuristic
        # e.g. try larger 'a' first
        def dfs_large(prefix, diffs, last_a, depth, current_seq):
            nonlocal best_seq, best_len, deadline
            if time.time() > deadline:
                return
            if depth + (256 - last_a) <= best_len:
                return
            if depth > best_len:
                best_len = depth
                best_seq = current_seq[:]
            
            # Try larger a first
            for a in range(256, last_a, -1):
                new_p = prefix[-1] + a
                new_diffs = []
                valid = True
                for i in range(len(prefix)):
                    d = new_p - prefix[i]
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                prefix.append(new_p)
                diffs.update(new_diffs)
                current_seq.append(a)
                dfs_large(prefix, diffs, a, depth + 1, current_seq)
                prefix.pop()
                diffs.difference_update(new_diffs)
                current_seq.pop()
                if time.time() > deadline:
                    return

        dfs_large([0], set(), 0, 0, [])

    return best_seq
