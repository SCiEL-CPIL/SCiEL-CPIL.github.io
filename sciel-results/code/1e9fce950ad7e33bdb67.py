import sys
import time

sys.setrecursionlimit(10000)

def construct():
    # We will use a bitset for sums to speed up membership testing.
    # Max possible sum: 256 * 20 (max length estimate) = 5120.
    # Let's use 10000 to be safe.
    MAX_SUM = 10000
    sums_bitset = 0
    
    best_seq = []
    
    start_time = time.time()
    time_limit = 55.0
    
    # Greedy baseline to get a good starting point
    # The current greedy (smallest valid) gives 9.
    # Let's try to find a longer one.
    
    def dfs(current_seq, last_val, target_len):
        nonlocal sums_bitset
        if time.time() - start_time > time_limit:
            return False
        
        if len(current_seq) == target_len:
            return True
            
        # Pruning:
        # 1. Not enough numbers left
        if len(current_seq) + (256 - last_val) < target_len:
            return False
            
        # 2. Sum bound
        # The number of new sums added by extending to length L is L.
        # Total sums will be sum_{i=1}^L i = L(L+1)/2.
        # These sums must fit in [1, MAX_SUM].
        # This is always true for L=10, 11.
        
        # Calculate suffix sums of current_seq
        # suffix_sums[0] = 0
        # suffix_sums[i] = sum of last i elements
        suffix_sums = [0]
        s = 0
        for a in reversed(current_seq):
            s += a
            suffix_sums.append(s)
        
        # Try candidates
        # Heuristic: Try values that are "far" from existing sums?
        # Or just iterate?
        # To speed up, we can pre-compute which x are valid?
        # For each x, check if any (x + s) is in sums_bitset.
        
        # We want to find *any* sequence of length target_len.
        # To find a longer one, we should perhaps try to keep the sequence sparse.
        # Powers of 2 are sparse.
        # Let's try candidates in increasing order, but skip those that are "too close" to previous?
        
        for x in range(last_val + 1, 257):
            # Check validity
            valid = True
            new_sums = []
            for s in suffix_sums:
                ns = x + s
                if ns >= MAX_SUM:
                    valid = False
                    break
                if (sums_bitset >> ns) & 1:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            # Save state
            old_bitset = sums_bitset
            for ns in new_sums:
                sums_bitset |= (1 << ns)
                
            # Recurse
            if dfs(current_seq + [x], x, target_len):
                return True
                
            # Restore state
            sums_bitset = old_bitset
            
        return False

    # Start with a good seed?
    # The greedy smallest-valid sequence is a good seed for structure,
    # but we are doing DFS from scratch.
    # To speed up DFS, we can use the greedy sequence to guide the order?
    # Or just run DFS for target=10, then 11.
    
    # Let's try to find length 10 first.
    # If found, try 11.
    
    # To make DFS faster, we can use a heuristic ordering of x.
    # If we prefer larger x, we keep sums larger, avoiding small sum collisions.
    # Let's try iterating x from high to low?
    # No, strictly increasing sequence, so x > last_val.
    # If we pick large x early, we might run out of room for more elements?
    # If we pick small x, we pack densely, might block early.
    
    # Let's try a modified greedy that picks the *largest* valid x?
    # No, let's stick to DFS but optimize the inner loop.
    
    # One optimization: Pre-calculate suffix sums once per node.
    # Another: Use a list for sums if bitset is slow? Bitset is faster for membership.
    
    # Let's try target lengths from 10 upwards.
    # But DFS from scratch for target 10 might be slow if the first branch fails deep.
    
    # Alternative: Use the greedy sequence as a "hint".
    # If greedy gives [1, 2, 4, 8, 16, 32, 64, 128, 256], 
    # we know these are valid.
    # Can we replace 256 with two smaller numbers?
    # e.g. replace 256 with 129, 130?
    # 129 > 128.
    # Sums for 1..128 are fixed.
    # Add 129: Sums 129, 257, ...
    # Add 130: Sums 130, 130+129=259, ...
    # Check collisions.
    
    # This "local search" on the greedy solution might be faster than global DFS.
    # But the directive asks for a specific design.
    
    # Let's implement the DFS with bitsets and time limit.
    # We'll try to find length 10.
    
    # To improve speed, we can use a heuristic: 
    # Try x in an order that maximizes the "gap" to the previous element?
    # Or just try all.
    
    # Let's add a node limit to DFS to prevent hanging on a dead branch.
    nodes = 0
    node_limit = 500000
    
    def dfs_limited(current_seq, last_val, target_len):
        nonlocal sums_bitset, nodes
        if time.time() - start_time > time_limit:
            return False
        if nodes > node_limit:
            return False
        nodes += 1
        
        if len(current_seq) == target_len:
            return True
            
        if len(current_seq) + (256 - last_val) < target_len:
            return False
            
        suffix_sums = [0]
        s = 0
        for a in reversed(current_seq):
            s += a
            suffix_sums.append(s)
        
        # Heuristic: Try x that are multiples of 2? Or just increasing?
        # Let's try increasing.
        for x in range(last_val + 1, 257):
            valid = True
            new_sums = []
            for s in suffix_sums:
                ns = x + s
                if ns >= MAX_SUM:
                    valid = False
                    break
                if (sums_bitset >> ns) & 1:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            old_bitset = sums_bitset
            for ns in new_sums:
                sums_bitset |= (1 << ns)
                
            if dfs_limited(current_seq + [x], x, target_len):
                return True
                
            sums_bitset = old_bitset
            
        return False

    # Try to find length 10
    # We can try to seed with the greedy prefix?
    # Greedy prefix of length 8: 1, 2, 4, 8, 16, 32, 64, 128
    # Sums of this prefix are known.
    # We can start DFS from this prefix and try to add 2 more elements.
    # This reduces the depth from 10 to 2.
    
    # Let's calculate sums for the greedy prefix.
    prefix = [1, 2, 4, 8, 16, 32, 64, 128]
    prefix_sums = set()
    n = len(prefix)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += prefix[j]
            prefix_sums.add(s)
            
    # Initialize bitset
    sums_bitset = 0
    for s in prefix_sums:
        sums_bitset |= (1 << s)
        
    # Now DFS to add 2 more elements to this prefix.
    # Target total length 10.
    # So we need to find 2 elements x, y > 128.
    
    # Reset nodes
    nodes = 0
    node_limit = 1000000 # Higher limit for shallow DFS
    
    def dfs_extend(current_seq, last_val, target_len):
        nonlocal sums_bitset, nodes
        if time.time() - start_time > time_limit:
            return False
        if nodes > node_limit:
            return False
        nodes += 1
        
        if len(current_seq) == target_len:
            return True
            
        if len(current_seq) + (256 - last_val) < target_len:
            return False
            
        suffix_sums = [0]
        s = 0
        for a in reversed(current_seq):
            s += a
            suffix_sums.append(s)
        
        for x in range(last_val + 1, 257):
            valid = True
            new_sums = []
            for s in suffix_sums:
                ns = x + s
                if ns >= MAX_SUM:
                    valid = False
                    break
                if (sums_bitset >> ns) & 1:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            old_bitset = sums_bitset
            for ns in new_sums:
                sums_bitset |= (1 << ns)
                
            if dfs_extend(current_seq + [x], x, target_len):
                return True
                
            sums_bitset = old_bitset
            
        return False

    # Try to extend the prefix to length 10
    # We need to add 2 elements.
    # So we call dfs_extend(prefix, 128, 10)
    # But dfs_extend expects current_seq to be the full sequence so far.
    # It checks len(current_seq) == target_len.
    
    if dfs_extend(prefix, 128, 10):
        # We found a sequence of length 10?
        # Wait, dfs_extend doesn't return the sequence, just True/False.
        # We need to recover the sequence.
        # Let's modify to store the best sequence found.
        pass

    # Let's re-implement to return the sequence.
    
    best_seq_found = None
    
    def dfs_find(current_seq, last_val, target_len):
        nonlocal sums_bitset, best_seq_found, nodes
        if time.time() - start_time > time_limit:
            return False
        if nodes > node_limit:
            return False
        nodes += 1
        
        if len(current_seq) == target_len:
            best_seq_found = current_seq[:]
            return True
            
        if len(current_seq) + (256 - last_val) < target_len:
            return False
            
        suffix_sums = [0]
        s = 0
        for a in reversed(current_seq):
            s += a
            suffix_sums.append(s)
        
        for x in range(last_val + 1, 257):
            valid = True
            new_sums = []
            for s in suffix_sums:
                ns = x + s
                if ns >= MAX_SUM:
                    valid = False
                    break
                if (sums_bitset >> ns) & 1:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            old_bitset = sums_bitset
            for ns in new_sums:
                sums_bitset |= (1 << ns)
                
            if dfs_find(current_seq + [x], x, target_len):
                return True
                
            sums_bitset = old_bitset
            
        return False

    # Try extending the greedy prefix to length 10
    # Reset state
    sums_bitset = 0
    for s in prefix_sums:
        sums_bitset |= (1 << s)
    nodes = 0
    best_seq_found = None
    
    if dfs_find(prefix, 128, 10):
        return best_seq_found
        
    # If failed, try from scratch for length 10?
    # Or just return the greedy length 9?
    
    # Let's try to find length 10 from scratch with a tighter time limit?
    # No, the prefix extension is a good heuristic.
    
    # If that fails, return the greedy sequence.
    return prefix # Length 8? No, greedy was length 9.
    # The prefix above is length 8.
    # Let's use the full greedy sequence as fallback.
    
    # Recalculate greedy
    values, sums_set = [], set()
    for x in range(1, 257):
        suffix_sums = [0]
        s = 0
        for a in reversed(values):
            s += a
            suffix_sums.append(s)
        valid = True
        new_sums = []
        for s in suffix_sums:
            ns = x + s
            if ns in sums_set:
                valid = False
                break
            new_sums.append(ns)
        if valid:
            values.append(x)
            sums_set.update(new_sums)
            
    return values
