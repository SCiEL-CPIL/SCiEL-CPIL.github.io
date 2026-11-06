import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    N = 256

    def try_build_seq(max_len, time_budget, random_seed=None):
        """
        Build a strictly increasing sequence by backtracking with constraint propagation.
        State: current sequence (list), set of used values, set of prefix sums.
        When we append a new value v at position k (0-indexed), the new prefix sum is
        prefix_sum[k] + v (where prefix_sum[k] is the sum of the first k elements).
        All new consecutive block sums ending at position k are:
            prefix_sum[k] + v - prefix_sum[i] for i in 0..k
        These must not collide with any existing block sum.
        
        We maintain a set of all block sums seen so far. When appending v, we check
        all new block sums (k+1 of them) against the existing set.
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        best_seq = []
        best_len = 0
        end_time = time.time() + time_budget
        
        # We'll do multiple random restarts with different initial choices
        for attempt in range(500):
            if time.time() > end_time:
                break
            
            # Randomize the starting point
            seq = []
            used = set()
            block_sums = set()
            prefix_sums = [0]  # prefix_sums[i] = sum of first i elements
            
            # Try to build up to max_len elements
            success = True
            for pos in range(max_len):
                if time.time() > end_time:
                    break
                
                # Find candidate values: must be > last element (if any), not in used
                min_val = (seq[-1] + 1) if seq else 1
                candidates = [v for v in range(min_val, N + 1) if v not in used]
                
                if not candidates:
                    success = False
                    break
                
                # Shuffle candidates for randomization
                random.shuffle(candidates)
                
                found = False
                for v in candidates:
                    # Check if appending v creates any conflict
                    # New prefix sum
                    new_prefix = prefix_sums[-1] + v
                    # New block sums: new_prefix - prefix_sums[i] for i in 0..len(seq)
                    conflict = False
                    new_sums = []
                    for i in range(len(prefix_sums)):
                        s = new_prefix - prefix_sums[i]
                        if s in block_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    
                    if not conflict:
                        # Accept v
                        seq.append(v)
                        used.add(v)
                        prefix_sums.append(new_prefix)
                        for s in new_sums:
                            block_sums.add(s)
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            
            if best_len >= max_len:
                break
        
        return best_seq
    
    # First, find a good max_len to target via quick greedy
    # Greedy: always pick the smallest available value that doesn't conflict
    def greedy_build():
        seq = []
        used = set()
        block_sums = set()
        prefix_sums = [0]
        
        for pos in range(20):  # try to build up to 20
            min_val = (seq[-1] + 1) if seq else 1
            found = False
            for v in range(min_val, N + 1):
                if v in used:
                    continue
                new_prefix = prefix_sums[-1] + v
                conflict = False
                new_sums = []
                for i in range(len(prefix_sums)):
                    s = new_prefix - prefix_sums[i]
                    if s in block_sums:
                        conflict = True
                        break
                    new_sums.append(s)
                if not conflict:
                    seq.append(v)
                    used.add(v)
                    prefix_sums.append(new_prefix)
                    for s in new_sums:
                        block_sums.add(s)
                    found = True
                    break
            if not found:
                break
        
        return seq
    
    # Get initial estimate
    greedy_seq = greedy_build()
    initial_len = len(greedy_seq)
    
    # Try a few different max_len targets
    best_seq = greedy_seq
    best_len = initial_len
    
    # Phase 1: Try building longer sequences with backtracking
    for target_len in range(initial_len + 1, min(initial_len + 15, 30)):
        if time.time() > deadline - 5:
            break
        budget = min(10, deadline - 5 - time.time())
        if budget < 1:
            break
        result = try_build_seq(target_len, budget)
        if len(result) > best_len:
            best_len = len(result)
            best_seq = result[:]
    
    # Phase 2: Try to improve with more aggressive search
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        # Try to build even longer sequences
        for target_len in range(best_len + 1, min(best_len + 10, 35)):
            if time.time() > deadline - 5:
                break
            budget = min(8, deadline - 5 - time.time())
            if budget < 1:
                break
            result = try_build_seq(target_len, budget)
            if len(result) > best_len:
                best_len = len(result)
                best_seq = result[:]
    
    return best_seq
