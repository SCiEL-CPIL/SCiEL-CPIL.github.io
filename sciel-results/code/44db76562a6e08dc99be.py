import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    best_seq = []
    best_len = 0
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            return True
        return False

    def is_valid_append(seq, sums, suf_sums, x):
        if seq and x <= seq[-1]:
            return False, None, None
        # New sums are x + s for s in suf_sums (which includes 0)
        # We need to check if any new sum is already in sums
        new_sums = []
        for s in suf_sums:
            ns = x + s
            if ns in sums:
                return False, None, None
            new_sums.append(ns)
        
        # Also, the new sums themselves must be distinct?
        # Since x is fixed and s values are distinct (as they are sums of distinct suffixes of a strictly increasing sequence? 
        # Wait, are suffix sums distinct?
        # If seq is strictly increasing, suffix sums are strictly increasing.
        # suf_k < suf_{k-1} < ... < suf_1.
        # So x + suf_k < x + suf_{k-1} < ...
        # So new_sums are distinct.
        # And they are distinct from old sums by the check above.
        
        # Update suf_sums
        # New suffixes of seq + [x]:
        # [x] -> sum x
        # [a_k, x] -> sum suf_k + x
        # ...
        # [a_1...a_k, x] -> sum suf_1 + x
        # The set of suffix sums for the new sequence is {x} U {s + x for s in old_suf_sums if s != 0?}
        # Actually, old_suf_sums contains 0.
        # If we include 0 in old_suf_sums, then {x + s for s in old_suf_sums} covers x (when s=0) and all other suffix+x.
        # So new_suf_sums = {x + s for s in old_suf_sums}
        
        new_suf_sums = {x + s for s in suf_sums}
        return True, new_sums, new_suf_sums

    def is_valid_prepend(seq, sums, pref_sums, x):
        if seq and x >= seq[0]:
            return False, None, None
        # New sums are x + p for p in pref_sums (which includes 0)
        # Pref sums of seq: sums of prefixes.
        # p_1 = a_1
        # p_2 = a_1 + a_2
        # ...
        # p_k = sum(seq)
        # Plus 0.
        new_sums = []
        for p in pref_sums:
            ns = x + p
            if ns in sums:
                return False, None, None
            new_sums.append(ns)
        
        # New prefix sums of [x] + seq:
        # [x] -> x
        # [x, a_1] -> x + p_1
        # ...
        # [x, a_1...a_k] -> x + p_k
        # So new_pref_sums = {x + p for p in old_pref_sums}
        
        new_pref_sums = {x + p for p in pref_sums}
        return True, new_sums, new_pref_sums

    def dfs(seq, sums, suf_sums, pref_sums, depth, max_depth):
        nonlocal best_seq, best_len
        nonlocal start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return

        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

        if depth >= max_depth:
            return

        # Try appending
        # Generate candidates
        # To make it faster, we don't check all 256 numbers every time.
        # We can limit the range or just iterate. 256 is small.
        
        # Heuristic: Try candidates in a random order to explore different branches
        # But we need to ensure we don't miss the optimal path.
        # Since we are doing multiple randomized runs, this is fine.
        
        # Optimization: Only check candidates that are "likely" to work?
        # For now, iterate all valid x.
        
        valid_appends = []
        min_val = seq[-1] + 1 if seq else 1
        for x in range(min_val, 257):
            valid, ns, nsuf = is_valid_append(seq, sums, suf_sums, x)
            if valid:
                valid_appends.append((x, ns, nsuf))
                
        valid_prepends = []
        max_val = seq[0] - 1 if seq else 256
        for x in range(1, max_val + 1):
            valid, ns, npref = is_valid_prepend(seq, sums, pref_sums, x)
            if valid:
                valid_prepends.append((x, ns, npref))
                
        # Randomize order to avoid local optima
        random.shuffle(valid_appends)
        random.shuffle(valid_prepends)
        
        # Limit branching factor to keep DFS fast
        # If we have too many candidates, pick a random subset
        branch_limit = 5
        if len(valid_appends) > branch_limit:
            valid_appends = random.sample(valid_appends, branch_limit)
        if len(valid_prepends) > branch_limit:
            valid_prepends = random.sample(valid_prepends, branch_limit)

        # Recurse on appends
        for x, ns, nsuf in valid_appends:
            new_seq = seq + [x]
            new_sums = sums.union(ns)
            # Update pref_sums for the new sequence?
            # If we append x, the prefix sums of the new sequence are:
            # Old prefix sums, plus the total sum of the new sequence.
            # Total sum of new seq = sum(seq) + x
            # sum(seq) is the largest element in pref_sums (if we stored sums of prefixes).
            # Let's maintain total_sum separately or derive it.
            # Actually, pref_sums contains sums of prefixes.
            # The new prefixes are the old prefixes, and the whole sequence.
            # So new_pref_sums = old_pref_sums U {total_sum + x}
            # We need total_sum.
            # Let's pass total_sum or calculate it.
            # sum(seq) is the max of pref_sums? No, pref_sums are cumulative.
            # The last prefix sum is the total sum.
            # So total_sum = max(pref_sums) if pref_sums is not empty?
            # Wait, pref_sums includes 0. So total_sum = max(pref_sums).
            
            total_sum = max(pref_sums) if pref_sums else 0
            new_total = total_sum + x
            new_pref_sums = pref_sums.union({new_total})
            
            dfs(new_seq, new_sums, nsuf, new_pref_sums, depth + 1, max_depth)
            
            if time.time() - start_time > time_limit:
                return

        # Recurse on prepends
        for x, ns, npref in valid_prepends:
            new_seq = [x] + seq
            new_sums = sums.union(ns)
            # Update suf_sums for the new sequence?
            # If we prepend x, the suffix sums of the new sequence are:
            # Old suffix sums, plus the total sum of the new sequence.
            # total_sum = max(suf_sums)
            total_sum = max(suf_sums) if suf_sums else 0
            new_total = total_sum + x
            new_suf_sums = suf_sums.union({new_total})
            
            dfs(new_seq, new_sums, new_suf_sums, npref, depth + 1, max_depth)
            
            if time.time() - start_time > time_limit:
                return

    # Run multiple randomized DFS trials
    # Each trial starts from empty sequence
    # We can also try starting from a greedy solution to improve it
    
    # 1. Get a greedy baseline
    def greedy_baseline(reverse=False):
        seq = []
        sums = set()
        suf_sums = {0}
        pref_sums = {0}
        
        if reverse:
            for x in range(256, 0, -1):
                valid, ns, npref = is_valid_prepend(seq, sums, pref_sums, x)
                if valid:
                    seq.insert(0, x)
                    sums.update(ns)
                    pref_sums = npref
                    # Update suf_sums
                    total_sum = max(suf_sums) if suf_sums else 0
                    new_total = total_sum + x
                    suf_sums = suf_sums.union({new_total})
        else:
            for x in range(1, 257):
                valid, ns, nsuf = is_valid_append(seq, sums, suf_sums, x)
                if valid:
                    seq.append(x)
                    sums.update(ns)
                    suf_sums = nsuf
                    # Update pref_sums
                    total_sum = max(pref_sums) if pref_sums else 0
                    new_total = total_sum + x
                    pref_sums = pref_sums.union({new_total})
                    
        return seq, sums, suf_sums, pref_sums

    # Try greedy baselines
    seq1, sums1, suf1, pref1 = greedy_baseline(reverse=False)
    seq2, sums2, suf2, pref2 = greedy_baseline(reverse=True)
    
    if len(seq1) > len(best_seq):
        best_seq = seq1
        best_len = len(seq1)
    if len(seq2) > len(best_seq):
        best_seq = seq2
        best_len = len(seq2)

    # Now run randomized DFS from empty and from greedy results
    # To save time, we will run DFS from empty with a moderate depth limit
    # and from greedy results to try to extend/improve
    
    # Trial 1: DFS from empty
    dfs([], set(), {0}, {0}, 0, 20)
    
    # Trial 2: DFS extending greedy1
    if time.time() - start_time < time_limit - 5:
        dfs(list(seq1), sums1, suf1, pref1, 0, 10)
        
    # Trial 3: DFS extending greedy2
    if time.time() - start_time < time_limit - 5:
        dfs(list(seq2), sums2, suf2, pref2, 0, 10)

    # More randomized trials from empty
    while time.time() - start_time < time_limit - 2:
        # Random seed for this trial
        random.seed(int(time.time() * 1000) % 10000)
        dfs([], set(), {0}, {0}, 0, 15)
        
    return best_seq

