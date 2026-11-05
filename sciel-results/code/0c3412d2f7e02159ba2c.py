import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def incremental_valid_add(seq, x):
        """Check if x can be added to sorted seq, returning new sums or None if invalid.
        seq must be sorted. x is inserted at position k = bisect_left(seq, x).
        New sums are: x, x + suffixes of seq[k:], prefixes of seq[:k] + x, prefixes of seq[:k] + x + suffixes of seq[k:]
        Actually, simpler: new consecutive blocks involving x are:
        - [x] itself
        - [x] + suffix of seq[k:] (blocks starting at x and extending right)
        - prefix of seq[:k] + [x] (blocks ending at x and extending left)
        - prefix of seq[:k] + [x] + suffix of seq[k:] (blocks containing x with elements on both sides)
        
        Wait, that's not quite right. Let me think again.
        
        If we insert x at position k in sorted seq, the new consecutive blocks that include x are:
        - For i <= k <= j (0-indexed in the new list), the block seq[i:j+1] in the new list.
        - The new blocks are exactly those where the block in the old list would have been split by x.
        
        Actually, the cleanest way: the new consecutive block sums are:
        1. x alone
        2. x + (consecutive suffix of seq[k:]) - i.e., x + seq[k], x + seq[k]+seq[k+1], etc.
        3. (consecutive prefix of seq[:k]) + x - i.e., seq[k-1]+x, seq[k-2]+seq[k-1]+x, etc.
        4. (consecutive prefix of seq[:k]) + x + (consecutive suffix of seq[k:]) - i.e., blocks that span across x
        
        But actually, for a strictly increasing sequence, if we insert x at position k, the new consecutive blocks involving x are:
        - Starting at position k: x, x+seq[k], x+seq[k]+seq[k+1], ... (going right)
        - Ending at position k: seq[k-1]+x, seq[k-2]+seq[k-1]+x, ... (going left)
        - Spanning: seq[k-1]+x+seq[k], seq[k-2]+seq[k-1]+x+seq[k], etc.
        
        This is getting complex. Let me just use the simple approach: compute all new block sums involving x by considering all (i,j) pairs where the block [i,j] in the new list contains position k.
        
        For efficiency, I'll compute:
        - Left extensions: for i from k down to 0, sum = sum(seq[i:k]) + x
        - Right extensions: for j from k to L, sum = x + sum(seq[k:j])  
        - Both: for i from k down to 0, for j from k to L, sum = sum(seq[i:k]) + x + sum(seq[k:j])
        
        But that's O(L^2) in the worst case. For small L (up to ~20), this is fine.
        """
        import bisect
        k = bisect.bisect_left(seq, x)
        L = len(seq)
        
        new_sums = set()
        new_sums.add(x)
        
        # Left extensions: blocks ending at x
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            s = left_sum + x
            new_sums.add(s)
        
        # Right extensions: blocks starting at x
        right_sum = 0
        for j in range(k, L):
            right_sum += seq[j]
            s = x + right_sum
            new_sums.add(s)
        
        # Both sides: blocks containing x with elements on both sides
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            right_sum = 0
            for j in range(k, L):
                right_sum += seq[j]
                s = left_sum + x + right_sum
                new_sums.add(s)
        
        return new_sums

    def is_valid(seq):
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_all_sums(seq):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                sums.add(s)
        return sums

    # DFS with backtracking: try to build a sequence by choosing from available values
    # State: current_seq (sorted), available values, current set of sums
    # We try to add values in a smart order
    
    def dfs_backtrack(available, current_seq, sums, depth, max_depth, best_ref):
        """
        available: set of values not yet used
        current_seq: sorted list of chosen values
        sums: set of all consecutive block sums so far
        depth: current depth
        max_depth: target depth to reach
        best_ref: [best_seq, best_len] to update
        """
        if time.time() > deadline:
            return
        
        if len(current_seq) > best_ref[1]:
            best_ref[1] = len(current_seq)
            best_ref[0] = list(current_seq)
        
        if depth >= max_depth:
            return
        
        if not available:
            return
        
        # Try adding each available value
        # Sort available by some heuristic: try values that create fewer conflicts first
        # For now, try in a mixed order: alternating small and large
        candidates = sorted(available)
        
        # Try a subset of candidates to limit branching
        # If too many candidates, sample
        if len(candidates) > 20:
            # Take first 10, last 10, and some middle
            candidates = candidates[:10] + candidates[len(candidates)//2-5:len(candidates)//2+5] + candidates[-10:]
        
        for x in candidates:
            if time.time() > deadline:
                return
            
            # Check if x can be added
            import bisect
            k = bisect.bisect_left(current_seq, x)
            L = len(current_seq)
            
            new_sums = {x}
            
            # Left extensions
            left_sum = 0
            for i in range(k - 1, -1, -1):
                left_sum += current_seq[i]
                s = left_sum + x
                if s in sums:
                    break
                new_sums.add(s)
            else:
                # Right extensions
                right_sum = 0
                for j in range(k, L):
                    right_sum += current_seq[j]
                    s = x + right_sum
                    if s in sums or s in new_sums:
                        break
                    new_sums.add(s)
                else:
                    # Both sides
                    valid = True
                    left_sum = 0
                    for i in range(k - 1, -1, -1):
                        left_sum += current_seq[i]
                        right_sum = 0
                        for j in range(k, L):
                            right_sum += current_seq[j]
                            s = left_sum + x + right_sum
                            if s in sums or s in new_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        # Add x
                        new_seq = current_seq[:k] + [x] + current_seq[k:]
                        new_sums_set = sums | new_sums
                        available_copy = available - {x}
                        
                        dfs_backtrack(available_copy, new_seq, new_sums_set, depth + 1, max_depth, best_ref)
        
        return

    # Greedy initialization with incremental check
    def greedy_forward():
        values = []
        sums = set()
        for x in range(1, 257):
            k = len(values)  # x is always appended since we go in increasing order
            new_sums = {x}
            # Right extensions: x + suffixes of values
            right_sum = 0
            for v in reversed(values):
                right_sum += v
                s = x + right_sum
                new_sums.add(s)
            
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        return values

    def greedy_reverse():
        values = []
        sums = set()
        for x in range(256, 0, -1):
            # x will be prepended (smallest)
            k = 0
            new_sums = {x}
            # Left extensions: x + prefixes of values
            left_sum = 0
            for v in values:
                left_sum += v
                s = x + left_sum
                new_sums.add(s)
            
            if not sums.intersection(new_sums):
                values.insert(0, x)
                sums.update(new_sums)
        return values

    # Get initial best from greedy
    seq1 = greedy_forward()
    seq2 = greedy_reverse()
    
    best_seq = seq1 if len(seq1) >= len(seq2) else seq2
    best_len = len(best_seq)
    best_final = list(best_seq)
    
    # DFS to try to beat greedy
    all_values = set(range(1, 257))
    initial_sums = get_all_sums(best_seq)
    
    # Try DFS from empty with target depth = best_len + 2
    # But this might be too slow. Let's try from a smaller starting point.
    
    # Actually, let's use the DFS to explore from scratch with a limited depth
    # and see if we can find something longer
    best_ref = [list(best_seq), best_len]
    
    # Try DFS from empty, targeting best_len + 1
    # This is expensive, so limit it
    if time.time() < deadline - 10:
        dfs_backtrack(all_values, [], set(), 0, best_len + 2, best_ref)
    
    if best_ref[1] > best_len:
        best_len = best_ref[1]
        best_final = list(best_ref[0])
    
    # Local search from best_final
    current_seq = list(best_final)
    current_set = set(current_seq)
    all_candidates = list(range(1, 257))
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.7:  # Add
            available = [x for x in all_candidates if x not in current_set]
            if available:
                x = random.choice(available)
                new_seq = current_seq + [x]
                new_seq.sort()
                if is_valid(new_seq):
                    current_seq = new_seq
                    current_set = set(current_seq)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_final = list(current_seq)
        
        elif r < 0.9:  # Swap
            if current_seq:
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    new_seq = [x for x in current_seq if x != x_rem]
                    new_seq.append(x_add)
                    new_seq.sort()
                    if is_valid(new_seq):
                        current_seq = new_seq
                        current_set = set(current_seq)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_final = list(current_seq)
        
        else:  # Remove
            if len(current_seq) > 1:
                x_rem = random.choice(current_seq)
                current_seq.remove(x_rem)
                current_set = set(current_seq)
    
    return sorted(best_final)
