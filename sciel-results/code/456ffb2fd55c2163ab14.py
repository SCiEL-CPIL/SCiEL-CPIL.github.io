import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    def is_valid_full(seq):
        """Check if all non-empty consecutive block sums are distinct. O(L^2)."""
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

    def get_sums(seq):
        """Calculate the set of all consecutive block sums."""
        sums = set()
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        for i in range(L):
            for j in range(i + 1, L + 1):
                sums.add(P[j] - P[i])
        return sums

    def greedy_increasing():
        """Standard greedy: add smallest possible next (append to end)."""
        values = []
        sums = set()
        for x in range(1, 257):
            # New sums if we append x:
            # x, x + v_k, x + v_k + v_{k-1}, ...
            new_sums = {x}
            current_suffix = 0
            for v in reversed(values):
                current_suffix += v
                new_sums.add(x + current_suffix)
                
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        return values

    def greedy_decreasing():
        """Greedy: add largest possible next (prepend to start)."""
        values = []
        sums = set()
        for x in range(256, 0, -1):
            # New sums if we prepend x:
            # x, x + v_1, x + v_1 + v_2, ...
            new_sums = {x}
            current_prefix = 0
            for v in values: # values is sorted ascending
                current_prefix += v
                new_sums.add(x + current_prefix)
                
            if not sums.intersection(new_sums):
                # Insert x at the beginning (since x < all values in values)
                values.insert(0, x)
                sums.update(new_sums)
        return values

    # 1. Get baseline from greedy
    seq_inc = greedy_increasing()
    seq_dec = greedy_decreasing()
    
    best_seq = seq_inc if len(seq_inc) >= len(seq_dec) else seq_dec
    best_len = len(best_seq)
    current_seq = list(best_seq)
    current_sums = get_sums(current_seq)
    
    # 2. Local Search with Incremental Updates
    # We maintain `current_sums` to avoid O(L^2) recalculation on every move.
    # When inserting x at index k:
    # New sums are sums of blocks that include x.
    # These are (sum of suffix ending at k-1) + x + (sum of prefix starting at k).
    # Actually, it's easier to think:
    # Any block including x is a concatenation of a suffix of current_seq[0:k] 
    # and a prefix of current_seq[k:], plus x? No.
    # Let's derive:
    # Seq: A, x, B.
    # Blocks including x:
    # 1. x
    # 2. x + b1
    # 3. x + b1 + b2
    # ...
    # 4. a_m + x
    # 5. a_m + x + b1
    # ...
    # Where a_m is suffix of A, b1... is prefix of B.
    # So new_sums = { suffix_A + x + prefix_B | suffix_A in suffixes(A) U {0}, prefix_B in prefixes(B) U {0} }
    # Wait, suffixes(A) should include empty? Yes, if suffix is empty, it's just x + prefix_B.
    # Let's define:
    # suffixes_A = {0} U {sum(A[i:]) for i in range(len(A))}
    # prefixes_B = {0} U {sum(B[:j]) for j in range(1, len(B)+1)}
    # new_sums = { s + x + p for s in suffixes_A for p in prefixes_B }
    # This is O(k * (L-k)) which is O(L^2) worst case. 
    # However, L is small (~10-15). 
    # But we can do better? 
    # Actually, the number of new sums is exactly (k+1)*(L-k+1).
    # For L=10, max is ~25. Very fast.
    
    # When removing x at index k:
    # We remove all sums that include x.
    # We can just recalculate if L is small, or subtract the set of sums including x.
    # Subtracting is safer if we have the set.
    
    all_candidates = list(range(1, 257))
    
    # Helper to calculate sums including a specific element in a sorted list
    def get_sums_including(seq, x):
        """
        Returns the set of all consecutive block sums in `seq` that include the value `x`.
        Assumes `x` is in `seq`.
        """
        k = seq.index(x)
        L = len(seq)
        sums = set()
        
        # Suffixes of seq[:k] (including empty)
        suffixes = [0]
        s = 0
        for i in range(k - 1, -1, -1):
            s += seq[i]
            suffixes.append(s)
            
        # Prefixes of seq[k+1:] (including empty)
        prefixes = [0]
        s = 0
        for i in range(k + 1, L):
            s += seq[i]
            prefixes.append(s)
            
        for suf in suffixes:
            for pre in prefixes:
                sums.add(suf + x + pre)
                
        return sums

    while time.time() < deadline:
        # Randomly decide move type
        # Bias towards adding to maximize length
        r = random.random()
        
        if r < 0.6: # Add
            # Pick a random candidate not in current_seq
            available = [x for x in all_candidates if x not in current_seq]
            if not available:
                continue
            x = random.choice(available)
            
            # Find insertion point
            lo, hi = 0, len(current_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if current_seq[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            k = lo
            
            # Calculate new sums
            # Suffixes of current_seq[:k]
            suffixes = [0]
            s = 0
            for i in range(k - 1, -1, -1):
                s += current_seq[i]
                suffixes.append(s)
                
            # Prefixes of current_seq[k:]
            prefixes = [0]
            s = 0
            for i in range(k, len(current_seq)):
                s += current_seq[i]
                prefixes.append(s)
                
            new_sums = set()
            for suf in suffixes:
                for pre in prefixes:
                    new_sums.add(suf + x + pre)
            
            # Check for conflicts
            if not current_sums.intersection(new_sums):
                # Accept move
                current_seq.insert(k, x)
                current_sums.update(new_sums)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = list(current_seq)
        
        elif r < 0.8: # Remove
            if not current_seq:
                continue
            x = random.choice(current_seq)
            k = current_seq.index(x)
            
            # Calculate sums to remove
            sums_to_remove = get_sums_including(current_seq, x)
            
            # Perform removal
            current_seq.pop(k)
            current_sums.difference_update(sums_to_remove)
            
        else: # Swap
            if not current_seq:
                continue
            available = [x for x in all_candidates if x not in current_seq]
            if not available:
                continue
            
            x_rem = random.choice(current_seq)
            x_add = random.choice(available)
            
            # We can try to do this as a Remove then Add, or direct check.
            # Direct check is complex. Let's simulate Remove then Add.
            # But we need to revert if Add fails.
            
            k_rem = current_seq.index(x_rem)
            sums_to_remove = get_sums_including(current_seq, x_rem)
            
            # Temporarily remove
            seq_temp = list(current_seq)
            seq_temp.pop(k_rem)
            sums_temp = set(current_sums)
            sums_temp.difference_update(sums_to_remove)
            
            # Find insertion point for x_add in seq_temp
            lo, hi = 0, len(seq_temp)
            while lo < hi:
                mid = (lo + hi) // 2
                if seq_temp[mid] < x_add:
                    lo = mid + 1
                else:
                    hi = mid
            k_add = lo
            
            # Calculate new sums for adding x_add to seq_temp
            suffixes = [0]
            s = 0
            for i in range(k_add - 1, -1, -1):
                s += seq_temp[i]
                suffixes.append(s)
                
            prefixes = [0]
            s = 0
            for i in range(k_add, len(seq_temp)):
                s += seq_temp[i]
                prefixes.append(s)
                
            new_sums = set()
            for suf in suffixes:
                for pre in prefixes:
                    new_sums.add(suf + x_add + pre)
            
            if not sums_temp.intersection(new_sums):
                # Accept swap
                seq_temp.insert(k_add, x_add)
                sums_temp.update(new_sums)
                
                current_seq = seq_temp
                current_sums = sums_temp
                
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = list(current_seq)

    return sorted(best_seq)
