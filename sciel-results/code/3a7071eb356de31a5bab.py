import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256

    def get_consecutive_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_consecutive_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    # ---- NEW: Branch-and-bound with prefix-sum collision tracking ----
    # State: (seq, prefix_sums_set, total_sum)
    # When appending x to seq, new block sums are:
    #   x, x+last, x+last+second_last, ..., x+total_sum
    # These must not collide with existing prefix-sum differences.
    # 
    # Key insight: maintain the set of all consecutive-block-sums explicitly.
    # For a sequence [a1, a2, ..., an], the consecutive block sums are
    # prefix[j] - prefix[i] for 0 <= i < j <= n, where prefix[0]=0.
    # So we track the set of all (prefix[j] - prefix[i]) values.
    #
    # When we append x, new prefix sum = total + x.
    # New block sums: (total + x) - prefix[i] for i in 0..n
    # i.e., x, x + (total - a_n), x + (total - a_n - a_{n-1}), ..., x + total

    def branch_and_bound(seq, sums_set, total, lo, hi, best_len, time_limit, rng):
        """
        Try to extend seq by picking next value from [lo, hi].
        sums_set contains all consecutive block sums of seq.
        total is the sum of all elements in seq.
        Returns (best_seq, best_len) found.
        """
        current = list(seq)
        current_sums = set(sums_set)
        current_total = total
        cur_len = len(current)
        
        if cur_len >= best_len and time.time() > time_limit - 5.0:
            return current, cur_len

        # Try to add one more element
        # Upper bound: we can add at most (MAX_VAL - lo) / 1 more elements
        # but more tightly, we need distinct sums, and each new element adds
        # len(current)+1 new sums, so max additional elements is bounded by
        # how many sums we can still fit.
        
        # Greedy upper bound: at each step, the smallest new value must be > last
        # and all new sums must be distinct from existing ones.
        # A simple bound: remaining values in [lo, MAX_VAL] is MAX_VAL - lo + 1
        # But we also need strictly increasing, so at most MAX_VAL - current[-1] elements.
        upper_bound = cur_len + (MAX_VAL - current[-1]) if current else cur_len + MAX_VAL
        
        if upper_bound <= best_len:
            return current, cur_len

        # Try values from hi down to lo (larger values first give more headroom for sums)
        # Actually, try both directions with randomization
        candidates = list(range(lo, hi + 1))
        if rng.random() < 0.5:
            candidates.reverse()
        
        for x in candidates:
            if time.time() > time_limit - 2.0:
                break
            
            # Compute new block sums when appending x
            # New sums: x, x + suffix_sum_1, x + suffix_sum_2, ..., x + total
            # where suffix_sum_k = sum of last k elements
            new_sums = set()
            s = x
            new_sums.add(s)
            # suffix sums: sum of last 1, last 2, ..., last n elements
            suffix = 0
            for a in reversed(current):
                suffix += a
                s = x + suffix
                new_sums.add(s)
            
            # Check collision
            if new_sums & current_sums:
                continue
            
            # Also check: all new sums must be distinct among themselves
            # (they are, since suffix sums are strictly increasing for positive elements)
            
            new_seq = current + [x]
            new_sums_set = current_sums | new_sums
            new_total = current_total + x
            
            cur_len = len(new_seq)
            if cur_len > best_len:
                best_len = cur_len
            
            # Recurse
            new_lo = x + 1
            if new_lo > MAX_VAL:
                continue
            best_seq, best_len = branch_and_bound(
                new_seq, new_sums_set, new_total, new_lo, MAX_VAL,
                best_len, time_limit, rng
            )
            
            # Backtrack
            current = current[:-1]
            current_sums = set(current_sums)  # restore (we passed by value)
            current_total = current_total  # unchanged
        
        return current, cur_len

    # ---- Improved: iterative deepening B&B with better pruning ----
    def bb_search(seq, time_limit, rng):
        """Branch and bound starting from seq."""
        current = list(seq)
        # Build sums set
        sums_set = get_consecutive_sums(current)
        total = sum(current)
        cur_len = len(current)
        
        if not current:
            lo = 1
        else:
            lo = current[-1] + 1
        
        if lo > MAX_VAL:
            return current, cur_len
        
        best_seq = list(current)
        best_len = cur_len
        
        # Recursive helper with explicit stack to avoid recursion limit
        # Use iterative approach with backtracking
        def dfs(seq, sums, tot, lo_val):
            nonlocal best_seq, best_len
            if time.time() > time_limit - 1.0:
                return
            n = len(seq)
            if n >= best_len:
                best_seq = list(seq)
                best_len = n
            
            # Upper bound: max additional elements
            if n > 0:
                max_add = MAX_VAL - seq[-1]
            else:
                max_add = MAX_VAL
            if n + max_add <= best_len:
                return
            
            # Try candidate values
            # Heuristic: try values that maximize "spread" - larger gaps
            # Try from high to low first (more sum headroom)
            vals = list(range(lo_val, MAX_VAL + 1))
            # Randomize order slightly for diversity
            if rng.random() < 0.3:
                rng.shuffle(vals)
            
            for x in vals:
                if time.time() > time_limit - 1.0:
                    return
                # New sums when appending x
                new_sums = set()
                s = x
                new_sums.add(s)
                suffix = 0
                for a in reversed(seq):
                    suffix += a
                    s = x + suffix
                    new_sums.add(s)
                
                if new_sums & sums:
                    continue
                
                # Also need x > seq[-1] (ensured by lo_val)
                new_seq = seq + [x]
                new_sums_set = sums | new_sums
                new_total = tot + x
                
                dfs(new_seq, new_sums_set, new_total, x + 1)
        
        dfs(current, sums_set, total, lo)
        return best_seq, best_len

    # ---- Initial greedy seeds ----
    def greedy_desc(start=MAX_VAL, step=1):
        vals = []
        sums = set()
        for x in range(start, 0, -step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        vals.sort()
        return vals

    def greedy_asc(start=1, step=1):
        vals = []
        sums = set()
        for x in range(start, MAX_VAL + 1, step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        return vals

    # ---- Build diverse candidate pool ----
    candidates = []
    candidates.append(greedy_desc())
    candidates.append(greedy_asc())
    for step in (2, 3, 4):
        candidates.append(greedy_desc(step=step))
        candidates.append(greedy_asc(step=step))

    best_seq = max(candidates, key=len)
    best_len = len(best_seq)

    # ---- Branch-and-bound refinement on top candidates ----
    rng = random.Random(42)
    time_per_bb = (TIME_LIMIT - 15.0) / max(1, len(candidates))
    
    for seq in sorted(candidates, key=len, reverse=True):
        if time.time() > TIME_LIMIT - 5.0:
            break
        if len(seq) < best_len - 1:
            continue
        t_start = time.time()
        t_end = min(t_start + time_per_bb, TIME_LIMIT - 3.0)
        ext_seq, ext_len = bb_search(list(seq), t_end, rng)
        if ext_len > best_len:
            best_seq, best_len = ext_seq, ext_len

    # Final push on the best
    if time.time() < TIME_LIMIT - 3.0:
        ext_seq, ext_len = bb_search(list(best_seq), TIME_LIMIT - 1.0, rng)
        if ext_len > best_len:
            best_seq, best_len = ext_seq, ext_len

    return best_seq
