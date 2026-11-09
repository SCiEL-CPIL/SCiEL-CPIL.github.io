import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    def _greedy():
        """
        Greedy construction: pick smallest x that doesn't conflict.
        """
        values = []
        sums = set()
        for x in range(1, 257):
            # Calculate new sums introduced by x
            # Blocks ending at x: [x], [prev, x], [prev2, prev, x], ...
            new_sums = []
            s = x
            new_sums.append(s)
            for a in reversed(values):
                s += a
                new_sums.append(s)
            
            # Check for conflicts
            conflict = False
            for ns in new_sums:
                if ns in sums:
                    conflict = True
                    break
            
            if not conflict:
                values.append(x)
                sums.update(new_sums)
                
        return values

    # Get baseline
    best_seq = _greedy()
    best_len = len(best_seq)

    # Optimization: Use bitsets for sum tracking in DFS
    # Max sum for length 15 is approx 15 * 256 = 3840
    # Python integers handle this easily.

    def _dfs(vals, last_val, used_bits, suffix_sums, depth, target_len):
        """
        vals: current sequence
        last_val: last element in vals
        used_bits: bitset of all consecutive sums used so far
        suffix_sums: list of suffix sums of vals. suffix_sums[i] = sum(vals[i:])
        depth: current length of vals
        target_len: desired length
        """
        if time.time() > deadline:
            return None
        
        if depth == target_len:
            return vals[:]

        # Pruning: Not enough numbers left to reach target_len
        # We need (target_len - depth) more numbers, all > last_val
        if 256 - last_val < target_len - depth:
            return None

        # Try next values
        # Heuristic: Try smaller values first to keep sums small?
        # Or larger values?
        # Smaller values are generally "safer" as they create smaller sums,
        # which are less likely to collide with the large sums created by later large numbers?
        # Actually, it's complex. Let's stick to increasing order.
        
        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            # New suffix sums: [x, suffix_sums[0] + x, suffix_sums[1] + x, ...]
            # Wait, suffix_sums[0] is sum(vals[0:]), which is the total sum.
            # The new sequence is vals + [x].
            # The new suffix sums are:
            # S_new[k] = x
            # S_new[i] = S_old[i] + x for i in 0..k-1
            
            new_suffix_sums = [x]
            # We need to check collisions for these new sums.
            # The new sums are exactly the elements of new_suffix_sums.
            
            # Check collisions with used_bits
            mask = 0
            valid = True
            for s_val in new_suffix_sums:
                bit = 1 << s_val
                if used_bits & bit:
                    valid = False
                    break
                mask |= bit
            
            if not valid:
                continue

            # Add existing suffix sums + x to new_suffix_sums and mask
            # Note: We already added x. Now add the rest.
            for i in range(len(suffix_sums)):
                s_val = suffix_sums[i] + x
                # s_val is guaranteed to be distinct from previous new_suffix_sums?
                # Yes, because suffix_sums are strictly decreasing (since vals > 0).
                # So s_val + x are strictly decreasing.
                # And x is smaller than any suffix_sum + x?
                # suffix_sums[i] >= vals[i] >= 1. So suffix_sums[i] + x > x.
                # So no internal collision in new_suffix_sums.
                
                bit = 1 << s_val
                if used_bits & bit:
                    valid = False
                    break
                mask |= bit
            
            if not valid:
                continue

            # Recurse
            # Construct new_suffix_sums for recursion
            # Order doesn't matter for the check, but let's keep it consistent.
            # The list should represent suffix sums of the new sequence.
            # New sequence: v0...vk-1, x
            # Suffix sums: 
            # index k: x
            # index k-1: v_{k-1} + x
            # ...
            # index 0: sum(v0...vk-1) + x
            
            # In our current suffix_sums list for vals:
            # suffix_sums[0] is sum(vals)
            # suffix_sums[-1] is vals[-1]
            
            # So new_suffix_sums for the new list should be:
            # [sum(vals)+x, ..., vals[-1]+x, x]
            # This matches the order if we prepend the old ones + x.
            
            new_suffix_list = [s + x for s in suffix_sums]
            new_suffix_list.append(x)
            
            res = _dfs(vals + [x], x, used_bits | mask, new_suffix_list, depth + 1, target_len)
            if res is not None:
                return res

        return None

    # Try to find longer sequences
    # Start from scratch to allow structural changes that greedy might have missed
    # We try increasing target lengths
    
    # Initial setup for DFS from scratch
    # vals = [], last_val = 0, used_bits = 0, suffix_sums = []
    
    # We know best_len is possible. Try best_len + 1.
    # If that fails, we can stop, as finding a longer sequence is harder.
    # However, sometimes a different structure of length L+1 exists even if greedy failed to extend.
    
    # To save time, we can also try to extend the greedy sequence specifically?
    # The DFS from scratch covers that, but might be slower if the search space is large.
    # Given N=256, DFS from scratch for L=12 is likely fast enough with pruning.
    
    for target_len in range(best_len + 1, 16):  # Cap at 15/16
        if time.time() > deadline - 5:
            break
        
        # Reset search
        # We can use a small heuristic: if we found a sequence of length L, 
        # we can try to use the first few elements of the best found sequence as a prefix?
        # No, let's just run DFS.
        
        # Optimization: If we have a known valid sequence of length best_len,
        # we can try to extend it directly first (faster).
        # But DFS from scratch is more robust.
        
        res = _dfs([], 0, 0, [], 0, target_len)
        
        if res is not None:
            best_seq = res
            best_len = target_len
        else:
            # If we couldn't find length target_len, we likely can't find target_len + 1 quickly
            # unless the search was cut off by time.
            # If it was cut off by time, we should break.
            # If it returned None (exhausted search), we break.
            break

    return best_seq
