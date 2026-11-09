import sys
from functools import lru_cache

def construct():
        # 1. Greedy baseline
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        greedy_len = len(values)
        greedy_seq = values
        
        # 2. Try to find a longer sequence
        # We search for a sequence of length greedy_len + 1
        target_len = greedy_len + 1
        
        # If greedy_len is already very high, maybe skip?
        # Max possible length is roughly sqrt(2 * 256 * something)
        # Let's just try.
        
        found = None
        # Use a recursive function with a node counter
        node_count = 0
        MAX_NODES = 200000 # Safety limit
        
        def backtrack(current_seq, current_sums, last_val):
            nonlocal found, node_count
            if found is not None:
                return
            node_count += 1
            if node_count > MAX_NODES:
                return
            
            if len(current_seq) == target_len:
                found = current_seq
                return
            
            # Pruning: Not enough numbers left
            if 256 - last_val < target_len - len(current_seq):
                return
            
            # Try next numbers
            # Heuristic: Try smaller numbers first?
            # If we try smaller, we fill up the low end.
            # If we try larger, we might skip over "bad" numbers.
            # Let's try smaller first, as it leaves more room.
            
            for x in range(last_val + 1, 257):
                # Calculate new sums
                new_sums = []
                s = 0
                for a in reversed(current_seq):
                    s += a
                    new_sums.append(s)
                # Note: new_sums are x, x+last, x+last+second_last...
                # Wait, the loop above adds a to s.
                # First iter: s = x + last.
                # We need x itself too.
                # Let's re-verify.
                # Suffix sums of [ ... , last, x]
                # x
                # x + last
                # x + last + second_last
                # ...
                
                # My loop:
                # a = last. s = x + last.
                # a = second_last. s = x + last + second_last.
                # It misses x.
                
                # Correct calculation:
                new_sums = [x]
                s = x
                for a in reversed(current_seq):
                    s += a
                    new_sums.append(s)
                
                # Check intersection
                if any(ns in current_sums for ns in new_sums):
                    continue
                
                new_sums_set = current_sums.union(new_sums)
                backtrack(current_seq + [x], new_sums_set, x)
                if found is not None:
                    return
        
        backtrack([], set(), 0)
        
        if found:
            return found
        else:
            return greedy_seq

    # Test locally? No, just output.
