import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    best_seq = []
    
    def dfs(seq, sums):
        nonlocal best_seq, start_time
        
        # Deadline check
        if time.time() - start_time > 85:
            return
        
        # Update best
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # If we can't add more (last is 256), return
        if not seq:
            last = 0
        else:
            last = seq[-1]
            
        if last >= 256:
            return
            
        # Try next values
        for x in range(last + 1, 257):
            # Pruning: Max possible length if we pick all remaining numbers
            # Current length: len(seq)
            # We add x. New length: len(seq) + 1
            # Max remaining numbers: 256 - x
            # Max total length: len(seq) + 1 + (256 - x)
            if len(seq) + 1 + (256 - x) <= len(best_seq):
                continue

            # Calculate new suffix sums
            suffix_sums = [x]
            t = 0
            for a in reversed(seq):
                t += a
                suffix_sums.append(t + x)
            
            # Check intersection
            # If any new sum is in existing sums, invalid
            valid = True
            for s in suffix_sums:
                if s in sums:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Recurse
            new_sums_set = sums.union(suffix_sums)
            dfs(seq + [x], new_sums_set)
            
            # Early exit if time is up
            if time.time() - start_time > 85:
                return

    dfs([], set())
    
    return best_seq
