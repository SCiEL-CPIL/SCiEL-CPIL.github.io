import time
import sys

def construct():
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin for 60s limit
    
    best_seq = []
    
    # Initial state
    # We can start with an empty sequence
    # But we know 1,2,4... is a valid start.
    # Let's just start DFS from empty.
    
    # To speed up, we can use a recursive function
    # sys.setrecursionlimit(10000)
    
    def dfs(start_val, seq, sums, suffix_sums):
        nonlocal best_seq
        nonlocal start_time
        
        # Check time
        if time.time() - start_time > time_limit:
            return
            
        # Update best
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # Pruning: even if we pick all remaining numbers, can we beat best?
        # Max possible length from here is len(seq) + (256 - start_val + 1)
        if len(seq) + (256 - start_val + 1) <= len(best_seq):
            return
            
        # Try next values
        for x in range(start_val, 257):
            # Calculate new suffix sums
            # new_suffix_sums[0] = x
            # new_suffix_sums[i] = x + suffix_sums[i-1]
            # We can check collisions on the fly
            
            new_suffix_sums = [x]
            # Check if x is in sums
            if x in sums:
                continue
                
            valid = True
            for s in suffix_sums:
                new_s = x + s
                new_suffix_sums.append(new_s)
                if new_s in sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If we get here, x is a valid next element
            # New sums set
            new_sums = sums.union(new_suffix_sums)
            
            # Recurse
            dfs(x + 1, seq + [x], new_sums, new_suffix_sums)
            
            # Optional: If we found a solution that uses up most of the range, 
            # we might break early? No, we want to explore other branches too.
            
    # Start DFS
    # We can start with 1
    # Or we can try to be smarter.
    # Starting with 1 is generally good for maximizing length.
    dfs(1, [], set(), [])
    
    return best_seq

if __name__ == "__main__":
    print(construct())
