import time

def construct():
    start_time = time.time()
    # Leave a safety margin for Python overhead and evaluation script
    deadline = start_time + 85.0
    
    best_seq = []
    
    def dfs(vals, suffix_sums, sums, depth):
        nonlocal best_seq
        
        # Time check
        if time.time() > deadline:
            return
            
        # Update best if we have a longer sequence
        if depth > len(best_seq):
            best_seq = vals[:]
            
        # Pruning: Even if we pick every remaining integer, can we beat the best?
        last_val = vals[-1] if vals else 0
        max_possible_additional = 256 - last_val
        if depth + max_possible_additional <= len(best_seq):
            return
            
        # Try next values
        # We want to find longer sequences, so we prefer smaller x to allow more steps
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Check for collisions
            # New sums are x, x + suffix_sums[0], x + suffix_sums[1], ...
            # suffix_sums[0] is the last element a_k
            # suffix_sums[i] is sum of last i+1 elements
            
            new_sums = []
            valid = True
            
            # Check x
            if x in sums:
                valid = False
            else:
                new_sums.append(x)
                # Check x + s for each s in suffix_sums
                for s in suffix_sums:
                    ns = x + s
                    if ns in sums:
                        valid = False
                        break
                    new_sums.append(ns)
                    
            if not valid:
                continue
                
            # Recurse
            dfs(vals + [x], new_sums, sums.union(new_sums), depth + 1)
            
    dfs([], [], set(), 0)
    
    return best_seq
