import sys
import time

sys.setrecursionlimit(10000)

def construct():
    # 1. Generate a baseline using the greedy algorithm
    values, sums = [], set()
    for x in range(1, 257):
        # Calculate suffix sums of current 'values'
        # suffix_sums[0] = 0
        # suffix_sums[i] = sum(values[n-i:])
        suffix_sums = [0]
        s = 0
        for a in reversed(values):
            s += a
            suffix_sums.append(s)
        
        # Check if x can be added
        valid = True
        new_sums = []
        for s in suffix_sums:
            ns = x + s
            if ns in sums:
                valid = False
                break
            new_sums.append(ns)
        
        if valid:
            values.append(x)
            sums.update(new_sums)
            
    best_len = len(values)
    best_seq = values[:]
    
    # 2. Try to find a longer sequence using DFS
    # We will try to find a sequence of length best_len + 1, then best_len + 2, etc.
    # To do this efficiently, we can run a DFS that tries to extend the sequence.
    # We use a global time limit to ensure we don't run out of time.
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for overhead
    
    # State: current sequence, set of sums, last value
    # We will try to build a sequence of length target_len.
    # If we fail for target_len, we stop.
    
    # Optimization: The greedy sequence is [1, 2, 4, 8, 16, 32, 64, 128, 256].
    # Let's see if we can find a sequence of length 10.
    
    def dfs(current_seq, current_sums, start_val, target_len):
        if time.time() - start_time > time_limit:
            return None
        
        if len(current_seq) == target_len:
            return current_seq[:]
        
        # Pruning: Not enough numbers left
        if len(current_seq) + (256 - start_val + 1) < target_len:
            return None
            
        # Pruning: If we can't beat the current best (if we were optimizing for max length generally)
        # Here we are targeting a specific length, so we just need to find *any* sequence of that length.
        
        last_val = current_seq[-1] if current_seq else 0
        
        # Calculate suffix sums for current_seq
        suffix_sums = [0]
        s = 0
        for a in reversed(current_seq):
            s += a
            suffix_sums.append(s)
            
        for x in range(start_val, 257):
            # Check validity
            valid = True
            new_sums_list = []
            for s in suffix_sums:
                ns = x + s
                if ns in current_sums:
                    valid = False
                    break
                new_sums_list.append(ns)
                
            if not valid:
                continue
                
            # Recurse
            next_sums = set(current_sums)
            next_sums.update(new_sums_list)
            
            res = dfs(current_seq + [x], next_sums, x + 1, target_len)
            if res is not None:
                return res
                
        return None

    # Try to find longer sequences
    target = best_len + 1
    while target <= 20: # Upper bound on length
        if time.time() - start_time > time_limit:
            break
            
        # To make DFS faster, we can use the greedy sequence as a guide?
        # Or just start from scratch?
        # Starting from scratch is safer but slower.
        # We can try to start with the prefix of the greedy sequence?
        # No, the optimal sequence might not share a prefix with the greedy one.
        
        # However, DFS from scratch might be too slow if the branching factor is high.
        # Let's add a heuristic: try x in a specific order.
        # Smaller x first?
        
        # Let's try to find a sequence of length 'target'.
        # We can use a modified DFS that prioritizes branches that keep the sequence "sparse" (powers of 2 like)
        # or "dense".
        
        # For now, let's just run the DFS.
        # To speed it up, we can sort the candidates?
        # The loop `for x in range(start_val, 257)` is already sorted.
        
        # One issue: DFS explores all possibilities. If there are no solutions of length 10, it will explore the entire tree of valid prefixes.
        # This could be huge.
        
        # Alternative: Use the greedy result as a seed for local search?
        # Or, realize that the greedy result is likely optimal or close to it.
        
        # Let's check if the greedy result is indeed the maximum.
        # The problem is equivalent to finding the longest sequence with distinct interval sums.
        # This is a known problem.
        # For n=9, [1, 2, 4, 8, 16, 32, 64, 128, 256] works.
        # Can we fit 10?
        # If we use 10 numbers, the smallest possible sum of the largest interval is sum of 10 smallest distinct ints >= 1.
        # 1+2+3+4+5+6+7+8+9+10 = 55.
        # The largest sum is sum of the 10 numbers.
        # The number of intervals is 55.
        # The sums must be distinct.
        
        # Let's try a different strategy:
        # Instead of blind DFS, we can use the fact that the greedy algorithm is very good.
        # We will just return the greedy sequence if the DFS is too slow or fails to find a longer one quickly.
        
        # Let's limit the DFS depth or nodes.
        # Actually, let's just try the DFS for a short time. If it finds a longer one, great.
        # If not, we return the greedy one.
        
        # To make the DFS effective, we should prune aggressively.
        # A strong bound: 
        # The sums of the intervals must be distinct.
        # The smallest possible sum for the k-th added element x is x.
        # The largest possible sum is sum(all).
        
        # Let's just run the DFS for target = best_len + 1.
        # If it finds one, we update best_seq and try target+1.
        
        # To avoid infinite loops or long runs, we add a node limit.
        nodes_visited = 0
        node_limit = 100000
        
        def dfs_limited(current_seq, current_sums, start_val, target_len):
            nonlocal nodes_visited
            if time.time() - start_time > time_limit:
                return None
            if nodes_visited > node_limit:
                return None
            nodes_visited += 1
            
            if len(current_seq) == target_len:
                return current_seq[:]
            
            if len(current_seq) + (256 - start_val + 1) < target_len:
                return None
                
            last_val = current_seq[-1] if current_seq else 0
            
            suffix_sums = [0]
            s = 0
            for a in reversed(current_seq):
                s += a
                suffix_sums.append(s)
                
            for x in range(start_val, 257):
                valid = True
                new_sums_list = []
                for s in suffix_sums:
                    ns = x + s
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums_list.append(ns)
                    
                if not valid:
                    continue
                    
                next_sums = set(current_sums)
                next_sums.update(new_sums_list)
                
                res = dfs_limited(current_seq + [x], next_sums, x + 1, target_len)
                if res is not None:
                    return res
                    
            return None

        res = dfs_limited([], set(), 1, target)
        if res is not None:
            best_seq = res
            best_len = len(res)
            target += 1
        else:
            break

    return best_seq
