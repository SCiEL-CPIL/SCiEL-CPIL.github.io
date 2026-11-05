import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_seq = []
    best_len = 0
    best_sums = set()
    
    def update_best(seq, sums):
        nonlocal best_seq, best_len, best_sums
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            best_sums = sums.copy()
            
    def can_add(seq, sums, x):
        if x in sums:
            return False, set()
        
        new_sums = {x}
        s = 0
        for i in range(len(seq) - 1, -1, -1):
            s += seq[i]
            new_sum = s + x
            if new_sum in sums or new_sum in new_sums:
                return False, set()
            new_sums.add(new_sum)
        
        return True, new_sums

    def dfs(seq, sums, next_min, depth):
        nonlocal best_len, best_seq, best_sums
        if time.time() > deadline:
            return
        
        update_best(seq, sums)
        
        # Pruning
        if len(seq) + (256 - next_min + 1) <= best_len:
            return
            
        # To avoid getting stuck in one deep path, we limit the time spent 
        # in a single branch by checking time frequently.
        
        for x in range(next_min, 257):
            if time.time() > deadline:
                return
            
            # Heuristic: If we are deep and time is running out, 
            # maybe try larger x's first? No, smaller x's are usually better for packing.
            
            valid, new_sums = can_add(seq, sums, x)
            if valid:
                seq.append(x)
                sums.update(new_sums)
                
                dfs(seq, sums, x + 1, depth + 1)
                
                # Backtrack
                seq.pop()
                sums.difference_update(new_sums)
                
                # If we found a better solution, we can prune this branch more aggressively?
                # The pruning at the start of the next iteration handles this.
                
                # If we just found the best possible length (256), return.
                if best_len == 256:
                    return
                
                # If we are at a high depth and have spent a lot of time, 
                # we might want to break and try a different start?
                # For now, let the time limit handle it.
                
                # Optimization: If x is small, we are likely to find a long sequence.
                # If x is large, we are likely to find a shorter one.
                # So we try small x first.
                
                # If we have tried all x's, we return.
                
                # One more optimization: 
                # If len(seq) == best_len, we can break early?
                # No, we want to see if we can go deeper.
                
                # But if len(seq) + (256 - x) < best_len, we can break.
                if len(seq) + (256 - x) < best_len:
                    break

    # Run the main DFS
    dfs([], set(), 1, 0)
    
    # If time remains, run a few randomized greedy passes to see if we beat the DFS
    # (DFS might have gotten stuck in a local minimum if it was too deep)
    if time.time() < deadline - 2:
        for _ in range(20):
            if time.time() > deadline:
                break
            
            seq = []
            sums = set()
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                valid, new_sums = can_add(seq, sums, x)
                if valid:
                    # Insert in sorted order
                    # Since we are iterating over shuffled candidates, we need to maintain sorted order in seq
                    # But can_add assumes x is appended to the end?
                    # No, can_add checks sums ending at x. 
                    # If we insert x in the middle, the sums change for blocks after x too.
                    # My can_add only handles APPENDING.
                    # So for randomized greedy, I must ensure I only append?
                    # No, the sequence must be strictly increasing.
                    # So if I pick x, and seq is [2, 5], and I pick 3, I must insert it between 2 and 5.
                    # This invalidates the simple "append" check.
                    
                    # So, for the randomized greedy, I should just pick the next smallest valid number?
                    # That's just the standard greedy.
                    
                    # To do randomized greedy properly, I need a general insertion check.
                    # That is O(N^2).
                    
                    # Let's skip the complex randomized greedy and rely on the DFS.
                    # The DFS explores the space of "append" choices, which covers all strictly increasing sequences.
                    # Because any strictly increasing sequence can be built by appending elements in sorted order.
                    # So the DFS is complete. It just might be slow.
                    
                    pass

    return best_seq
