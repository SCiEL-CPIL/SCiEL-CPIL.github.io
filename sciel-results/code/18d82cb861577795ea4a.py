import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Safety margin for 60s limit (or 90s if evaluator allows, but 60s is stated in prompt "Each evaluation must finish within 60 s")
    # The prompt says "evaluator KILLS the program after 90s wall-clock" in the directive, 
    # but the public task contract says "Each evaluation must finish within 60 s".
    # To be safe, I will use 55s to ensure return before 60s, which is well within 90s.

    best = []
    best_len = 0

    def get_block_sums(seq):
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        # Check if all consecutive block sums are distinct
        sums = get_block_sums(seq)
        # Number of blocks is n*(n+1)/2
        n = len(seq)
        return len(sums) == n * (n + 1) // 2

    # 1. Greedy Initialization
    # Try a few different greedy strategies to get a good starting point
    def greedy(descending=False):
        seq = []
        sums = set()
        # We iterate through potential values. 
        # If descending is True, we might want to pick large values to ensure distinctness?
        # But we need strictly increasing. So we pick values from 1 to 256.
        # Standard greedy: pick smallest valid x.
        # Alternative: pick largest valid x? No, that reduces length.
        # Let's stick to smallest valid x for max length potential, 
        # but maybe we can try "largest valid x" to see if it leads to a different, perhaps longer, 
        # sequence due to avoiding collisions? Unlikely to be longer, but different.
        
        for x in range(1, 257):
            if not seq:
                # First element
                if x not in sums:
                    seq.append(x)
                    sums.add(x)
            else:
                # Check new sums
                new_sums = []
                total = x
                # Sums ending at x: x, a_{n-1}+x, ...
                for a in reversed(seq):
                    total += a
                    new_sums.append(total)
                # Note: the loop above adds a_{n-1}+x, then a_{n-2}+a_{n-1}+x, etc.
                # We also need x itself. The loop `for a in reversed(seq)` adds a to total.
                # Start total=x.
                # Iter 1: total = x + seq[-1]. Append.
                # Iter 2: total = x + seq[-1] + seq[-2]. Append.
                # We missed x itself if seq is not empty? 
                # Wait, if seq is empty, we handled it.
                # If seq is not empty, the blocks ending at x are:
                # [x], [seq[-1], x], [seq[-2], seq[-1], x], ...
                # My loop:
                # total = x
                # a = seq[-1]: total += a -> x + seq[-1]. new_sums.append(total).
                # This misses [x].
                # Let's fix the logic.
                
                # Correct logic for new sums when appending x to seq:
                # Sums are:
                # x
                # seq[-1] + x
                # seq[-2] + seq[-1] + x
                # ...
                # sum(seq) + x
                
                candidate_sums = [x]
                running_sum = x
                for a in reversed(seq):
                    running_sum += a
                    candidate_sums.append(running_sum)
                
                if not sums.intersection(candidate_sums):
                    seq.append(x)
                    sums.update(candidate_sums)
        return seq

    # Run standard greedy (smallest first)
    g1 = greedy()
    if len(g1) > best_len:
        best = g1
        best_len = len(g1)
    
    # Try a variant: Greedy with some randomization or different start?
    # Let's try "Greedy from the right"? No, sequence must be increasing.
    
    # 2. DFS with Randomized Candidate Order
    # We will perform multiple DFS runs, each with a time budget.
    
    def dfs(prefix, current_sums, current_len):
        nonlocal best, best_len
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if current_len > best_len:
            best = list(prefix)
            best_len = current_len
            
        if not prefix:
            min_val = 1
        else:
            min_val = prefix[-1] + 1
            
        if min_val > 256:
            return
            
        # Pruning: Even if we add the max possible number of elements (1 per value), 
        # can we beat best_len?
        # Max additional elements = 256 - min_val + 1
        if current_len + (256 - min_val + 1) <= best_len:
            return

        # Generate candidates
        candidates = list(range(min_val, 257))
        
        # Heuristic: Shuffle candidates to explore different branches
        # Sometimes trying larger jumps first helps avoid local minima in sum collisions
        random.shuffle(candidates)
        
        # Optional: Sort by some heuristic? 
        # Let's try sorting by value descending to see if "large gaps" strategy works better in DFS
        # Or keep random. Random is safer for coverage.
        
        for d in candidates:
            if time.time() - start_time > TIME_LIMIT:
                return
            
            new_val = d
            
            # Check validity
            # New sums:
            # new_val
            # prefix[-1] + new_val
            # ...
            # sum(prefix) + new_val
            
            candidate_sums = [new_val]
            running_sum = new_val
            for a in reversed(prefix):
                running_sum += a
                candidate_sums.append(running_sum)
            
            # Check intersection
            # Optimization: Check intersection early
            valid = True
            for s in candidate_sums:
                if s in current_sums:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Recurse
            prefix.append(new_val)
            current_sums.update(candidate_sums)
            
            dfs(prefix, current_sums, current_len + 1)
            
            prefix.pop()
            current_sums.difference_update(candidate_sums)

    # Run multiple DFS instances
    # Each instance gets a slice of the time
    # We can't easily parallelize in single thread, so we run sequentially with random states
    
    num_restarts = 10
    time_per_restart = (TIME_LIMIT - (time.time() - start_time)) / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Reset random seed for reproducibility in this run? Or just use global random?
        # Using global random is fine.
        
        # Start DFS from empty or from a short prefix?
        # Starting from empty is most general.
        # However, starting from the greedy result might be faster to find improvements?
        # No, greedy result is a leaf. We need to explore siblings.
        # Starting from empty allows finding completely different structures.
        
        # To avoid wasting time on the exact same first few moves, 
        # we can force the first move to be random?
        # The shuffle in candidates handles this.
        
        dfs([], set(), 0)
        
        # If we found a significantly better solution, we might reduce remaining time for others?
        # Or just continue.
        
    return best

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    # Verify
    sums = set()
    valid = True
    for i in range(len(res)):
        s = 0
        for j in range(i, len(res)):
            s += res[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
