import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe (90s kill, 60s eval limit)
    
    # Helper: Check if sequence has distinct consecutive block sums
    # Using prefix sums for faster computation
    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        
        sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper: Fast validity check for adding one element to a sorted sequence
    # Returns True if seq + [x] is valid, where seq is sorted
    def can_add(seq, x):
        n = len(seq)
        # Find insertion point
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        # Build new sequence
        new_seq = seq[:lo] + [x] + seq[lo:]
        return is_valid(new_seq)

    # Helper: Greedy fill a sequence with candidates in a specific order
    def greedy_fill(seq, order, time_limit):
        current = seq[:]
        current_set = set(current)
        
        for x in order:
            if time.time() > time_limit:
                break
            if x in current_set:
                continue
            
            # Try adding x
            new_seq = current + [x]
            new_seq.sort()
            
            if is_valid(new_seq):
                current = new_seq
                current_set.add(x)
                
        return current

    # Depth-first backtracking search with pruning
    def dfs_backtrack(current_seq, remaining_candidates, depth, max_depth, deadline):
        """
        Try to extend current_seq by picking from remaining_candidates.
        Returns the longest valid sequence found.
        """
        if time.time() > deadline:
            return current_seq
        
        best = current_seq[:]
        
        # Try each candidate
        for i, x in enumerate(remaining_candidates):
            if time.time() > deadline:
                break
            
            new_seq = current_seq + [x]
            new_seq.sort()
            
            if is_valid(new_seq):
                # Recurse with remaining candidates excluding x
                remaining = remaining_candidates[:i] + remaining_candidates[i+1:]
                result = dfs_backtrack(new_seq, remaining, depth+1, max_depth, deadline)
                if len(result) > len(best):
                    best = result
        
        return best

    # Initial Baselines
    best_seq = []
    
    # 1. Standard Ascending Greedy
    seq_asc = greedy_fill([], list(range(1, 257)), deadline)
    if len(seq_asc) > len(best_seq):
        best_seq = seq_asc
        
    # 2. Descending Greedy (Picks largest first)
    seq_desc = greedy_fill([], list(range(256, 0, -1)), deadline)
    if len(seq_desc) > len(best_seq):
        best_seq = seq_desc
        
    # 3. "Spaced" Greedy: Pick largest valid number at each step
    def greedy_largest_first():
        selected = []
        available = list(range(256, 0, -1))
        while available:
            found = False
            for x in available:
                candidate = selected + [x]
                candidate.sort()
                if is_valid(candidate):
                    selected = candidate
                    available.remove(x)
                    found = True
                    break
            if not found:
                break
        return selected

    seq_largest = greedy_largest_first()
    if len(seq_largest) > len(best_seq):
        best_seq = seq_largest

    # 4. DFS Backtracking from greedy results
    # Use the best greedy result as a starting point for DFS
    if len(best_seq) > 0:
        remaining = [x for x in range(1, 257) if x not in set(best_seq)]
        # Limit DFS depth to avoid explosion
        dfs_limit = min(55.0, time.time() + 15.0)
        seq_dfs = dfs_backtrack(best_seq, remaining, 0, 10, dfs_limit)
        if len(seq_dfs) > len(best_seq):
            best_seq = seq_dfs

    # 5. Random Restarts with Greedy Fill
    rng = random.Random(42)
    
    for _ in range(3):
        order = list(range(1, 257))
        rng.shuffle(order)
        seq_rand = greedy_fill([], order, deadline)
        if len(seq_rand) > len(best_seq):
            best_seq = seq_rand

    best_len = len(best_seq)
    all_candidates = set(range(1, 257))
    
    # Local Search / Randomized Improvement
    current_seq = best_seq[:]
    current_set = set(current_seq)
    
    stall_counter = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        r = rng.random()
        
        if r < 0.6:
            # Try to ADD a random unused element
            available = list(all_candidates - current_set)
            if not available:
                break
            
            x = rng.choice(available)
            new_seq = current_seq + [x]
            new_seq.sort()
            
            if is_valid(new_seq):
                current_seq = new_seq
                current_set.add(x)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                stall_counter += 1
                
        elif r < 0.8:
            # Try to REMOVE a random element
            if current_seq:
                x = rng.choice(current_seq)
                current_seq.remove(x)
                current_set.remove(x)
                stall_counter += 1
                
        else:
            # Try to SWAP
            if current_seq:
                x_to_remove = rng.choice(current_seq)
                available = list(all_candidates - current_set)
                if available:
                    x_to_add = rng.choice(available)
                    if x_to_add != x_to_remove:
                        new_seq = [v for v in current_seq if v != x_to_remove]
                        new_seq.append(x_to_add)
                        new_seq.sort()
                        if is_valid(new_seq):
                            current_seq = new_seq
                            current_set.remove(x_to_remove)
                            current_set.add(x_to_add)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = current_seq[:]
                                last_improvement_time = time.time()
                                stall_counter = 0
                            else:
                                stall_counter += 1
                        else:
                            stall_counter += 1
        
        # Periodic Greedy Polish
        if stall_counter > 20 or (time.time() - last_improvement_time > 1.0 and stall_counter > 5):
            polish_limit = min(deadline, time.time() + 2.0)
            
            test_seq = greedy_fill(current_seq, sorted(all_candidates - current_set), polish_limit)
            if len(test_seq) > len(current_seq):
                current_seq = test_seq
                current_set = set(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                stall_counter = 0
            
            if time.time() < deadline:
                test_seq = greedy_fill(current_seq, sorted(all_candidates - current_set, reverse=True), polish_limit)
                if len(test_seq) > len(current_seq):
                    current_seq = test_seq
                    current_set = set(current_seq)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = current_seq[:]
                        last_improvement_time = time.time()
                    stall_counter = 0
            
            if time.time() < deadline:
                rem = list(all_candidates - current_set)
                rng.shuffle(rem)
                test_seq = greedy_fill(current_seq, rem, polish_limit)
                if len(test_seq) > len(current_seq):
                    current_seq = test_seq
                    current_set = set(current_seq)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = current_seq[:]
                        last_improvement_time = time.time()
                    stall_counter = 0
            
            if stall_counter > 100:
                if len(current_seq) > 3:
                    to_remove = rng.sample(current_seq, 3)
                    for x in to_remove:
                        current_seq.remove(x)
                        current_set.remove(x)
                stall_counter = 0

    return best_seq

if __name__ == "__main__":
    res = construct()
    # print(len(res))
    pass
