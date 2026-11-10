import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def is_valid_extension(seq, sums, x):
        """Check if x can be appended to seq without creating duplicate consecutive sums."""
        n = len(seq)
        total = x
        if total in sums:
            return None
        new_sums = [total]
        for i in range(n - 1, -1, -1):
            total += seq[i]
            if total in sums:
                return None
            new_sums.append(total)
        return new_sums
    
    def greedy_construct(strategy, seed_val=0):
        """Greedy construction with given strategy. Returns best sequence found."""
        if seed_val:
            random.seed(seed_val)
        
        seq = []
        sums = set()
        
        if strategy == 'asc':
            candidates = list(range(1, 257))
        elif strategy == 'desc':
            candidates = list(range(256, 0, -1))
        else:  # random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        
        for x in candidates:
            if seq and x <= seq[-1]:
                continue
            if not seq and x < 1:
                continue
            
            new_sums = is_valid_extension(seq, sums, x)
            if new_sums is not None:
                seq.append(x)
                sums.update(new_sums)
        
        return seq
    
    # Try multiple strategies
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # baseline length 9
    best_len = len(best_seq)
    
    # Strategy 1: Greedy ascending
    seq1 = greedy_construct('asc')
    if len(seq1) > best_len:
        best_seq = seq1
        best_len = len(seq1)
    
    # Strategy 2: Greedy descending (start with large numbers)
    seq2 = greedy_construct('desc')
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        seq_r = greedy_construct('random', seed)
        if len(seq_r) > best_len:
            best_seq = seq_r
            best_len = len(seq_r)
    
    # Strategy 4: Backtracking search from a good prefix
    # Use the best sequence found so far as a guide
    if time.time() < deadline - 10:
        def backtrack(seq, sums, last_val):
            nonlocal best_seq, best_len, deadline
            
            if time.time() > deadline:
                return
            
            if len(seq) > best_len:
                best_seq = seq[:]
                best_len = len(seq)
            
            # Pruning
            if len(seq) + (256 - last_val) <= best_len:
                return
            
            # Try adding next elements in increasing order
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                
                new_sums = is_valid_extension(seq, sums, x)
                if new_sums is not None:
                    seq.append(x)
                    sums.update(new_sums)
                    
                    backtrack(seq, sums, x)
                    
                    # Backtrack
                    for s in new_sums:
                        sums.discard(s)
                    seq.pop()
        
        # Start backtracking from a small prefix
        # Try starting with [1]
        backtrack([1], {1}, 1)
    
    # Strategy 5: More randomized restarts if time permits
    remaining = deadline - time.time()
    if remaining > 5:
        for seed in range(100, 200):
            if time.time() > deadline:
                break
            seq_r = greedy_construct('random', seed)
            if len(seq_r) > best_len:
                best_seq = seq_r
                best_len = len(seq_r)
    
    return best_seq
