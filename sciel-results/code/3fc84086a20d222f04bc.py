import time
import random

def _try_append(seq, sums, x):
    n = len(seq)
    if n == 0:
        if x in sums:
            return False, None
        return True, [x]
    
    # Check new sums: x + suffix_sums
    suffix_sum = 0
    new_sums_list = [x]
    
    if x in sums:
        return False, None
        
    for i in range(n - 1, -1, -1):
        suffix_sum += seq[i]
        new_val = suffix_sum + x
        if new_val in sums:
            return False, None
        new_sums_list.append(new_val)
        
    return True, new_sums_list

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_seq = []
    
    def run_greedy_smallest():
        seq = []
        sums = set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(seq, sums, x)
            if ok:
                seq.append(x)
                sums.update(new_sums)
        return seq

    def run_greedy_largest():
        seq = []
        sums = set()
        last_val = 0
        while True:
            if time.time() > deadline:
                break
            found = None
            for x in range(256, last_val, -1):
                ok, new_sums = _try_append(seq, sums, x)
                if ok:
                    found = (x, new_sums)
                    break
            if not found:
                break
            x, new_sums = found
            seq.append(x)
            sums.update(new_sums)
            last_val = x
        return seq

    def run_random_greedy(seed_val):
        random.seed(seed_val)
        seq = []
        sums = set()
        last_val = 0
        
        while True:
            if time.time() > deadline:
                break
            valid_candidates = []
            x = last_val + 1
            # Find first 20 valid candidates to keep it fast and biased to small
            while x <= 256:
                if time.time() > deadline:
                    return seq
                ok, new_sums = _try_append(seq, sums, x)
                if ok:
                    valid_candidates.append((x, new_sums))
                    if len(valid_candidates) >= 20:
                        break
                x += 1
                
            if not valid_candidates:
                break
                
            chosen_idx = random.randint(0, len(valid_candidates) - 1)
            chosen_x, chosen_new_sums = valid_candidates[chosen_idx]
            
            seq.append(chosen_x)
            sums.update(chosen_new_sums)
            last_val = chosen_x
            
        return seq

    # 1. Deterministic Smallest
    seq1 = run_greedy_smallest()
    if len(seq1) > len(best_seq):
        best_seq = seq1
        
    # 2. Deterministic Largest
    seq2 = run_greedy_largest()
    if len(seq2) > len(best_seq):
        best_seq = seq2
        
    # 3. Randomized Restarts
    while time.time() < deadline - 1.0:
        seed_val = time.time_ns()
        seq_r = run_random_greedy(seed_val)
        if len(seq_r) > len(best_seq):
            best_seq = seq_r
            # If we find a very long one, we might want to restart the deadline?
            # No, just keep searching.
            
    return best_seq
