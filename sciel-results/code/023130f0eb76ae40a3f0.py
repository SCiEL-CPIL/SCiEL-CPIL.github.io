import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def greedy_deterministic():
        seq = []
        sums_set = set()
        while True:
            last = seq[-1] if seq else 0
            found = False
            for x in range(last + 1, 257):
                total = x
                if total in sums_set:
                    continue
                valid = True
                for a in reversed(seq):
                    total += a
                    if total in sums_set:
                        valid = False
                        break
                if valid:
                    seq.append(x)
                    sums_set.add(total) # Wait, total is the last one. Need to add all new sums.
                    # Re-calculate new sums for update
                    total = x
                    sums_set.add(total)
                    for a in reversed(seq[:-1]):
                        total += a
                        sums_set.add(total)
                    found = True
                    break
            if not found:
                break
        return seq

    def greedy_randomized():
        seq = []
        sums_set = set()
        while True:
            last = seq[-1] if seq else 0
            valid_candidates = []
            
            for x in range(last + 1, 257):
                total = x
                if total in sums_set:
                    continue
                valid = True
                for a in reversed(seq):
                    total += a
                    if total in sums_set:
                        valid = False
                        break
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            next_val = random.choice(valid_candidates)
            
            seq.append(next_val)
            
            total = next_val
            sums_set.add(total)
            for a in reversed(seq[:-1]):
                total += a
                sums_set.add(total)
                
        return seq

    best_seq = greedy_deterministic()
    
    while time.time() < deadline:
        seq = greedy_randomized()
        if len(seq) > len(best_seq):
            best_seq = seq
            
    return best_seq
