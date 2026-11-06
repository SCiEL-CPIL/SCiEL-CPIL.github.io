import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    best_seq = []
    
    candidates = list(range(1, 257))
    
    # First, run the deterministic greedy to get a baseline
    seq = []
    sums = set()
    for x in candidates:
        new_sums = []
        total = x
        for a in reversed(seq):
            total += a
            new_sums.append(total)
        new_sums.append(x)
        if not sums.intersection(new_sums):
            seq.append(x)
            sums.update(new_sums)
    best_seq = seq
    
    # Then, run randomized greedy
    while time.time() < deadline:
        random.shuffle(candidates)
        seq = []
        sums = set()
        
        for x in candidates:
            new_sums = []
            total = x
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            new_sums.append(x)
            
            if not sums.intersection(new_sums):
                seq.append(x)
                sums.update(new_sums)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            if len(best_seq) == 256:
                break
            
    return best_seq
