"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_seq = []
    best_len = 0
    
    # First, run the deterministic greedy to get a baseline
    def deterministic_greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    best_seq = deterministic_greedy()
    best_len = len(best_seq)
    
    # Now try randomized greedy many times
    rng = random.Random(42)
    
    for attempt in range(500000):
        if time.time() > deadline:
            break
        
        # Random ordering of candidates
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        values, sums = [], set()
        for x in candidates:
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        if len(values) > best_len:
            best_len = len(values)
            best_seq = values
    
    return best_seq
# EVOLVE-BLOCK-END
