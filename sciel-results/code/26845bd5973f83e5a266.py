import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    # Leave 5s buffer for safety against 90s limit
    deadline = start_time + 85.0
    
    def is_time_up():
        return time.time() > deadline

    def greedy_randomized(rng, candidate_window=10):
        """
        Greedy algorithm with randomization.
        At each step, considers the first 'candidate_window' valid candidates
        and picks one randomly. This helps escape local optima.
        """
        values = []
        sums = set()
        suffix_sums = []
        last = 0
        
        while True:
            if is_time_up():
                break
                
            candidates = []
            # We only need to check up to a certain limit? No, we must check all to find valid ones.
            # But we can stop early if we have enough candidates for the random pick?
            # If we want to pick from the first 'candidate_window' valid ones, we must find them.
            
            found = 0
            for x in range(last + 1, 257):
                new_sums = [x]
                for s in suffix_sums:
                    new_sums.append(x + s)
                
                if sums.intersection(new_sums):
                    continue
                
                candidates.append((x, new_sums))
                found += 1
                if found >= candidate_window:
                    break
            
            if not candidates:
                break
            
            # Pick a candidate
            # Bias towards smaller numbers? Or uniform?
            # Uniform among the first 'candidate_window' valid ones.
            idx = rng.randint(0, len(candidates) - 1)
            x, new_sums = candidates[idx]
            
            values.append(x)
            sums.update(new_sums)
            suffix_sums = new_sums
            last = x
            
        return values

    best_seq = []
    best_len = 0
    
    # Run deterministic greedy first
    # This is equivalent to candidate_window=1, picking index 0
    rng_det = random.Random(42)
    seq_det = greedy_randomized(rng_det, candidate_window=1)
    if len(seq_det) > best_len:
        best_len = len(seq_det)
        best_seq = seq_det
        
    # Run randomized greedy
    # Use a fixed seed for reproducibility in evaluation if needed, or just system time?
    # The task doesn't specify reproducibility, but deterministic is safer for debugging.
    # However, randomization is the point. Let's use a master seed.
    master_seed = 12345
    
    # Number of iterations depends on time.
    # Each greedy run is O(N^2) roughly, where N is length.
    # N is small (< 20), so it's very fast. We can do thousands.
    
    max_iters = 5000
    for i in range(max_iters):
        if is_time_up():
            break
            
        rng = random.Random(master_seed + i)
        # Vary the window size to explore different "aggressiveness"
        # Small window = close to greedy. Large window = more random.
        window = rng.randint(1, 20)
        
        seq = greedy_randomized(rng, window)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
    return best_seq
