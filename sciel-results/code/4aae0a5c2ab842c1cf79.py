import time
import random

def compute_max_abs_sum(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n):
        for r in range(min(d, n)):
            seq = [signs[pos] for pos in range(r, n, d)]
            m = len(seq)
            if m == 0:
                continue
            max_ending = seq[0]
            max_so_far = seq[0]
            min_ending = seq[0]
            min_so_far = seq[0]
            for i in range(1, m):
                v = seq[i]
                if max_ending < 0:
                    max_ending = v
                else:
                    max_ending += v
                if max_ending > max_so_far:
                    max_so_far = max_ending
                if min_ending > 0:
                    min_ending = v
                else:
                    min_ending += v
                if min_ending < min_so_far:
                    min_so_far = min_ending
            candidate = max(abs(max_so_far), abs(min_so_far))
            if candidate > max_abs:
                max_abs = candidate
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    n = 256
    
    # Thue-Morse initialization
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    best_signs = signs[:]
    best_val = compute_max_abs_sum(best_signs)
    
    rng = random.Random(42)
    
    # Iterative improvement with restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        if restart == 0:
            signs = best_signs[:]
        else:
            signs = best_signs[:]
            num_flips = rng.randint(3, 15)
            for _ in range(num_flips):
                signs[rng.randint(0, n-1)] *= -1
        
        # Hill climbing
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                signs[i] *= -1
                val = compute_max_abs_sum(signs)
                if val < best_val:
                    best_val = val
                    best_signs = signs[:]
                    improved = True
                else:
                    signs[i] *= -1
            if not improved:
                break
    
    return best_signs
