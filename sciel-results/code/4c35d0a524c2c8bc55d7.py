import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = np.arange(TARGET_START, TARGET_END + 1)
    
    def evaluate(A):
        """
        Returns (max_mult, num_gaps) for the set A.
        Uses convolution for speed.
        """
        # Create indicator vector
        x = np.zeros(N + 1, dtype=np.int32)
        for a in A:
            x[a] = 1
        
        # Convolution x * x gives count of ordered pairs (a,b) with a+b=s
        c = np.convolve(x, x, mode='full')
        
        # c[s] is count of ordered pairs.
        # We need unordered pairs count for s in [128, 256].
        
        max_mult = 0
        num_gaps = 0
        
        for s in TARGETS:
            cs = c[s]
            if cs == 0:
                num_gaps += 1
                continue
            
            if s % 2 == 0:
                mid = s // 2
                # N_s = (c_s + x_mid) / 2
                n_s = (cs + x[mid]) // 2
            else:
                # N_s = c_s / 2
                n_s = cs // 2
            
            if n_s == 0:
                num_gaps += 1
            else:
                if n_s > max_mult:
                    max_mult = n_s
                    
        return max_mult, num_gaps

    def get_cost(A):
        max_mult, num_gaps = evaluate(A)
        # Heavy penalty for gaps to ensure validity
        return max_mult + 1000 * num_gaps

    # Initial candidates
    # A good starting point is often a dense set or a structured set.
    # Let's try a few random initializations and a greedy one.
    
    candidates = []
    
    # 1. Greedy construction
    # Start with empty set, add elements that cover most new targets with minimal peak increase?
    # Or start with a valid set and prune.
    # Let's try to build a valid set first.
    # A simple valid set: {0, 1, ..., 128} covers everything but has high multiplicity.
    # We want small A.
    
    # Let's try random sets of varying sizes and keep the best valid ones.
    random.seed(42)
    
    for m in range(20, 60, 5):
        for _ in range(50):
            A = random.sample(range(N + 1), m)
            cost = get_cost(A)
            if cost < 1000: # Valid
                candidates.append(A)
                
    # Also try some structured sets
    # Arithmetic progressions?
    for step in range(2, 10):
        A = list(range(0, N+1, step))
        cost = get_cost(A)
        if cost < 1000:
            candidates.append(A)
            
    if not candidates:
        # Fallback: Full set is always valid
        return list(range(N + 1))

    # Sort candidates by cost
    candidates.sort(key=lambda x: get_cost(x))
    best_A = candidates[0]
    best_cost = get_cost(best_A)
    
    # Local Search (Simulated Annealing or Hill Climbing with restarts)
    T = 10.0
    cooling = 0.999
    
    # We will run multiple SA restarts
    best_global_A = best_A[:]
    best_global_cost = best_cost
    
    while time.time() < deadline:
        # Reset to best global or random candidate?
        # Let's mutate the best global.
        A = best_global_A[:]
        T = 10.0
        
        # Run SA for a short duration
        sa_start = time.time()
        while time.time() < sa_start + 5.0 and time.time() < deadline:
            # Random Move
            move_type = random.choice(['swap', 'add', 'remove'])
            
            if move_type == 'swap' and len(A) > 1:
                idx = random.randint(0, len(A) - 1)
                old_val = A[idx]
                new_val = random.randint(0, N)
                if new_val == old_val or new_val in A:
                    continue
                A[idx] = new_val
            elif move_type == 'add' and len(A) < N + 1:
                new_val = random.randint(0, N)
                if new_val in A:
                    continue
                A.append(new_val)
            elif move_type == 'remove' and len(A) > 1:
                idx = random.randint(0, len(A) - 1)
                A.pop(idx)
            else:
                continue
                
            new_cost = get_cost(A)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = A[:]
                if best_cost < best_global_cost:
                    best_global_cost = best_cost
                    best_global_A = best_A[:]
            else:
                delta = new_cost - best_cost
                if delta < 0:
                    accept = True
                else:
                    # Accept with probability
                    if T > 0.1:
                        accept = random.random() < np.exp(-delta / T)
                    else:
                        accept = False
                
                if accept:
                    best_A = A[:]
                    best_cost = new_cost
            
            T *= cooling
            if T < 0.1:
                T = 0.1

    # Final check
    max_mult, num_gaps = evaluate(best_global_A)
    if num_gaps > 0:
        # If somehow invalid, return a known valid set (e.g. full set or first candidate)
        # The first candidate was valid.
        return candidates[0]
        
    return best_global_A
