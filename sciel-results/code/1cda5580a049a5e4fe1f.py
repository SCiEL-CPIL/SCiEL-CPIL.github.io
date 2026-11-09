import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    
    def evaluate(A):
        """
        Returns (max_mult, num_gaps) for the set A.
        """
        A_set = set(A)
        counts = [0] * (TARGET_END + 1)
        
        # Count sums
        for i, a in enumerate(A):
            for j, b in enumerate(A):
                if i <= j: # Unordered, a <= b
                    s = a + b
                    if TARGET_START <= s <= TARGET_END:
                        counts[s] += 1
        
        num_gaps = 0
        max_mult = 0
        for s in TARGETS:
            c = counts[s]
            if c == 0:
                num_gaps += 1
            elif c > max_mult:
                max_mult = c
                
        return max_mult, num_gaps

    def get_cost(A):
        max_mult, num_gaps = evaluate(A)
        # Heavy penalty for gaps
        return max_mult + 1000 * num_gaps

    # Initial candidates
    candidates = []
    
    # 1. Seed
    seed_A = list(range(64, 129))
    candidates.append(seed_A)
    
    # 2. Try to find smaller valid sets by random removal from seed
    # This might not be the best structure, but gives valid starts.
    for _ in range(20):
        A = seed_A[:]
        random.shuffle(A)
        # Try removing elements
        valid = True
        # We don't know if it's valid until we check.
        # Let's just generate random sets of size m
        pass

    # 3. Random sets
    for m in range(20, 50, 2):
        for _ in range(10):
            A = random.sample(range(N + 1), m)
            cost = get_cost(A)
            if cost < 1000: # No gaps
                candidates.append(A)
                if len(candidates) > 5:
                    break

    if not candidates:
        # Fallback to seed
        return seed_A

    # Sort candidates by cost
    candidates.sort(key=lambda x: get_cost(x))
    best_A = candidates[0]
    best_cost = get_cost(best_A)
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random candidate to mutate? Or just mutate the best?
        # Let's mutate the current best.
        A = best_A[:]
        
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
            # Optionally, add to candidates if significantly better
        else:
            # Accept with probability
            delta = new_cost - best_cost
            if delta < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / T)
            
            if accept:
                best_A = A[:]
                best_cost = new_cost
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
        # Occasionally reset to a random valid candidate to escape local minima?
        # Or just keep going.
        
    # Final check
    max_mult, num_gaps = evaluate(best_A)
    if num_gaps > 0:
        # If somehow invalid, return seed
        return seed_A
        
    return best_A
