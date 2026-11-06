import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 30
    # Initial state: all zeros
    current = [0] * n
    
    # Calculate initial sum (which is 0, but we need nonzero)
    # Let's start with a random valid state or a simple one
    # A simple nonzero state: [1, 0, ..., 0] -> sum = 1/1 = 1
    # Or [0, 1, 0, ..., 0] -> sum = 1/2
    # Let's start with a random state that is nonzero
    while True:
        candidate = [random.choice([-1, 0, 1]) for _ in range(n)]
        s = sum(Fraction(candidate[i], i+1) for i in range(n) if candidate[i] != 0)
        if s != 0:
            current = candidate
            current_val = s
            current_abs = abs(s)
            break
            
    best = current
    best_val = current_val
    best_abs = current_abs
    
    while time.time() < deadline:
        # Generate a neighbor
        idx = random.randrange(n)
        # Change value at idx to a different value in {-1, 0, 1}
        old_val = current[idx]
        new_val = random.choice([-1, 0, 1])
        while new_val == old_val:
            new_val = random.choice([-1, 0, 1])
            
        current[idx] = new_val
        
        # Calculate new sum
        # Optimization: delta = (new_val - old_val) / (idx+1)
        delta = Fraction(new_val - old_val, idx + 1)
        new_sum = current_val + delta
        
        if new_sum == 0:
            # Revert and continue, or try another move
            current[idx] = old_val
            continue
            
        new_abs = abs(new_sum)
        
        # Accept if better or with some probability (simulated annealing-like, but simple hill climb is often enough for this small space)
        if new_abs < current_abs:
            current_val = new_sum
            current_abs = new_abs
            
            if current_abs < best_abs:
                best_abs = current_abs
                best_val = current_val
                best = current[:]
        else:
            # Revert
            current[idx] = old_val
            # Occasionally restart to escape local minima
            if random.random() < 0.01:
                while True:
                    candidate = [random.choice([-1, 0, 1]) for _ in range(n)]
                    s = sum(Fraction(candidate[i], i+1) for i in range(n) if candidate[i] != 0)
                    if s != 0:
                        current = candidate
                        current_val = s
                        current_abs = abs(s)
                        break

    return best
# EVOLVE-BLOCK-END
