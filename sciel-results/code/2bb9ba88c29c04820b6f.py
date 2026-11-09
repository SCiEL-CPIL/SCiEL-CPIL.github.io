"""Construct binary error-correcting code of length 18, distance 6 via Simulated Annealing with Reference Counting."""
import random
import time
import math

N = 18
MIN_DIST = 6
SIZE = 1 << N

def popcount(x):
    return bin(x).count('1')

def get_neighbors(w, max_dist):
    """Generate all words within max_dist from w."""
    # We need neighbors at distance 1 to max_dist-1? 
    # If we add w, we block w and all neighbors at dist 1..MIN_DIST-1.
    # So we block dist 0 to MIN_DIST-1.
    # Here max_dist is MIN_DIST - 1 = 5.
    neighbors = [w]
    # Generate all subsets of bits to flip
    # This is O(sum C(N, i))
    # For N=18, max_dist=5: 1 + 18 + 153 + 816 + 3060 + 8568 = 12616
    # This is fast enough if called ~1000 times.
    
    # Efficient generation:
    # Iterate through all masks with popcount <= max_dist
    # But generating them on the fly for each w is slow.
    # Pre-compute all masks with popcount <= 5?
    # There are 12616 such masks.
    # We can pre-compute the list of masks once.
    pass

# Pre-compute masks
MASKS = []
for i in range(SIZE):
    if popcount(i) <= MIN_DIST - 1:
        MASKS.append(i)

def add_word(code, blocked_count, w):
    """Add word w to code, update blocked_count. Returns True if successful."""
    if blocked_count[w] > 0:
        return False
    code.append(w)
    for mask in MASKS:
        neighbor = w ^ mask
        blocked_count[neighbor] += 1
    return True

def remove_word(code, blocked_count, w):
    """Remove word w from code, update blocked_count."""
    if w not in code:
        return
    code.remove(w) # O(N)
    for mask in MASKS:
        neighbor = w ^ mask
        blocked_count[neighbor] -= 1

def construct():
    """Construct the largest possible binary code of length 18, distance 6."""
    best_code = []
    start_time = time.time()
    time_limit = 55.0 # Leave 5s margin
    
    # Simulated Annealing
    # State: code (list), blocked_count (list of ints)
    
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        code = []
        blocked_count = [0] * SIZE
        
        # Initialize with 0
        add_word(code, blocked_count, 0)
        
        current_len = len(code)
        best_local_len = current_len
        best_local_code = list(code)
        
        # Temperature
        T = 1.0
        cooling = 0.999
        
        # Iterations
        # Each iteration involves a random move
        # We want to try many moves
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and time.time() - start_time < time_limit:
            steps += 1
            
            # Choose move type
            # 1. Add random word
            # 2. Remove random word
            # 3. Swap
            
            r = random.random()
            
            if r < 0.6:
                # Try to add a random word
                w = random.randint(0, SIZE - 1)
                if blocked_count[w] == 0:
                    add_word(code, blocked_count, w)
                    if len(code) > best_local_len:
                        best_local_len = len(code)
                        best_local_code = list(code)
                        # Restart cooling?
                        T = 1.0
                else:
                    # Failed add, cool down slightly?
                    pass
                    
            elif r < 0.8:
                # Try to remove a random word
                if code:
                    w = random.choice(code)
                    remove_word(code, blocked_count, w)
                    # Accept removal? 
                    # In SA for maximization, we usually accept worse states with prob exp(-dE/T).
                    # dE = 1 (loss of 1 codeword).
                    if random.random() < math.exp(-1.0 / T):
                        pass # Keep removal
                    else:
                        add_word(code, blocked_count, w) # Revert
                        
            else:
                # Swap
                if code:
                    w1 = random.choice(code)
                    w2 = random.randint(0, SIZE - 1)
                    # Temporarily remove w1
                    remove_word(code, blocked_count, w1)
                    if blocked_count[w2] == 0:
                        add_word(code, blocked_count, w2)
                        # Swap successful, size unchanged
                        # But structure changed
                        if random.random() < 0.5:
                            pass # Keep
                        else:
                            # Revert
                            remove_word(code, blocked_count, w2)
                            add_word(code, blocked_count, w1)
                    else:
                        # Cannot add w2, revert
                        add_word(code, blocked_count, w1)
                        
            T *= cooling
            if T < 0.01:
                T = 0.01

        if best_local_len > len(best_code):
            best_code = best_local_code

    # Final check
    if not best_code:
        best_code = [0]
        
    return best_code
