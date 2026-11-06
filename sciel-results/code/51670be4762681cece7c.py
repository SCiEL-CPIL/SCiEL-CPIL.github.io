import random
import time

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N
MASKS = []
CANDIDATES = list(range(MAX_WORD))

def _init_masks():
    global MASKS
    for m in range(MAX_WORD):
        if m.bit_count() < MIN_DIST:
            MASKS.append(m)

def construct():
    _init_masks()
    best_code = []
    best_size = 0
    
    # Use a fixed seed for reproducibility in local tests, but random in eval?
    # random.seed() is better for exploration.
    
    start_time = time.time()
    # Limit iterations to prevent infinite loops if time check fails
    max_iterations = 1000 
    
    for i in range(max_iterations):
        if time.time() - start_time > 55:
            break
            
        # Shuffle candidates
        random.shuffle(CANDIDATES)
        
        # Initialize availability
        # Using bytearray for speed
        is_avail = bytearray([1] * MAX_WORD)
        code = []
        
        for idx in CANDIDATES:
            if is_avail[idx]:
                code.append(idx)
                # Mark neighbors as unavailable
                # Unroll loop? No, Python loop is fine for 9k iterations * 256 times
                for m in MASKS:
                    v = idx ^ m
                    is_avail[v] = 0
                    
        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # Early exit if we hit a very high number? 
            # The upper bound is 340. 
            # If we get > 300, that's exceptional.
            # But 258 is the known LB.
            pass

    return best_code
