import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N

def get_initial_state():
    indices = np.arange(NUM_INPUTS)
    state = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        state[:, i] = (indices >> i) & 1
    return state

INITIAL_STATE = get_initial_state()

def verify_network(comparators):
    state = INITIAL_STATE.copy()
    
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
        
    mask = (state[:, :-1] == 1) & (state[:, 1:] == 0)
    return not np.any(mask)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Seed: Odd-Even Transposition
    comparators = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Verify seed
    if not verify_network(comparators):
        return comparators

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate backwards
        for idx in range(len(comparators) - 1, -1, -1):
            if time.time() > deadline:
                return comparators
                
            candidate = comparators[:idx] + comparators[idx+1:]
            
            if verify_network(candidate):
                comparators = candidate
                changed = True
                break
                
    return comparators
