import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Initial odd-even transposition network: 120 comparators
    initial_net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    # Target: sorted 0s then 1s.
    # If k ones, target is ((1 << k) - 1) << (N - k)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)
    # Note: if k=0, (1<<0)-1 = 0. 0 << 16 = 0. Correct.
    # if k=16, (1<<16)-1 = 65535. 65535 << 0 = 65535. Correct.

    def verify(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            # Condition: bit i is 1, bit j is 0
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    # Verify initial
    if not verify(initial_net):
        # Fallback to seed if something went wrong (should not happen)
        return initial_net

    current_net = initial_net
    best_net = initial_net
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    random.seed(42) # Reproducibility
    
    while time.time() - start_time < deadline:
        # Try to improve
        # Strategy: Random removal
        if len(current_net) > 1:
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
        
        # Occasionally try swapping to escape local minima
        # Swapping doesn't reduce size, but might enable future removals
        if len(current_net) > 1 and random.random() < 0.2:
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 != i2:
                candidate = current_net.copy()
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                # Swapping a valid network usually keeps it valid?
                # Not necessarily, but checking is cheap enough.
                if verify(candidate):
                    current_net = candidate
                    # Update best only if we find a shorter one later
                    # For now, just move to this state
                    # We don't update best_net here unless len < best_len (which is false)
        
    return best_net
# EVOLVE-BLOCK-END
