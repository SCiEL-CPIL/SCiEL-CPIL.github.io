"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_MAX = 512
_N = 32
_DEADLINE = 110.0  # safety margin before 150s kill

def _score_bitmask(mask):
    """
    Count distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is a Python integer where bit i is 1 if i is in the set.
    """
    count = 0
    # Max d such that 2*d + 1 <= 512 is 255.
    # If a=1, a+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        # Check if there exists a such that bits a, a+d, a+2d are set
        # (mask & (mask << d) & (mask << (2*d))) will have bit a set if such a exists
        if (mask & (mask << d) & (mask << (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    rng = random.Random(42)
    
    # Initialize with a random set of 32 distinct integers in 1..512
    initial_elements = rng.sample(range(1, _MAX + 1), _N)
    current_mask = 0
    for x in initial_elements:
        current_mask |= (1 << x)
        
    current_score = _score_bitmask(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    
    # We need to efficiently pick elements to swap.
    # Maintain a list of current elements for fast removal.
    current_elements = set(initial_elements)
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate a random neighbor: swap one element in with one element out
        # Pick a random element to remove
        if not current_elements:
            break
        rem = rng.choice(list(current_elements))
        
        # Pick a random element to add
        # Since 32/512 is small, rejection sampling is fast
        while True:
            add = rng.randint(1, _MAX)
            if add not in current_elements:
                break
                
        # Apply move
        current_mask ^= (1 << rem)
        current_mask |= (1 << add)
        current_elements.remove(rem)
        current_elements.add(add)
        
        new_score = _score_bitmask(current_mask)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_mask = current_mask
        else:
            # Revert move
            current_mask ^= (1 << add)
            current_mask |= (1 << rem)
            current_elements.remove(add)
            current_elements.add(rem)
            
        T *= cooling
        
    # Convert best_mask back to list of integers
    best_elements = []
    for i in range(1, _MAX + 1):
        if (best_mask >> i) & 1:
            best_elements.append(i)
            
    return sorted(best_elements)
# EVOLVE-BLOCK-END
