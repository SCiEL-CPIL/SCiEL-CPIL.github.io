import time
import random

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

def _generate_offsets(min_dist):
    """Generate all bitmasks with popcount < min_dist."""
    offsets = []
    for mask in range(2**N):
        if mask.bit_count() < min_dist:
            offsets.append(mask)
    return offsets

def _bitset_greedy(deadline, seed=0):
    random.seed(seed)
    # Precompute offsets
    offsets = _generate_offsets(MIN_DISTANCE)
    
    # available is a bitmask of all 2^N words
    # We represent the set of available words as a Python integer
    # Bit i is 1 if word i is available
    available = (1 << (2**N)) - 1
    
    selected = []
    
    start_time = time.time()
    
    while available:
        if time.time() > deadline:
            break
        
        # Pick a word
        # Strategy 1: Lowest index
        # lowbit = available & -available
        # w = lowbit.bit_length() - 1
        
        # Strategy 2: Random index from available
        # We need to pick a random set bit from available.
        # This is tricky with big ints.
        # We can convert to a list of indices? No, too slow.
        # We can use a random mask?
        # Let's stick to Lowest Index for determinism and speed.
        # To break symmetry, we can rotate the bits?
        # Or just use different seeds for the offsets order?
        # Actually, Lowest Index is a specific deterministic path.
        # Let's try to pick the word that is "most isolated" in the remaining graph?
        # That's expensive.
        
        # Let's try Lowest Index first.
        lowbit = available & -available
        w = lowbit.bit_length() - 1
        
        selected.append(w)
        
        # Remove w and its neighbors from available
        # conflict_mask = 0
        # for offset in offsets:
        #     neighbor = w ^ offset
        #     conflict_mask |= (1 << neighbor)
        
        # Optimization: 
        # Instead of building a huge int, we can just iterate and clear bits?
        # No, we need to clear 120k bits.
        # Building the mask is O(120k).
        
        conflict_mask = 0
        for offset in offsets:
            neighbor = w ^ offset
            conflict_mask |= (1 << neighbor)
            
        available &= ~conflict_mask
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Run a few variations
    # 1. Standard Lowest Index
    code1 = _bitset_greedy(deadline)
    best_code = code1
    best_len = len(code1)
    
    # 2. Try to improve by shuffling offsets?
    # If we shuffle offsets, the conflict mask construction order changes, 
    # but the mask is the same. So it doesn't help.
    
    # 3. Try starting with a different word?
    # If we clear bit 0, we force it to pick the next one.
    # We can try picking the word with max weight?
    # Let's implement a version that picks the highest index.
    
    def _bitset_greedy_high(deadline):
        offsets = _generate_offsets(MIN_DISTANCE)
        available = (1 << (2**N)) - 1
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            
            # Pick highest index
            # highbit = 1 << (available.bit_length() - 1)
            # w = available.bit_length() - 1
            w = available.bit_length() - 1
            
            selected.append(w)
            
            conflict_mask = 0
            for offset in offsets:
                neighbor = w ^ offset
                conflict_mask |= (1 << neighbor)
                
            available &= ~conflict_mask
            
        return selected

    if time.time() < deadline - 5:
        code2 = _bitset_greedy_high(deadline)
        if len(code2) > best_len:
            best_code = code2
            best_len = len(code2)
            
    # 4. Randomized picking
    # To pick a random word from available:
    # We can sample a random position and check if it's set?
    # Or use a random mask and find intersection?
    # Let's try a simple random restart with a random initial offset in the "rotation" of the cube?
    # Actually, the problem is symmetric.
    
    return best_code
