import numpy as np
import time

N = 18
TOTAL_INPUTS = 1 << N  # 262144

def _make_initial_network():
    """Bubble sort network for N wires."""
    comps = []
    for i in range(N - 1):
        for j in range(N - 1 - i):
            comps.append((j, j + 1))
    return comps

def _build_bitmasks(comps):
    """Precompute bitmasks for each comparator's min and max operations."""
    min_masks = []
    max_masks = []
    for i, j in comps:
        # min: wire i gets 1 only if both i and j are 1
        min_mask = (1 << i) & (1 << j)
        # max: wire j gets 1 if either i or j is 1
        max_mask = (1 << i) | (1 << j)
        min_masks.append(min_mask)
        max_masks.append(max_mask)
    return np.array(min_masks, dtype=np.uint32), np.array(max_masks, dtype=np.uint32)

def _verify(comps):
    """Verify sorting network over all 2^N Boolean inputs using bit-parallel ops.
    Returns True if all inputs are sorted correctly."""
    if len(comps) == 0:
        return False
    
    min_masks, max_masks = _build_bitmasks(comps)
    
    # All possible inputs as uint32 array (each input is a bitmask of set wires)
    inputs = np.arange(1 << N, dtype=np.uint32)
    
    for k in range(len(comps)):
        i, j = comps[k]
        # For each input, compute min and max of wires i and j
        # min: both bits set -> 1, else 0
        # max: either bit set -> 1, else 0
        # Using precomputed masks:
        # bit_i = (inputs >> i) & 1
        # bit_j = (inputs >> j) & 1
        # min_bit = bit_i & bit_j
        # max_bit = bit_i | bit_j
        # new_input = inputs with wire i set to min_bit and wire j set to max_bit
        
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        min_bit = bit_i & bit_j
        max_bit = bit_i | bit_j
        
        # Clear bits i and j, then set them
        clear_mask = ~((1 << i) | (1 << j))
        inputs = (inputs & clear_mask) | (min_bit << i) | (max_bit << j)
    
    # Check if all inputs are sorted: for each input, if bit k is 0, all higher bits must be 0
    # i.e., the set bits must be a contiguous prefix from bit 0
    # A number is "sorted" (non-increasing from high to low wire) iff it equals 2^k - 1 for some k
    # Actually, sorted means: if wire j has 1 and j > i, then wire i must also have 1
    # Equivalently: the set of 1-bits forms {0, 1, ..., k-1} for some k
    # Check: for each input x, x should satisfy: for all i < j, if bit_j=1 then bit_i=1
    # Fast check: x & (x+1) == 0 is NOT the right check.
    # The right check: x is of the form 2^k - 1, i.e., x == (1 << popcount(x)) - 1
    # Alternative: x & (x >> 1) ... no.
    # Simplest: for each x, check that if bit j is set, all bits below j are set.
    # This is equivalent to: x == (1 << (x.bit_count())) - 1
    # In numpy: popcount is not directly available in older numpy, but we can use lookup or bit tricks.
    
    # Efficient check: x is sorted iff x & (x + 1) == 0 is WRONG.
    # Let's think: sorted means bits are 111...1000...0 (ones at bottom).
    # So x should be of the form (1 << k) - 1.
    # Check: (x + 1) & x == 0 is true for x = 0, 1, 3, 7, 15, ... i.e., x = 2^k - 1.
    # Wait: (x+1) & x == 0 means x+1 is a power of 2, so x = 2^k - 1. Yes!
    
    # But we need: for each input, the OUTPUT must be of the form 2^k - 1.
    # So check: (inputs + 1) & inputs == 0 for all inputs.
    
    return bool(np.all(((inputs + 1) & inputs) == 0))

def _verify_fast(comps):
    """Faster verification using vectorized numpy operations."""
    if len(comps) == 0:
        return False
    
    inputs = np.arange(1 << N, dtype=np.uint32)
    
    for i, j in comps:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        min_bit = bit_i & bit_j
        max_bit = bit_i | bit_j
        clear_mask = ~((1 << i) | (1 << j))
        inputs = (inputs & clear_mask) | (min_bit << i) | (max_bit << j)
    
    return bool(np.all(((inputs + 1) & inputs) == 0))

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Start with bubble sort network
    comps = _make_initial_network()
    
    # Verify initial network
    if not _verify_fast(comps):
        # Should not happen, but return as-is
        return comps
    
    best_comps = list(comps)
    best_len = len(comps)
    
    # Local search: try removing each comparator and check validity
    # Use a more efficient approach: try removals in random order
    import random
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try to remove a random comparator
        removed_any = False
        # Try all positions, but in random order for diversity
        indices = list(range(len(comps)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            new_comps = comps[:idx] + comps[idx+1:]
            
            if _verify_fast(new_comps):
                comps = new_comps
                best_comps = list(comps)
                best_len = len(comps)
                removed_any = True
                # Restart the shuffle since the list changed
                break
        
        if not removed_any:
            # Try adding and removing (swap) moves
            # Try swapping two comparators
            if len(comps) >= 2:
                idx1 = rng.randint(0, len(comps) - 1)
                idx2 = rng.randint(0, len(comps) - 1)
                if idx1 != idx2:
                    new_comps = list(comps)
                    new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                    if _verify_fast(new_comps):
                        # Swap doesn't change length, but might enable future removals
                        # Only accept if it helps (we'll check removals after)
                        pass
            
            # Try replacing one comparator with another
            # This is more complex; skip for now
            # If we can't remove anything, we're likely at a local minimum
            # Try a different strategy: start from a different initial network
            
            # Try odd-even transposition as alternative seed
            if best_len >= 150:
                # Try a different starting point
                alt_comps = []
                for phase in range(N):
                    for i in range(phase % 2, N - 1, 2):
                        alt_comps.append((i, i + 1))
                
                if _verify_fast(alt_comps):
                    comps = list(alt_comps)
                    # Now try to remove from this
                    indices = list(range(len(comps)))
                    rng.shuffle(indices)
                    for idx in indices:
                        if time.time() > deadline:
                            break
                        new_comps = comps[:idx] + comps[idx+1:]
                        if _verify_fast(new_comps):
                            comps = new_comps
                            best_comps = list(comps)
                            best_len = len(comps)
                            break
            else:
                # We're at a good local minimum, do more random restarts
                # Try random removals from current best
                if len(comps) > 68:  # Don't go below theoretical lower bound
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        idx = rng.randint(0, len(comps) - 1)
                        new_comps = comps[:idx] + comps[idx+1:]
                        if _verify_fast(new_comps):
                            comps = new_comps
                            best_comps = list(comps)
                            best_len = len(comps)
    
    return best_comps
# EVOLVE-BLOCK-END
