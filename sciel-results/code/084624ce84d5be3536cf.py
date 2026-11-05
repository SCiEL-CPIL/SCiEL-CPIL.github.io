"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array (each element is a packed N-bit value)."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _apply_comparator_packed(inputs, i, j):
    """Apply comparator (i, j) to packed inputs. min to i, max to j."""
    mask_i = np.uint64(1) << i
    mask_j = np.uint64(1) << j
    bit_i = (inputs & mask_i) != 0
    bit_j = (inputs & mask_j) != 0
    
    # min goes to i: both must be 1
    # max goes to j: at least one must be 1
    new_i = bit_i & bit_j
    new_j = bit_i | bit_j
    
    # Clear both bits, set new values
    inputs = inputs & ~(mask_i | mask_j)
    inputs = np.where(new_i, inputs | mask_i, inputs)
    inputs = np.where(new_j, inputs | mask_j, inputs)
    return inputs

def _is_sorting_network_packed(comparators, inputs):
    """Check if network sorts all inputs using zero-one principle, with packed representation."""
    for i, j in comparators:
        inputs = _apply_comparator_packed(inputs, i, j)
    
    # Check if each input is sorted: for all k < N-1, if bit k is 1 then bit k+1 must be 1
    # i.e., no pattern where bit k=1 and bit k+1=0
    # For each adjacent pair (k, k+1), check: not (bit_k AND NOT bit_{k+1})
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        bit_k = (inputs & mask_k) != 0
        bit_k1 = (inputs & mask_k1) != 0
        # Violation: bit_k is 1 and bit_k1 is 0
        if np.any(bit_k & ~bit_k1):
            return False
    return True

def _try_remove_packed(comparators, idx, inputs):
    """Try removing comparator at idx, return True if still valid."""
    new_comps = comparators[:idx] + comparators[idx+1:]
    return _is_sorting_network_packed(new_comps, inputs)

def construct():
    start_time = time.time()
    
    # Base: odd-even transposition
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    inputs = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, inputs):
        return comps
    
    # Iteratively remove redundant comparators
    # Strategy: try removing from the end first (later comparators are more likely redundant),
    # then scan from beginning. Use multiple passes.
    
    changed = True
    passes = 0
    while changed and (time.time() - start_time) < DEADLINE - 3:
        changed = False
        passes += 1
        
        # Alternate between forward and backward scan to find different redundancies
        if passes % 2 == 1:
            indices = range(len(comps) - 1, -1, -1)  # backward
        else:
            indices = range(len(comps))  # forward
        
        for idx in indices:
            if (time.time() - start_time) > DEADLINE - 3:
                break
            if _try_remove_packed(comps, idx, inputs):
                comps = comps[:idx] + comps[idx+1:]
                changed = True
                break  # Restart loop since indices changed
    
    return comps

# EVOLVE-BLOCK-END
