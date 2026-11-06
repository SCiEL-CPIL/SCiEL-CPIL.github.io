"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _apply_comparator_packed(inputs, i, j):
    """Apply comparator (i, j) to packed inputs."""
    mask_i = np.uint64(1) << i
    mask_j = np.uint64(1) << j
    bit_i = (inputs & mask_i) != 0
    bit_j = (inputs & mask_j) != 0
    new_i = bit_i & bit_j
    new_j = bit_i | bit_j
    inputs = inputs & ~(mask_i | mask_j)
    inputs = np.where(new_i, inputs | mask_i, inputs)
    inputs = np.where(new_j, inputs | mask_j, inputs)
    return inputs

def _is_sorted_packed(inputs):
    """Check if all packed inputs are sorted (non-decreasing)."""
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        bit_k = (inputs & mask_k) != 0
        bit_k1 = (inputs & mask_k1) != 0
        if np.any(bit_k & ~bit_k1):
            return False
    return True

def _simulate_network_packed(comparators, inputs):
    """Simulate full network, return final packed state."""
    for i, j in comparators:
        inputs = _apply_comparator_packed(inputs, i, j)
    return inputs

def _verify_network_packed(comparators, inputs):
    """Verify network sorts all inputs."""
    final = _simulate_network_packed(comparators, inputs)
    return _is_sorted_packed(final)

def _generate_odd_even_transposition():
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _generate_bitonic_network():
    """Bitonic sorting network for N wires (power of 2 padded)."""
    size = 1
    while size < N:
        size *= 2
    comps = []
    
    def _bitonic_stage(start, length, ascending=True):
        if length == 1:
            return
        half = length // 2
        _bitonic_stage(start, half, ascending)
        _bitonic_stage(start + half, half, not ascending)
        for i in range(half):
            if ascending:
                comps.append((start + i, start + half + i))
            else:
                comps.append((start + half + i, start + i))
    
    for k in range(2, size + 1, 2):
        for start in range(0, size, k):
            _bitonic_stage(start, k, True)
    
    filtered = [(i, j) for i, j in comps if i < N and j < N]
    return filtered

def _remove_redundant_dag(comparators, inputs, time_budget):
    """
    Remove redundant comparators using a DAG-inspired topological pruning approach.
    We compute for each comparator the set of 'critical' inputs that depend on it.
    A comparator is redundant if no input's final sorted status depends on it.
    
    Strategy: 
    1. Compute prefix states.
    2. Compute suffix states (from end to start).
    3. For each comparator k, check if removing it changes the final state.
       If the final state remains sorted, it's redundant.
    """
    start_time = time.time()
    comps = list(comparators)
    changed = True
    
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        n_comps = len(comps)
        
        if n_comps == 0:
            break
            
        # Compute prefix states: state before each comparator
        prefix_states = [None] * (n_comps + 1)
        prefix_states[0] = inputs.copy()
        for k in range(n_comps):
            prefix_states[k + 1] = _apply_comparator_packed(prefix_states[k], *comps[k])
        
        # Compute suffix states: state after each comparator, assuming we start from prefix_states[k]
        # Actually, we want to know: if we skip comp k, what is the final state?
        # Let's compute "suffix_effect" from the end.
        # suffix_states[k] = state after applying comps[k:] to some input? No, that's not right.
        
        # Better approach:
        # For each k, we want to simulate:
        # 1. Apply comps[0..k-1] -> state_k
        # 2. Skip comp k
        # 3. Apply comps[k+1..end] -> final_state
        # 4. Check if final_state is sorted.
        
        # To speed this up, we can precompute the "tail" effect.
        # However, the tail effect depends on the input to the tail.
        # So we can't just precompute a single tail state.
        
        # We can iterate from the end.
        # For k from n_comps-1 down to 0:
        #   If we remove comp k, the input to the tail (comps[k+1..end]) is prefix_states[k].
        #   We need to simulate the tail on prefix_states[k].
        
        # Optimization: If we process from the end, we can reuse computations?
        # Not easily, because the input to the tail changes for each k.
        
        # Let's just try removing from the end first, as later comparators are more likely redundant.
        for k in range(n_comps - 1, -1, -1):
            if (time.time() - start_time) > time_budget:
                break
            
            # Simulate removal of comp k
            state = prefix_states[k].copy()
            for m in range(k + 1, n_comps):
                state = _apply_comparator_packed(state, *comps[m])
            
            if _is_sorted_packed(state):
                # Remove comp k
                comps = comps[:k] + comps[k+1:]
                changed = True
                break  # Restart because indices changed
    
    return comps

def construct():
    start_time = time.time()
    inputs = _make_all_inputs_packed()
    
    # Generate multiple candidate networks
    candidates = []
    
    # 1. Odd-even transposition
    oe = _generate_odd_even_transposition()
    candidates.append(oe)
    
    # 2. Bitonic network
    try:
        bitonic = _generate_bitonic_network()
        if len(bitonic) < len(oe):
            candidates.append(bitonic)
    except:
        pass
    
    # Verify and reduce each candidate
    best_comps = None
    best_count = float('inf')
    
    remaining_time = DEADLINE - (time.time() - start_time) - 5
    
    for idx, cand in enumerate(candidates):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        
        # Verify base network
        if not _verify_network_packed(cand, inputs):
            continue
        
        # Allocate time for reduction
        time_for_this = remaining_time / (len(candidates) - idx)
        
        reduced = _remove_redundant_dag(cand, inputs, time_for_this)
        
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_comps = reduced
    
    # If no valid reduced network found, fall back to base
    if best_comps is None:
        best_comps = oe
    
    # Final verification
    if not _verify_network_packed(best_comps, inputs):
        # Fallback to original
        best_comps = oe
    
    return best_comps

# EVOLVE-BLOCK-END
