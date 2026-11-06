"""
Size-efficient sorting networks on 17 wires.
Strategy: Use Python's arbitrary-precision integers as bitmasks representing all 2^17
Boolean inputs simultaneously. This leverages the 0-1 principle and allows extremely
fast verification (single big-int operations) enabling deep greedy deletion and local search.
"""
import time
import random

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _init_state_masks():
    """
    Initialize state masks for each wire.
    state_mask[k] is an integer where bit x is 1 iff input x has a 1 on wire k.
    Input x is an integer 0..2^N-1. Bit k of x corresponds to wire k.
    So state_mask[k] has bits set at positions x where (x & (1<<k)) != 0.
    This is a pattern of blocks of 2^k zeros and 2^k ones.
    """
    masks = [0] * N
    for k in range(N):
        # Pattern: 0...0 1...1 0...0 1...1 ...
        # Block size is 2^k.
        block = 1 << k
        # We can construct this efficiently.
        # The number is formed by repeating (2^block - 1) << (2*block * i)
        # Actually, it's easier to just loop or use a trick.
        # Since N is small, we can construct it.
        m = 0
        for i in range(0, NUM_INPUTS, 2 * block):
            # Set bits from i + block to i + 2*block - 1
            # This is a mask of 'block' bits, shifted by i+block
            m |= ((1 << block) - 1) << (i + block)
        masks[k] = m
    return masks

def _apply_comparator(masks, i, j):
    """
    Apply comparator (i, j) to the state masks.
    new_i = masks[i] & masks[j]
    new_j = masks[i] | masks[j]
    """
    masks[i] &= masks[j]
    masks[j] = masks[i] | masks[j] # Wait, masks[i] is already updated?
    # No, we need original values.
    # new_i = old_i & old_j
    # new_j = old_i | old_j
    # If I do masks[i] &= masks[j], masks[i] becomes new_i.
    # Then masks[j] = masks[i] | masks[j] would be new_i | old_j.
    # (old_i & old_j) | old_j = old_j. This is wrong. new_j should be old_i | old_j.
    # So we must compute both from originals.
    
    # Let's redo this carefully in the caller or use temp vars.
    # In the context of this function, 'masks' is a list.
    # We should return new values or update in place correctly.
    pass # Placeholder, will implement correctly in construct or helper

def _apply_comparator_correct(masks, i, j):
    """
    Correctly apply comparator (i, j).
    """
    old_i = masks[i]
    old_j = masks[j]
    masks[i] = old_i & old_j
    masks[j] = old_i | old_j

def _is_sorted(masks):
    """
    Check if the final state is sorted for all inputs.
    Sorted means: for all k, if wire k is 1, wire k+1 must be 1.
    Equivalently: masks[k] & ~masks[k+1] must be 0 for all k.
    Or: masks[k] | masks[k+1] == masks[k+1] (masks[k] is subset of masks[k+1])
    """
    for k in range(N - 1):
        if (masks[k] & ~masks[k+1]) != 0:
            return False
    return True

def _simulate_network(comparators):
    """
    Simulate the network on all inputs using bitmask masks.
    Returns the final list of masks.
    """
    masks = _init_state_masks()
    for i, j in comparators:
        _apply_comparator_correct(masks, i, j)
    return masks

def _odd_even_transposition():
    """
    Generate the odd-even transposition sorting network.
    """
    comps = []
    for phase in range(N):
        parity = phase % 2
        for i in range(parity, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    start_time = time.time()
    
    # 1. Initialize with Odd-Even Transposition Network
    comps = _odd_even_transposition()
    
    # 2. Greedy Deletion
    # We try to remove comparators one by one.
    # To be efficient, we maintain the state masks?
    # No, because removing a comparator changes the state for all subsequent comparators.
    # So we must re-simulate for each candidate removal.
    # However, simulation is very fast with big ints.
    
    # We can optimize by checking removals in a specific order or random order.
    # Let's do a pass-based approach.
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 2:
        improved = False
        # Try removing from the end first? Or random?
        # Removing from the end is often easier as it doesn't affect previous steps' validity as much?
        # Actually, any removal requires full re-verification.
        # Let's scan from 0 to len(comps)-1.
        
        # To make it faster, we can shuffle the indices we check?
        # Or just linear scan.
        
        for idx in range(len(comps) - 1, -1, -1):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            
            # Try removing comps[idx]
            new_comps = comps[:idx] + comps[idx+1:]
            
            # Verify
            final_masks = _simulate_network(new_comps)
            if _is_sorted(final_masks):
                comps = new_comps
                improved = True
                # Restart scan from end? Or continue?
                # If we continue, we might remove more.
                # But indices have shifted.
                # It's safer to break and restart the while loop.
                break
    
    # 3. Local Search / Random Perturbation
    # Try swapping adjacent comparators?
    # Try removing and re-inserting?
    # Given time constraints, greedy deletion is the main driver.
    # Let's add a simple random restart or perturbation if time permits.
    
    # For now, just return the result of greedy deletion.
    # The bitmask simulation is so fast that we might be able to do more passes.
    
    # Let's do multiple passes of greedy deletion to ensure convergence.
    # The while loop above already does this until no improvement.
    
    return comps

# EVOLVE-BLOCK-END
