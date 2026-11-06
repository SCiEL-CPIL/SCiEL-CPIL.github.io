"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
import time
import random

N = 17

# Pre-compute initial bitmasks
# S[i] is an integer where bit k is 1 iff input k has 1 at position i.
# Input k is the integer k. Bit i of k is (k >> i) & 1.
# So bit k of S[i] is (k >> i) & 1.
# This means S[i] is the number formed by taking the i-th bit of every k from 0 to 2^N-1.

def _get_initial_bitmasks(n):
    masks = [0] * n
    limit = 1 << n
    for i in range(n):
        m = 0
        # We need bit k of m to be 1 if (k >> i) & 1 == 1
        # This is a periodic pattern.
        for k in range(limit):
            if (k >> i) & 1:
                m |= (1 << k)
        masks[i] = m
    return masks

_INITIAL_MASKS = _get_initial_bitmasks(N)

def _is_valid_bitmask(net):
    """
    Checks if the network sorts all 0-1 inputs using bitmask simulation.
    Returns True if valid.
    """
    # Copy the initial masks
    state = _INITIAL_MASKS[:]
    
    for i, j in net:
        # min to i, max to j
        # min = state[i] & state[j]
        # max = state[i] | state[j]
        mi = state[i] & state[j]
        mj = state[i] | state[j]
        state[i] = mi
        state[j] = mj
        
    # Check if sorted
    # For all adjacent wires k, k+1, we must have state[k] subset of state[k+1]
    # i.e., state[k] & ~state[k+1] == 0
    for k in range(N - 1):
        if state[k] & ~state[k+1]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Initial network: Odd-Even Transposition (Bubble Sort)
    # This is known to be valid.
    current_net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            current_net.append((i, i + 1))
            
    # Verify initial network (should be fast)
    if not _is_valid_bitmask(current_net):
        # Fallback: should not happen
        pass

    best_net = current_net[:]
    best_len = len(current_net)
    
    # Greedy Deletion with Random Restarts/Shuffles
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(current_net):
                continue
            
            # Try to remove current_net[idx]
            removed = current_net.pop(idx)
            
            if _is_valid_bitmask(current_net):
                improved = True
                # Update best if smaller
                if len(current_net) < best_len:
                    best_len = len(current_net)
                    best_net = current_net[:]
            else:
                current_net.insert(idx, removed)
                
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
    # If we have time, we can try more aggressive search, 
    # but greedy deletion with random order is a strong baseline.
    
    # Final check on best_net to be absolutely sure
    if _is_valid_bitmask(best_net):
        return best_net
    else:
        return current_net

# EVOLVE-BLOCK-END
