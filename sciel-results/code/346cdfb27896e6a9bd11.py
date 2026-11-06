import numpy as np
import time
import random
import itertools

N = 16

# Pre-compute bit-parallel inputs
# Each wire is a vector of 1024 uint64s (1024 * 64 = 65536 bits)
# Input k (0 to 65535) is represented by the bit k in the vector.
# Wait, standard representation:
# Let's use a list of 16 arrays, each of shape (1024,) dtype uint64.
# Wire w has bit k set if input k has 1 at wire w.

def _init_bit_inputs():
    inputs = [np.zeros(1024, dtype=np.uint64) for _ in range(N)]
    for k in range(65536):
        # Determine bits for input k
        # Input k is a 16-bit integer.
        # Bit w of input k is (k >> w) & 1.
        # We need to set bit k in inputs[w] if that bit is 1.
        # This loop is 65536 * 16 = 1M operations. Fast enough for init.
        for w in range(N):
            if (k >> w) & 1:
                # Set bit k in inputs[w]
                # k is the index.
                # Array index: k // 64
                # Bit index: k % 64
                arr_idx = k // 64
                bit_idx = k % 64
                inputs[w][arr_idx] |= (1 << bit_idx)
    return inputs

BIT_INPUTS = _init_bit_inputs()

def _verify_bit(network):
    # Copy the inputs
    # List of arrays
    states = [arr.copy() for arr in BIT_INPUTS]
    
    for i, j in network:
        # min = AND, max = OR
        min_val = states[i] & states[j]
        max_val = states[i] | states[j]
        states[i] = min_val
        states[j] = max_val
        
    # Check if sorted:
    # For all inputs, state[0] <= state[1] <= ... <= state[15]
    # In bit-parallel:
    # state[i] <= state[j] means (state[i] & ~state[j]) == 0
    # i.e., there is no input where state[i] is 1 and state[j] is 0.
    
    for i in range(N - 1):
        # Check states[i] <= states[i+1]
        # If there is a 1 in states[i] and 0 in states[i+1], it's unsorted.
        # This is (states[i] & ~states[i+1])
        if np.any(states[i] & ~states[i+1]):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    if not _verify_bit(network):
        return network

    best_network = list(network)
    best_len = len(best_network)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    stuck_counter = 0
    
    while time.time() < deadline:
        current_len = len(network)
        if current_len == 0: break
        
        if random.random() < 0.85:
            # Try Removal
            idx = random.randint(0, current_len - 1)
            new_network = network[:idx] + network[idx+1:]
            
            if _verify_bit(new_network):
                network = new_network
                stuck_counter = 0
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = list(network)
            else:
                stuck_counter += 1
        else:
            # Try Mutation
            idx = random.randint(0, current_len - 1)
            new_pair = random.choice(all_pairs)
            
            if network[idx] == new_pair:
                continue
                
            new_network = network[:idx] + [new_pair] + network[idx+1:]
            
            if _verify_bit(new_network):
                network = new_network
                stuck_counter = 0
            else:
                stuck_counter += 1

        if stuck_counter > 50 and current_len > 1:
             idx1 = random.randint(0, current_len - 1)
             idx2 = random.randint(0, current_len - 1)
             if idx1 != idx2:
                 new_network = list(network)
                 new_network[idx1], new_network[idx2] = new_network[idx2], new_network[idx1]
                 if _verify_bit(new_network):
                     network = new_network
                     stuck_counter = 0
                     
    return best_network
