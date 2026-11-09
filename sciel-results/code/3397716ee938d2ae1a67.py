import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

def _pack_bits_to_u64(arr_2d):
    flat = arr_2d.flatten()
    packed = np.packbits(flat, axis=0, bitorder='little')
    packed = packed.reshape(NUM_BLOCKS, 8)
    return packed.view(np.uint64).flatten()

def _calc_popcount_u64(diff):
    try:
        return np.bitwise_count(diff).sum()
    except AttributeError:
        u8 = diff.view(np.uint8).reshape(NUM_BLOCKS, 8)
        unpacked = np.unpackbits(u8, axis=1, count=8, bitorder='little')
        return unpacked.sum()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Use more of the 90s budget

    # Pre-compute target states for all 2^18 inputs
    # Target for input k: wire i is 1 if i >= N - popcount(k)
    # We represent state as 18 arrays of uint64, each of length NUM_BLOCKS
    # Each uint64 holds 64 inputs.
    
    # Generate indices for all inputs
    # indices[b, bit] = b * 64 + bit
    indices = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(NUM_BLOCKS, BLOCK_SIZE)
    
    # Compute popcount for all inputs
    # We can do this by unpacking bits and summing
    # But we need popcount of the 18-bit integer.
    # Let's compute popcount for each of the 262144 inputs.
    # Using numpy:
    # Expand indices to bits?
    # Simpler: Loop over blocks and bits? No, vectorized.
    
    # Let's compute popcount using a lookup table for 8-bit chunks?
    # Or just use the fact that we can calculate the target directly.
    # Target wire i for input k is 1 if popcount(k) <= N - 1 - i ?
    # No: Sorted 0s then 1s.
    # If popcount(k) = c, there are c ones.
    # Wires 0 to N-1-c are 0.
    # Wires N-c to N-1 are 1.
    # So target[i] = 1 if i >= N - c.
    
    # Let's compute popcount for all indices.
    # indices is (NUM_BLOCKS, BLOCK_SIZE)
    # We can compute popcount by summing bits.
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2) # Shape (NUM_BLOCKS, BLOCK_SIZE)
    
    # Now compute targets for each wire i
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        targets.append(_pack_bits_to_u64(mask))
        
    # Initialize wires (identity permutation)
    wires = []
    for i in range(N):
        wires.append(_pack_bits_to_u64(bits[:, :, i]))
        
    # Pre-compute error for each wire
    wire_errors = np.array([_calc_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
    current_error = wire_errors.sum()
    
    # Function to calculate error delta when applying comparator (i, j)
    def calc_error_delta(wires, wire_errors, i, j):
        a = wires[i]
        b = wires[j]
        new_a = a & b
        new_b = a | b
        # Compute new errors for wires i and j
        new_err_i = _calc_popcount_u64(new_a ^ targets[i])
        new_err_j = _calc_popcount_u64(new_b ^ targets[j])
        old_err_i = wire_errors[i]
        old_err_j = wire_errors[j]
        delta = (new_err_i + new_err_j) - (old_err_i + old_err_j)
        return delta, new_a, new_b, new_err_i, new_err_j

    current_wires = wires
    network = []
    
    # Greedy Search
    while time.time() < deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        best_new_wire_errors = None
        
        for i in range(N):
            for j in range(i + 1, N):
                delta, new_a, new_b, new_err_i, new_err_j = calc_error_delta(current_wires, wire_errors, i, j)
                new_error = current_error + delta
                if new_error < best_error:
                    best_error = new_error
                    best_comp = (i, j)
                    # Only create new wires list when we find a better one
                    best_new_wires = current_wires.copy()
                    best_new_wires[i] = new_a
                    best_new_wires[j] = new_b
                    best_new_wire_errors = wire_errors.copy()
                    best_new_wire_errors[i] = new_err_i
                    best_new_wire_errors[j] = new_err_j
                    
        if best_comp is None:
            # Stuck
            break
            
        current_wires = best_new_wires
        current_error = best_error
        wire_errors = best_new_wire_errors
        network.append(best_comp)
        
        if current_error == 0:
            break
            
    # Check validity
    if current_error == 0:
        return network
    else:
        # Fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
