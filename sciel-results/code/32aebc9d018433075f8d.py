import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

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
    # bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    # This creates an array of shape (NUM_BLOCKS, BLOCK_SIZE, N).
    # Memory: 4096 * 64 * 18 * 4 bytes = ~18 MB. Acceptable.
    
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2) # Shape (NUM_BLOCKS, BLOCK_SIZE)
    
    # Now compute targets for each wire i
    # target[i] is an array of shape (NUM_BLOCKS, BLOCK_SIZE)
    # target[i][b, bit] = 1 if i >= N - popcounts[b, bit]
    targets = []
    for i in range(N):
        # Condition: i >= N - c  => c >= N - i
        mask = (popcounts >= (N - i))
        # Pack this boolean mask into uint64 arrays
        # We need to convert (NUM_BLOCKS, BLOCK_SIZE) bool to (NUM_BLOCKS,) uint64
        # Each uint64 packs 64 bools.
        # mask is (4096, 64).
        # We can view it as bytes?
        # mask.view(np.uint8).reshape(NUM_BLOCKS, 8) -> each byte is 8 bits.
        # Then pack 8 bytes into 1 uint64.
        # np.packbits is for 1D.
        # Let's do it manually or with a helper.
        
        # Helper to pack 2D bool array (B, 64) into 1D uint64 array (B,)
        def pack_bits_to_u64(arr_2d):
            # arr_2d shape (B, 64)
            # Flatten to (B*64,)
            flat = arr_2d.flatten()
            # Pack bits
            packed = np.packbits(flat, axis=0, bitorder='little') # Shape (B*8,)
            # Reshape to (B, 8)
            packed = packed.reshape(NUM_BLOCKS, 8)
            # View as uint64
            # Note: byte order. np.packbits with little gives LSB first in byte?
            # We need to ensure the bit order matches our simulation.
            # In simulation, bit 0 of the uint64 corresponds to the first input in the block.
            # np.packbits little: first element is bit 0.
            # So packed[b, 0] is bits 0-7 of block b.
            # We need to combine these 8 bytes into 1 uint64.
            # We can use .view(np.uint64) if aligned.
            # packed is C-contiguous.
            return packed.view(np.uint64).flatten()

        targets.append(pack_bits_to_u64(mask))
        
    # Initialize wires (identity permutation)
    # Wire i for input k is bit i of k.
    wires = []
    for i in range(N):
        # bits[:, :, i] is (NUM_BLOCKS, BLOCK_SIZE)
        # Pack it
        wires.append(pack_bits_to_u64(bits[:, :, i]))
        
    # Function to apply comparator
    def apply_comp(wires, i, j):
        new_wires = wires.copy()
        a = new_wires[i]
        b = new_wires[j]
        new_a = a & b
        new_b = a | b
        new_wires[i] = new_a
        new_wires[j] = new_b
        return new_wires

    # Function to calculate error (Hamming distance to target)
    def calc_error(wires):
        err = 0
        for i in range(N):
            diff = wires[i] ^ targets[i]
            # Popcount of diff
            # diff is uint64 array of length NUM_BLOCKS
            # Convert to uint8 and unpack
            # diff.view(np.uint8).reshape(NUM_BLOCKS, 8)
            # unpackbits
            # This is slow if done 153 times per step.
            # Optimization: Pre-compute popcount for all possible uint64? No.
            # Use np.bitwise_count if available.
            try:
                err += np.bitwise_count(diff).sum()
            except AttributeError:
                # Fallback
                u8 = diff.view(np.uint8).reshape(NUM_BLOCKS, 8)
                unpacked = np.unpackbits(u8, axis=1, count=8, bitorder='little')
                err += unpacked.sum()
        return err

    current_wires = wires
    current_error = calc_error(current_wires)
    network = []
    
    # Greedy Search
    while time.time() < deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        
        for i in range(N):
            for j in range(i + 1, N):
                new_wires = apply_comp(current_wires, i, j)
                new_error = calc_error(new_wires)
                if new_error < best_error:
                    best_error = new_error
                    best_comp = (i, j)
                    best_new_wires = new_wires
                    
        if best_comp is None:
            # Stuck
            break
            
        current_wires = best_new_wires
        current_error = best_error
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
