"""Size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
SEED_COUNT = 153
HARD_DEADLINE = 88.0  # seconds, safety margin before 90s kill


def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps


def _bitonic_merge(n, start=0, reverse=False):
    """Generate comparators for a bitonic merge on wires [start, start+n)."""
    comps = []
    if n <= 1:
        return comps
    half = n // 2
    comps.extend(_bitonic_merge(half, start, reverse))
    comps.extend(_bitonic_merge(half, start + half, reverse))
    for i in range(half):
        a = start + i
        b = start + n - 1 - i
        if reverse:
            a, b = b, a
        if a < b:
            comps.append((a, b))
        else:
            comps.append((b, a))
    return comps


def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (power of 2)."""
    comps = []
    if n <= 1:
        return comps
    half = n // 2
    comps.extend(_bitonic_sort_network(half))
    comps.extend(_bitonic_sort_network(half))
    comps.extend(_bitonic_merge(n))
    return comps


def _verify_network_bitparallel(comps, n=N):
    """
    Verify that the network sorts all Boolean inputs using bit-parallelism.
    Returns True if correct, False otherwise.
    """
    if not comps:
        return True
    
    # Precompute comparator operations
    comp_ops = []
    for i, j in comps:
        if i > j:
            i, j = j, i
        mask_i = 1 << i
        mask_j = 1 << j
        comp_ops.append((i, j, mask_i, mask_j))
    
    for mask in range(1 << n):
        wires = mask
        for i, j, mask_i, mask_j in comp_ops:
            bit_i = (wires & mask_i) != 0
            bit_j = (wires & mask_j) != 0
            if bit_i and not bit_j:
                # Swap: set j, clear i
                wires |= mask_j
                wires &= ~mask_i
        # Check if sorted: all 0s before all 1s
        for i in range(n - 1):
            if ((wires >> i) & 1) > ((wires >> (i + 1)) & 1):
                return False
    return True


def _construct_initial_network():
    """
    Construct an initial sorting network for N=18 wires.
    Uses a combination of known structures.
    """
    # Start with odd-even transposition (153 comparators)
    return _odd_even_transposition_network(N)


def _vectorized_greedy_build(deadline):
    """
    Build a sorting network using vectorized greedy search over all 2^18 Boolean inputs.
    This is a structural change: instead of pruning a fixed network, we build one from scratch
    by greedily selecting comparators that maximize the reduction in total sorting error.
    """
    start_time = time.time()
    
    # Pre-compute target states for all 2^18 inputs
    # Target for input k: wire i is 1 if i >= N - popcount(k)
    # We represent the state of all inputs as a bitset for each wire.
    # Wire i's state is a 2^18-bit integer (or numpy array of uint64s).
    
    num_inputs = 1 << N
    
    # Use numpy arrays of uint64 for efficiency
    # 2^18 inputs = 262144 bits = 4096 uint64s
    num_blocks = num_inputs // 64
    
    # Initialize wires: wire i is 1 for inputs where bit i is set
    # We need to generate the bitsets for each wire.
    # Input k has bit i set if (k >> i) & 1 == 1.
    
    # To generate these efficiently, we can use numpy.
    # indices = np.arange(num_inputs)
    # bits_i = (indices >> i) & 1
    # packed = np.packbits(bits_i, axis=0, bitorder='little')
    
    # Let's try a faster way using the structure of binary numbers.
    # For wire i, the pattern is 0...0 1...1 0...0 1...1 ...
    # Period is 2^(i+1), with 2^i zeros and 2^i ones.
    
    wires = []
    for i in range(N):
        # Create the bit pattern for wire i
        # Pattern: 2^i zeros, 2^i ones, repeated num_inputs / 2^(i+1) times
        # But we need to pack it into uint64s.
        
        # Alternative: generate the array of 0s and 1s
        # This might be memory intensive: 18 * 262144 bytes = ~4.5 MB. That's fine.
        
        # Let's use the simple generation method first, it's fast enough for initialization.
        indices = np.arange(num_inputs, dtype=np.uint32)
        bits = (indices >> i) & 1
        # Pack bits into uint64
        # np.packbits expects 0/1 or boolean.
        packed = np.packbits(bits, axis=0, bitorder='little')
        # Reshape to (num_blocks, 8) and view as uint64
        packed = packed.reshape(num_blocks, 8).view(np.uint64).flatten()
        wires.append(packed.copy())
    
    # Compute targets
    # Target for wire i: 1 if i >= N - popcount(k)
    # popcount(k) is the number of 1s in k.
    # We can compute popcount for all k using numpy.
    # k = np.arange(num_inputs)
    # popcounts = np.array([bin(k).count('1') for k in k]) # Too slow
    # Use a lookup table or bitwise trick.
    
    # Bitwise trick for popcount in numpy:
    # x = x - ((x >> 1) & 0x55555555)
    # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    # x = (x + (x >> 4)) & 0x0F0F0F0F
    # x = x + (x >> 8)
    # x = x + (x >> 16)
    # popcount = (x & 0xFF) >> 0 ? No, just x & 0xFF
    
    indices = np.arange(num_inputs, dtype=np.uint32)
    x = indices
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    popcounts = x & 0xFF
    
    # Targets: wire i is 1 if i >= N - popcount(k)
    # i >= N - pc  <=>  pc >= N - i
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        packed = np.packbits(mask.astype(np.uint8), axis=0, bitorder='little')
        packed = packed.reshape(num_blocks, 8).view(np.uint64).flatten()
        targets.append(packed)
    
    # Initialize error calculation
    # Error for wire i = popcount(wires[i] XOR targets[i])
    # We need a fast popcount for uint64 arrays.
    # np.bitwise_count is available in numpy >= 1.17?
    # Let's check. If not, we can use a lookup table or other method.
    
    def _popcount_u64(arr):
        # arr is a numpy array of uint64
        # Try bitwise_count
        try:
            return np.bitwise_count(arr).sum()
        except AttributeError:
            # Fallback: unpack to bytes and use a lookup table or other method
            # For now, let's assume bitwise_count exists or use a slow method
            # Since we are in a competitive setting, let's use a lookup table for 8-bit chunks
            # This is slower but robust.
            # Actually, let's just use the unpack method which is reasonably fast.
            u8 = arr.view(np.uint8).reshape(-1, 8)
            # Sum the bits in each byte
            # We can use a precomputed lookup table for 0-255
            lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
            return lut[u8].sum()
    
    # Current errors
    wire_errors = np.array([_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
    current_error = wire_errors.sum()
    
    network = []
    
    # All possible pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Greedy loop
    while time.time() < deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        best_new_wire_errors = None
        
        # Evaluate all single comparators
        for i, j in all_pairs:
            a = wires[i]
            b = wires[j]
            new_a = a & b
            new_b = a | b
            
            new_err_i = _popcount_u64(new_a ^ targets[i])
            new_err_j = _popcount_u64(new_b ^ targets[j])
            
            delta = (new_err_i + new_err_j) - (wire_errors[i] + wire_errors[j])
            new_error = current_error + delta
            
            if new_error < best_error:
                best_error = new_error
                best_comp = (i, j)
                best_new_wires = wires.copy()
                best_new_wires[i] = new_a
                best_new_wires[j] = new_b
                best_new_wire_errors = wire_errors.copy()
                best_new_wire_errors[i] = new_err_i
                best_new_wire_errors[j] = new_err_j
        
        if best_comp is None:
            # No single comparator improves the error.
            # Try 2-step look-ahead? Or break?
            # For now, let's try a limited 2-step look-ahead if we have time.
            found_improvement = False
            # Limit the number of pairs to check in 2-step to avoid timeout
            # Only check a subset of pairs for the first step
            subset_size = min(len(all_pairs), 20)
            for i1, j1 in all_pairs[:subset_size]:
                if time.time() > deadline - 5:
                    break
                
                a1 = wires[i1]
                b1 = wires[j1]
                new_a1 = a1 & b1
                new_b1 = a1 | b1
                
                new_err_i1 = _popcount_u64(new_a1 ^ targets[i1])
                new_err_j1 = _popcount_u64(new_b1 ^ targets[j1])
                
                err_after_first = current_error + (new_err_i1 + new_err_j1 - wire_errors[i1] - wire_errors[j1])
                
                temp_wires = wires.copy()
                temp_wires[i1] = new_a1
                temp_wires[j1] = new_b1
                temp_wire_errors = wire_errors.copy()
                temp_wire_errors[i1] = new_err_i1
                temp_wire_errors[j1] = new_err_j1
                
                # Try second comparator
                for i2, j2 in all_pairs:
                    if i2 == i1 and j2 == j1:
                        continue
                    
                    a2 = temp_wires[i2]
                    b2 = temp_wires[j2]
                    new_a2 = a2 & b2
                    new_b2 = a2 | b2
                    
                    new_err_i2 = _popcount_u64(new_a2 ^ targets[i2])
                    new_err_j2 = _popcount_u64(new_b2 ^ targets[j2])
                    
                    err_after_second = err_after_first + (new_err_i2 + new_err_j2 - temp_wire_errors[i2] - temp_wire_errors[j2])
                    
                    if err_after_second < best_error:
                        best_error = err_after_second
                        best_comp = ((i1, j1), (i2, j2))  # Mark as 2-step
                        best_new_wires = temp_wires.copy()
                        best_new_wires[i2] = new_a2
                        best_new_wires[j2] = new_b2
                        best_new_wire_errors = temp_wire_errors.copy()
                        best_new_wire_errors[i2] = new_err_i2
                        best_new_wire_errors[j2] = new_err_j2
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if best_comp is None:
                # Still stuck, break
                break
        
        # Apply the best move(s)
        if isinstance(best_comp, tuple) and len(best_comp) == 2 and isinstance(best_comp[0], tuple):
            # 2-step move
            network.extend(best_comp)
        else:
            network.append(best_comp)
        
        wires = best_new_wires
        current_error = best_error
        wire_errors = best_new_wire_errors
        
        if current_error == 0:
            break
    
    return network


def construct():
    """
    Construct a compare-exchange network that sorts all 18-input sequences.
    Returns ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + HARD_DEADLINE
    
    # Strategy 1: Vectorized greedy build (Structural change)
    # This builds a network from scratch, potentially finding smaller networks
    # than pruning a fixed seed.
    
    # We use a portion of the time for this.
    greedy_deadline = start_time + 60.0  # Leave 28s for fallback/pruning
    
    greedy_network = _vectorized_greedy_build(greedy_deadline)
    
    # Verify the greedy network
    if greedy_network and _verify_network_bitparallel(greedy_network, N):
        # Try to prune the greedy network
        # Simple pruning: try removing each comparator in reverse order
        pruned = greedy_network[:]
        for idx in range(len(pruned) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            candidate = pruned[:idx] + pruned[idx + 1:]
            if _verify_network_bitparallel(candidate, N):
                pruned = candidate
        
        return pruned
    
    # Fallback: If greedy failed or is invalid, use the original seed-based approach
    comparators = _construct_initial_network()
    
    # Verify it works
    if not _verify_network_bitparallel(comparators, N):
        return comparators
    
    # Try to prune redundant comparators
    if time.time() < deadline - 5:
        improved = True
        while improved:
            improved = False
            for idx in range(len(comparators) - 1, -1, -1):
                if time.time() > deadline - 2:
                    return comparators
                candidate = comparators[:idx] + comparators[idx + 1:]
                if _verify_network_bitparallel(candidate, N):
                    comparators = candidate
                    improved = True
                    break
    
    return comparators
