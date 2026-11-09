import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

# Pre-compute LUT for popcount
POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def _pack_bits_to_u64(arr_2d):
    flat = arr_2d.flatten()
    packed = np.packbits(flat, axis=0, bitorder='little')
    packed = packed.reshape(NUM_BLOCKS, 8)
    return np.ascontiguousarray(packed).view(np.uint64).flatten()

def _calc_popcount_u64(diff):
    # diff is 1D uint64 array
    # View as 2D uint8 array (NUM_BLOCKS, 8)
    u8_view = diff.reshape(-1, 8).view(np.uint8)
    counts = POPCOUNT_LUT[u8_view]
    return counts.sum()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Pre-compute target states and initial wires
    indices = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(NUM_BLOCKS, BLOCK_SIZE)
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2)
    
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        targets.append(_pack_bits_to_u64(mask))
        
    initial_wires = []
    for i in range(N):
        initial_wires.append(_pack_bits_to_u64(bits[:, :, i]))
        
    # Greedy Search
    wires = [w.copy() for w in initial_wires]
    wire_errors = np.array([_calc_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
    current_error = wire_errors.sum()
    
    network = []
    
    # Reserve time for pruning
    greedy_deadline = deadline - 10.0
    
    while time.time() < greedy_deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        best_new_wire_errors = None
        
        for i in range(N):
            for j in range(i + 1, N):
                a = wires[i]
                b = wires[j]
                new_a = a & b
                new_b = a | b
                new_err_i = _calc_popcount_u64(new_a ^ targets[i])
                new_err_j = _calc_popcount_u64(new_b ^ targets[j])
                old_err_i = wire_errors[i]
                old_err_j = wire_errors[j]
                delta = (new_err_i + new_err_j) - (old_err_i + old_err_j)
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
            break
            
        wires = best_new_wires
        current_error = best_error
        wire_errors = best_new_wire_errors
        network.append(best_comp)
        
        if current_error == 0:
            break
            
    # Pruning Phase
    if current_error == 0:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for k in range(len(network)):
                candidate_network = network[:k] + network[k+1:]
                
                sim_wires = [w.copy() for w in initial_wires]
                for i, j in candidate_network:
                    a = sim_wires[i]
                    b = sim_wires[j]
                    sim_wires[i] = a & b
                    sim_wires[j] = a | b
                
                valid = True
                for i in range(N):
                    if not np.array_equal(sim_wires[i], targets[i]):
                        valid = False
                        break
                
                if valid:
                    network = candidate_network
                    changed = True
                    break
                    
    # Fallback
    if current_error != 0:
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    return network
