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
        
    # Wire matrix: shape (N, NUM_BLOCKS) uint64
    wires_mat = np.array([w.copy() for w in initial_wires], dtype=np.uint64)
    targets_mat = np.array([t.copy() for t in targets], dtype=np.uint64)
    
    # Greedy Search
    wire_errors = np.array([_calc_popcount_u64(wires_mat[i] ^ targets_mat[i]) for i in range(N)], dtype=np.int64)
    current_error = wire_errors.sum()
    
    network = []
    
    # Reserve time for pruning and local search
    greedy_deadline = deadline - 20.0

    while time.time() < greedy_deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        best_new_wire_errors = None
        
        for i in range(N):
            for j in range(i + 1, N):
                a = wires_mat[i]
                b = wires_mat[j]
                new_a = a & b
                new_b = a | b
                new_err_i = _calc_popcount_u64(new_a ^ targets_mat[i])
                new_err_j = _calc_popcount_u64(new_b ^ targets_mat[j])
                old_err_i = wire_errors[i]
                old_err_j = wire_errors[j]
                delta = (new_err_i + new_err_j) - (old_err_i + old_err_j)
                new_error = current_error + delta
                
                if new_error < best_error:
                    best_error = new_error
                    best_comp = (i, j)
                    best_new_wires = wires_mat.copy()
                    best_new_wires[i] = new_a
                    best_new_wires[j] = new_b
                    best_new_wire_errors = wire_errors.copy()
                    best_new_wire_errors[i] = new_err_i
                    best_new_wire_errors[j] = new_err_j
                    
        if best_comp is None:
            break
            
        wires_mat = best_new_wires
        current_error = best_error
        wire_errors = best_new_wire_errors
        network.append(best_comp)
        
        if current_error == 0:
            break
        
        # Periodic pruning during greedy search
        if len(network) > 0 and len(network) % 5 == 0 and time.time() < greedy_deadline - 1.0:
            # Quick single-pass pruning using matrix simulation
            for k in range(len(network) - 1, -1, -1):
                if time.time() > greedy_deadline - 1.0:
                    break
                candidate_network = network[:k] + network[k+1:]
                
                sim_wires = wires_mat.copy()
                for i, j in candidate_network:
                    a = sim_wires[i]
                    b = sim_wires[j]
                    sim_wires[i] = a & b
                    sim_wires[j] = a | b
                
                if np.array_equal(sim_wires, targets_mat):
                    network = candidate_network
                    wires_mat = sim_wires
                    wire_errors = np.array([_calc_popcount_u64(wires_mat[i] ^ targets_mat[i]) for i in range(N)], dtype=np.int64)
                    current_error = wire_errors.sum()
                    break
            
    # Local Search Phase - try replacing one comparator to escape local optima
    if current_error > 0 and time.time() < deadline - 15.0:
        local_deadline = deadline - 15.0
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            for k in range(len(network)):
                if time.time() > local_deadline:
                    break
                # Try replacing network[k] with each other pair
                old_comp = network[k]
                for i in range(N):
                    for j in range(i + 1, N):
                        if (i, j) == old_comp:
                            continue
                        candidate_network = network[:k] + [(i, j)] + network[k+1:]
                        
                        sim_wires = wires_mat.copy()
                        for ii, jj in candidate_network:
                            a = sim_wires[ii]
                            b = sim_wires[jj]
                            sim_wires[ii] = a & b
                            sim_wires[jj] = a | b
                        
                        new_errors = np.array([_calc_popcount_u64(sim_wires[ii] ^ targets_mat[ii]) for ii in range(N)], dtype=np.int64)
                        new_error = new_errors.sum()
                        
                        if new_error < current_error:
                            network = candidate_network
                            wires_mat = sim_wires
                            wire_errors = new_errors
                            current_error = new_error
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Pruning Phase - vectorized using matrix operations
    if current_error == 0:
        prune_deadline = deadline - 2.0
        for pass_num in range(50):
            if time.time() > prune_deadline:
                break
            changed = False
            # Try removals in reverse order first
            for k in range(len(network) - 1, -1, -1):
                if time.time() > prune_deadline:
                    break
                candidate_network = network[:k] + network[k+1:]
                
                sim_wires = wires_mat.copy()
                for i, j in candidate_network:
                    a = sim_wires[i]
                    b = sim_wires[j]
                    sim_wires[i] = a & b
                    sim_wires[j] = a | b
                
                if np.array_equal(sim_wires, targets_mat):
                    network = candidate_network
                    wires_mat = sim_wires
                    changed = True
                    break
            if not changed:
                # Try forward order
                for k in range(len(network)):
                    if time.time() > prune_deadline:
                        break
                    candidate_network = network[:k] + network[k+1:]
                    
                    sim_wires = wires_mat.copy()
                    for i, j in candidate_network:
                        a = sim_wires[i]
                        b = sim_wires[j]
                        sim_wires[i] = a & b
                        sim_wires[j] = a | b
                    
                    if np.array_equal(sim_wires, targets_mat):
                        network = candidate_network
                        wires_mat = sim_wires
                        changed = True
                        break
            if not changed:
                break
                    
    # Fallback
    if current_error != 0:
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    return network
