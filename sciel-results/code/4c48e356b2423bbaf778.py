import numpy as np
import time
import random

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

def _simulate_network(network, initial_wires):
    """Simulate a network on the initial wires, returning the final wire states."""
    wires = [w.copy() for w in initial_wires]
    for i, j in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    return wires

def _check_network_validity(network, initial_wires, targets):
    """Check if the network sorts all inputs."""
    wires = _simulate_network(network, initial_wires)
    for i in range(N):
        if not np.array_equal(wires[i], targets[i]):
            return False
    return True

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
    
    # Reserve time for pruning and local search
    greedy_deadline = deadline - 20.0
    
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
        prune_deadline = deadline - 5.0
        while changed and time.time() < prune_deadline:
            changed = False
            for k in range(len(network)):
                candidate_network = network[:k] + network[k+1:]
                
                valid = _check_network_validity(candidate_network, initial_wires, targets)
                
                if valid:
                    network = candidate_network
                    changed = True
                    break
                    
    # Local Search Phase: Random swaps and insertions to escape local minima
    if current_error == 0 and time.time() < deadline - 2.0:
        best_network = network
        best_len = len(network)
        
        # Simple hill climbing on network size
        ls_deadline = deadline - 2.0
        no_improvement_count = 0
        max_no_improvement = 50
        
        while time.time() < ls_deadline and no_improvement_count < max_no_improvement:
            # Try a random move: insert, delete, or swap
            move_type = random.choice(['insert', 'delete', 'swap'])
            
            if move_type == 'insert' and len(network) < 200:
                # Insert a random comparator at a random position
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = random.randint(0, len(network))
                candidate_network = network[:pos] + [(i, j)] + network[pos:]
                
            elif move_type == 'delete' and len(network) > 1:
                # Delete a random comparator
                pos = random.randint(0, len(network) - 1)
                candidate_network = network[:pos] + network[pos+1:]
                
            elif move_type == 'swap' and len(network) > 1:
                # Swap two random comparators
                pos1, pos2 = random.sample(range(len(network)), 2)
                candidate_network = network.copy()
                candidate_network[pos1], candidate_network[pos2] = candidate_network[pos2], candidate_network[pos1]
            else:
                no_improvement_count += 1
                continue
            
            # Check validity
            if _check_network_validity(candidate_network, initial_wires, targets):
                if len(candidate_network) < best_len:
                    best_network = candidate_network
                    best_len = len(candidate_network)
                    network = candidate_network
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

        network = best_network

    # Fallback
    if current_error != 0:
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    return network
