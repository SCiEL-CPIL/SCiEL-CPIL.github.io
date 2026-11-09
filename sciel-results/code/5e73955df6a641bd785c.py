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
        
    # Helper to check if a network is valid
    def _check_network_validity(net):
        sim_wires = [w.copy() for w in initial_wires]
        for i, j in net:
            a = sim_wires[i]
            b = sim_wires[j]
            sim_wires[i] = a & b
            sim_wires[j] = a | b
        
        for i in range(N):
            if not np.array_equal(sim_wires[i], targets[i]):
                return False
        return True

    # Helper to prune a network
    def _prune_network(net):
        # Try removing each comparator in reverse order first, then forward
        # We iterate until no more changes can be made
        while True:
            if time.time() > deadline - 5.0:
                break
            changed = False
            # Reverse pass
            for k in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                candidate = net[:k] + net[k+1:]
                if _check_network_validity(candidate):
                    net = candidate
                    changed = True
                    break
            if changed:
                continue
            # Forward pass
            for k in range(len(net)):
                if time.time() > deadline - 5.0:
                    break
                candidate = net[:k] + net[k+1:]
                if _check_network_validity(candidate):
                    net = candidate
                    changed = True
                    break
            if not changed:
                break
        return net

    # Greedy Search
    wires = [w.copy() for w in initial_wires]
    wire_errors = np.array([_calc_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
    current_error = wire_errors.sum()
    
    network = []
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Reserve time for pruning
    greedy_deadline = deadline - 15.0
    
    while time.time() < greedy_deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        best_new_wire_errors = None
        
        # 1-step greedy
        for i, j in all_pairs:
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
                
        # If 1-step failed, try 2-step look-ahead
        if best_comp is None:
            found_improvement = False
            # Limit look-ahead depth/time to prevent blowup
            # We only do this if we have plenty of time left
            if time.time() < greedy_deadline - 10.0:
                for i1, j1 in all_pairs:
                    if time.time() > greedy_deadline - 10.0:
                        break
                    a1 = wires[i1]
                    b1 = wires[j1]
                    new_a1 = a1 & b1
                    new_b1 = a1 | b1
                    new_err_i1 = _calc_popcount_u64(new_a1 ^ targets[i1])
                    new_err_j1 = _calc_popcount_u64(new_b1 ^ targets[j1])
                    err_after_first = current_error + (new_err_i1 + new_err_j1 - wire_errors[i1] - wire_errors[j1])
                    
                    # If the first step already increases error significantly, skip?
                    # Or just proceed.
                    
                    temp_wires = wires.copy()
                    temp_wires[i1] = new_a1
                    temp_wires[j1] = new_b1
                    temp_wire_errors = wire_errors.copy()
                    temp_wire_errors[i1] = new_err_i1
                    temp_wire_errors[j1] = new_err_j1
                    
                    for i2, j2 in all_pairs:
                        if i2 == i1 and j2 == j1:
                            continue
                        a2 = temp_wires[i2]
                        b2 = temp_wires[j2]
                        new_a2 = a2 & b2
                        new_b2 = a2 | b2
                        new_err_i2 = _calc_popcount_u64(new_a2 ^ targets[i2])
                        new_err_j2 = _calc_popcount_u64(new_b2 ^ targets[j2])
                        err_after_second = err_after_first + (new_err_i2 + new_err_j2 - temp_wire_errors[i2] - temp_wire_errors[j2])
                        
                        if err_after_second < best_error:
                            best_error = err_after_second
                            best_comp = ((i1, j1), (i2, j2))
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
                # Stuck, break
                break
        
        # Apply the best move(s)
        if isinstance(best_comp, tuple) and len(best_comp) == 2 and isinstance(best_comp[0], tuple):
            network.extend(best_comp)
        else:
            network.append(best_comp)
            
        wires = best_new_wires
        current_error = best_error
        wire_errors = best_new_wire_errors
        
        if current_error == 0:
            break
        
        # Periodic pruning during greedy search to keep network small
        # This helps escape local minima where a comparator becomes redundant
        if len(network) > 0 and len(network) % 10 == 0 and time.time() < greedy_deadline - 5.0:
            pruned_net = _prune_network(network)
            if len(pruned_net) < len(network):
                # Re-simulate to get current wire states and errors
                wires = [w.copy() for w in initial_wires]
                for i, j in pruned_net:
                    a = wires[i]
                    b = wires[j]
                    wires[i] = a & b
                    wires[j] = a | b
                wire_errors = np.array([_calc_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
                current_error = wire_errors.sum()
                network = pruned_net
                # If pruning led to a state with error 0, we can stop
                if current_error == 0:
                    break

    # Final Pruning Phase
    if current_error == 0:
        prune_deadline = deadline - 2.0
        if time.time() < prune_deadline:
            network = _prune_network(network)
            
    # Fallback
    if current_error != 0:
        # If we didn't reach 0 error, return the best effort or a known valid network?
        # The task requires sorting EVERY input. If error != 0, it's invalid.
        # The fallback should be a valid network, e.g. the seed odd-even transposition
        # which is 153 comparators.
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    return network
