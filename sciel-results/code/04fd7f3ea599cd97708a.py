import numpy as np
import time
import random

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
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Pre-compute targets and initial state
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
        
    # Function to calculate error delta
    def calc_error_delta(wires, wire_errors, i, j):
        a = wires[i]
        b = wires[j]
        new_a = a & b
        new_b = a | b
        new_err_i = _calc_popcount_u64(new_a ^ targets[i])
        new_err_j = _calc_popcount_u64(new_b ^ targets[j])
        old_err_i = wire_errors[i]
        old_err_j = wire_errors[j]
        delta = (new_err_i + new_err_j) - (old_err_i + old_err_j)
        return delta, new_a, new_b, new_err_i, new_err_j

    def apply_comparator(wires, wire_errors, i, j):
        delta, new_a, new_b, new_err_i, new_err_j = calc_error_delta(wires, wire_errors, i, j)
        new_wires = wires.copy()
        new_wires[i] = new_a
        new_wires[j] = new_b
        new_wire_errors = wire_errors.copy()
        new_wire_errors[i] = new_err_i
        new_wire_errors[j] = new_err_j
        return new_wires, new_wire_errors, delta

    def get_error(wires, wire_errors):
        return wire_errors.sum()

    def is_sorted(wires, wire_errors):
        return wire_errors.sum() == 0

    def prune_network(network, wires, wire_errors, time_limit):
        """
        Remove redundant comparators from a valid network.
        Returns the pruned network if it remains valid, else the original.
        """
        if not is_sorted(wires, wire_errors):
            return network
            
        pruned = network[:]
        # We iterate backwards because removing a comparator affects the state
        # but we are checking if the *original* network minus this comparator still sorts.
        # Actually, a safer way is to try removing each comparator from the full list
        # and re-verify from scratch? No, that's expensive.
        # Better: Try removing comparators one by one from the end of the list.
        # If we remove (i,j) at position k, we need to check if the network without it sorts.
        # This requires re-running the simulation? Or we can use the fact that if the network
        # is valid, removing a comparator might break it.
        # A common heuristic: try removing the last comparator. If the network still sorts, keep it removed.
        # Then try the new last comparator, etc.
        
        # To check if a modified network sorts, we must re-simulate it on the bit-parallel state.
        # This is expensive. 
        # Alternative: Use the "local" property. If we remove a comparator, we can check if the 
        # state before that comparator was "already sorted" for the wires involved? No, that's not sufficient.
        
        # Let's implement a simple removal check:
        # 1. Take the current network.
        # 2. For each comparator in reverse order, try removing it.
        # 3. Re-simulate the remaining network on the initial wires to see if it sorts.
        # This is O(N^2 * K) where K is the number of comparators. 
        # With K~100 and N=18, simulation is fast (vectorized).
        
        current_net = network[:]
        
        while time.time() < time_limit:
            removed_any = False
            # Try removing from the end
            for k in range(len(current_net) - 1, -1, -1):
                if time.time() >= time_limit:
                    break
                test_net = current_net[:k] + current_net[k+1:]
                # Simulate test_net
                test_wires = [w.copy() for w in initial_wires]
                test_wire_errors = np.array([_calc_popcount_u64(test_wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
                
                valid = True
                for (i, j) in test_net:
                    a = test_wires[i]
                    b = test_wires[j]
                    new_a = a & b
                    new_b = a | b
                    new_err_i = _calc_popcount_u64(new_a ^ targets[i])
                    new_err_j = _calc_popcount_u64(new_b ^ targets[j])
                    test_wire_errors[i] = new_err_i
                    test_wire_errors[j] = new_err_j
                    test_wires[i] = new_a
                    test_wires[j] = new_b
                    
                if test_wire_errors.sum() == 0:
                    current_net = test_net
                    removed_any = True
                    break # Restart the reverse loop from the new end
            
            if not removed_any:
                break
                
        return current_net

    # Multi-start Simulated Annealing
    
    best_network = None
    best_len = float('inf')
    
    # Number of restarts depends on time
    # Each restart: init, SA loop, prune
    
    # Helper to run one SA restart
    def run_sa_restarts(num_restarts, time_budget_per_restart):
        nonlocal best_network, best_len
        
        for r in range(num_restarts):
            if time.time() > start_time + deadline - 10: # Leave 10s for final pruning
                break
                
            # 1. Initialize: Random network
            # Start with a valid network? Or random?
            # Random might be too far. Let's start with the greedy result?
            # No, the point is to find different basins.
            # Let's use a "randomized greedy" or just a random sequence of comparators.
            # A random sequence of length 100 is likely invalid.
            # Better: Start with a known valid network (e.g. odd-even) and perturb it?
            # Or start with the greedy result and perturb?
            
            # Let's try: Start with a random permutation of wires? No, wires are just state.
            # Let's start with the initial wires (identity) and apply random comparators.
            
            wires = [w.copy() for w in initial_wires]
            wire_errors = np.array([_calc_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
            current_net = []
            
            # Randomly add comparators until we have a "chance" to be valid?
            # Or just start with an empty net and add?
            # If we start empty, error is high.
            
            # Let's start with a "randomly shuffled" odd-even network?
            # Or simply: Run a fast greedy to get a valid net, then SA to improve it.
            # This combines the reliability of greedy with the exploration of SA.
            
            # Fast Greedy to get a starting point
            g_wires = [w.copy() for w in initial_wires]
            g_wire_errors = np.array([_calc_popcount_u64(g_wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
            g_net = []
            
            # Limit greedy time
            greedy_deadline = time.time() + 1.0
            while time.time() < greedy_deadline:
                best_error = g_wire_errors.sum()
                best_comp = None
                best_new_wires = None
                best_new_wire_errors = None
                
                # Sample a subset of comparators to speed up? Or all?
                # All is 153 pairs. Fast enough.
                for i in range(N):
                    for j in range(i + 1, N):
                        delta, new_a, new_b, new_err_i, new_err_j = calc_error_delta(g_wires, g_wire_errors, i, j)
                        new_error = best_error + delta
                        if new_error < best_error:
                            best_error = new_error
                            best_comp = (i, j)
                            best_new_wires = g_wires.copy()
                            best_new_wires[i] = new_a
                            best_new_wires[j] = new_b
                            best_new_wire_errors = g_wire_errors.copy()
                            best_new_wire_errors[i] = new_err_i
                            best_new_wire_errors[j] = new_err_j
                            
                if best_comp is None:
                    break
                g_wires = best_new_wires
                g_wire_errors = best_new_wire_errors
                g_net.append(best_comp)
                
                if best_error == 0:
                    break
            
            if best_error != 0:
                # Greedy failed to sort? Unlikely with 1s budget, but possible if stuck.
                # If stuck, skip this restart.
                continue
                
            # Now we have a valid network g_net.
            # Use it as the starting point for SA.
            # We want to minimize length.
            # SA State: The network itself? Or the wires?
            # If we change the network, we must re-simulate to get wires.
            # Moves:
            # 1. Remove a comparator (if it keeps the network valid).
            # 2. Add a comparator.
            # 3. Swap two comparators.
            
            # Since we want to minimize length, "Remove" is the most promising move.
            # "Add" increases length, so we only accept if it leads to a much better state?
            # Actually, standard SA minimizes an energy function.
            # Energy = Length + Lambda * Error?
            # If we restrict to valid networks, Energy = Length.
            # Then we only accept moves that keep the network valid.
            # This is a constrained optimization.
            
            # Let's do "Valid Network Hill Climbing" with restarts.
            # Move: Try removing a random comparator. If still valid, accept.
            # If stuck, try swapping? Or just restart?
            
            current_net = g_net[:]
            current_wires = g_wires[:]
            current_wire_errors = g_wire_errors[:]
            current_len = len(current_net)
            
            # Local search loop
            ls_deadline = time.time() + 5.0
            while time.time() < ls_deadline:
                improved = False
                # Try removing a random comparator
                if len(current_net) > 0:
                    idx = random.randint(0, len(current_net) - 1)
                    # Test removal
                    test_net = current_net[:idx] + current_net[idx+1:]
                    test_wires = [w.copy() for w in initial_wires]
                    test_wire_errors = np.array([_calc_popcount_u64(test_wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
                    
                    valid = True
                    for (i, j) in test_net:
                        a = test_wires[i]
                        b = test_wires[j]
                        new_a = a & b
                        new_b = a | b
                        new_err_i = _calc_popcount_u64(new_a ^ targets[i])
                        new_err_j = _calc_popcount_u64(new_b ^ targets[j])
                        test_wire_errors[i] = new_err_i
                        test_wire_errors[j] = new_err_j
                        test_wires[i] = new_a
                        test_wires[j] = new_b
                        
                    if test_wire_errors.sum() == 0:
                        # Accept removal
                        current_net = test_net
                        current_wires = test_wires
                        current_wire_errors = test_wire_errors
                        current_len -= 1
                        improved = True
                        if current_len < best_len:
                            best_len = current_len
                            best_network = current_net[:]
                            
                if not improved:
                    # Try swapping two comparators?
                    # Or just break and restart?
                    # Swapping might allow removal of others later.
                    if len(current_net) > 1:
                        idx1 = random.randint(0, len(current_net) - 1)
                        idx2 = random.randint(0, len(current_net) - 1)
                        if idx1 != idx2:
                            test_net = current_net[:]
                            test_net[idx1], test_net[idx2] = test_net[idx2], test_net[idx1]
                            # Verify validity
                            test_wires = [w.copy() for w in initial_wires]
                            test_wire_errors = np.array([_calc_popcount_u64(test_wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
                            
                            valid = True
                            for (i, j) in test_net:
                                a = test_wires[i]
                                b = test_wires[j]
                                new_a = a & b
                                new_b = a | b
                                new_err_i = _calc_popcount_u64(new_a ^ targets[i])
                                new_err_j = _calc_popcount_u64(new_b ^ targets[j])
                                test_wire_errors[i] = new_err_i
                                test_wire_errors[j] = new_err_j
                                test_wires[i] = new_a
                                test_wires[j] = new_b
                                
                            if test_wire_errors.sum() == 0:
                                current_net = test_net
                                current_wires = test_wires
                                current_wire_errors = test_wire_errors
                                improved = True
                                if current_len < best_len:
                                    best_len = current_len
                                    best_network = current_net[:]
                    
                    if not improved:
                        break # Local minimum, restart SA/greedy

    # Run the search
    # Estimate time per restart: ~1s greedy + 5s local search = 6s.
    # Budget 85s. ~10 restarts.
    
    # To be safe, we'll just run as many as fit.
    run_sa_restarts(100, 6.0)
    
    # Final Pruning on the best network found
    if best_network:
        # Re-simulate best_network to get final wires/errors for pruning
        final_wires = [w.copy() for w in initial_wires]
        final_wire_errors = np.array([_calc_popcount_u64(final_wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
        
        for (i, j) in best_network:
            a = final_wires[i]
            b = final_wires[j]
            new_a = a & b
            new_b = a | b
            new_err_i = _calc_popcount_u64(new_a ^ targets[i])
            new_err_j = _calc_popcount_u64(new_b ^ targets[j])
            final_wire_errors[i] = new_err_i
            final_wire_errors[j] = new_err_j
            final_wires[i] = new_a
            final_wires[j] = new_b
            
        if final_wire_errors.sum() == 0:
            # Prune
            pruned_net = prune_network(best_network, final_wires, final_wire_errors, time.time() + 5.0)
            if len(pruned_net) < best_len:
                best_network = pruned_net
                best_len = len(pruned_net)

    if best_network:
        return best_network
    else:
        # Fallback
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
