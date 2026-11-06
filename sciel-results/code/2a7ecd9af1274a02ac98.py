import numpy as np
import random
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0

# EVOLVE-BLOCK-START

def _verify_network_numpy(layers):
    """
    Verify the network using NumPy for speed.
    Returns True if the network sorts all 2^N binary inputs.
    """
    n = N
    total = 1 << n
    
    # Create an array of shape (total, n) containing all binary vectors
    # We can generate this efficiently
    # Using bit operations or simply arange and unpacking
    # np.array([[ (i >> j) & 1 for j in range(n) ] for i in range(total)]) is slow.
    
    # Better:
    # Create a range of numbers 0 to total-1
    # Convert to binary strings? No.
    
    # Vectorized bit extraction:
    # We want bits[i, j] = (i >> j) & 1
    # This can be done with:
    # indices = np.arange(total)[:, None]
    # shifts = np.arange(n)[None, :]
    # bits = (indices >> shifts) & 1
    
    # However, this creates a (131072, 17) int array. 
    # Memory is fine (131072 * 17 * 4 bytes ~ 9MB).
    
    # Let's try to be even faster.
    # We can just iterate over layers and apply them to the NumPy array.
    
    # Initialize the input matrix
    # Using the bit shift trick
    idx = np.arange(total, dtype=np.int64)[:, None]
    shifts = np.arange(n, dtype=np.int64)[None, :]
    # Note: We need to be careful with endianness or just use standard shifts.
    # (idx >> shifts) & 1 gives bit j of number idx.
    # We want wire 0 to be the LSB (bit 0).
    # So bits[:, 0] should be (idx >> 0) & 1.
    # bits[:, j] should be (idx >> j) & 1.
    
    state = (idx >> shifts) & 1
    
    for layer in layers:
        if not layer:
            continue
        # Process each comparator in the layer
        # Since comparators in a layer are disjoint, we can update in place?
        # No, we must be careful. If we update state[:, i] and state[:, j], 
        # and later in the same layer we have another comparator, it doesn't affect 
        # because they are disjoint.
        # So we can just loop through comparators in the layer and update state.
        
        for i, j in layer:
            # We need to swap state[:, i] and state[:, j] where state[:, i] > state[:, j]
            # Since values are 0 or 1, swap if state[:, i] == 1 and state[:, j] == 0.
            # mask = (state[:, i] == 1) & (state[:, j] == 0)
            
            # To do this in place efficiently:
            # We can use np.where or manual indexing.
            
            # Let's extract columns
            col_i = state[:, i]
            col_j = state[:, j]
            
            # Find where swap is needed: col_i > col_j
            # Since binary, this is col_i & ~col_j
            swap_mask = col_i & (~col_j)
            
            # If any swap is needed
            if np.any(swap_mask):
                # Perform swap
                # state[:, i][swap_mask] = 0
                # state[:, j][swap_mask] = 1
                
                # Advanced indexing
                state[swap_mask, i] = 0
                state[swap_mask, j] = 1
                
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    # This means state[:, :-1] <= state[:, 1:]
    # Or simply, check if the final state is sorted.
    # A binary vector is sorted if it is of the form 0...01...1.
    # So for each row, the transition from 0 to 1 happens at most once.
    
    # Check: for all rows, if state[:, k] == 1 and state[:, k+1] == 0, it's unsorted.
    # unsorted_mask = (state[:, :-1] == 1) & (state[:, 1:] == 0)
    # if np.any(unsorted_mask): return False
    
    # Faster check:
    # For each row, the number of 1s must be preserved? No, sorting doesn't change count of 1s.
    # But we don't need to check count.
    # Just check adjacency.
    
    if np.any((state[:, :-1] > state[:, 1:])):
        return False
        
    return True

def _count_comparators(layers):
    return sum(len(layer) for layer in layers)

def _generate_seed_network():
    """Generate the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _random_comparator():
    i = random.randint(0, N - 2)
    j = i + random.randint(1, N - 1 - i)
    return (i, j)

def _mutate_network(layers):
    """
    Mutate a network.
    Returns a new network (list of lists of tuples).
    """
    new_layers = [list(layer) for layer in layers]
    
    mutation_type = random.random()
    
    if mutation_type < 0.5:
        # Add a random comparator to a random layer
        if new_layers:
            layer_idx = random.randint(0, len(new_layers) - 1)
            # Ensure disjointness? 
            # If we add a comparator that shares a wire with an existing one in the layer,
            # it's no longer a valid layer (parallel execution conflict).
            # We must ensure the new comparator is disjoint from all in the layer.
            
            used_wires = set()
            for i, j in new_layers[layer_idx]:
                used_wires.add(i)
                used_wires.add(j)
            
            # Try to find a valid comparator
            for _ in range(10):
                comp = _random_comparator()
                if comp[0] not in used_wires and comp[1] not in used_wires:
                    new_layers[layer_idx].append(comp)
                    break
        else:
            # No layers? Create one
            new_layers.append([_random_comparator()])
            
    elif mutation_type < 0.8:
        # Remove a random comparator from a random layer
        if new_layers:
            layer_idx = random.randint(0, len(new_layers) - 1)
            if new_layers[layer_idx]:
                comp_idx = random.randint(0, len(new_layers[layer_idx]) - 1)
                new_layers[layer_idx].pop(comp_idx)
                if not new_layers[layer_idx]:
                    # Optionally remove empty layer to reduce depth overhead?
                    # But depth is not the score, comparator count is.
                    # However, empty layers don't add comparators.
                    # Keeping them might allow easier insertion later?
                    # Let's keep them for now, or remove to simplify.
                    # Removing empty layers is safe.
                    new_layers.pop(layer_idx)
                    
    else:
        # Swap a comparator between two layers
        if len(new_layers) >= 2:
            l1 = random.randint(0, len(new_layers) - 1)
            l2 = random.randint(0, len(new_layers) - 1)
            if l1 != l2 and new_layers[l1] and new_layers[l2]:
                c1_idx = random.randint(0, len(new_layers[l1]) - 1)
                c2_idx = random.randint(0, len(new_layers[l2]) - 1)
                
                c1 = new_layers[l1][c1_idx]
                c2 = new_layers[l2][c2_idx]
                
                # Check if c1 can go into l2
                used2 = set()
                for i, j in new_layers[l2]:
                    used2.add(i); used2.add(j)
                used2.discard(c2[0]); used2.discard(c2[1])
                
                # Check if c2 can go into l1
                used1 = set()
                for i, j in new_layers[l1]:
                    used1.add(i); used1.add(j)
                used1.discard(c1[0]); used1.discard(c1[1])
                
                if c1[0] not in used2 and c1[1] not in used2 and \
                   c2[0] not in used1 and c2[1] not in used1:
                    # Perform swap
                    new_layers[l1][c1_idx] = c2
                    new_layers[l2][c2_idx] = c1
                else:
                    # Just move c1 to l2 if possible? Or do nothing.
                    pass
    return new_layers

def construct():
    start_time = time.time()
    
    # 1. Generate Seed
    seed_layers = _generate_seed_network()
    
    # Verify seed
    if not _verify_network_numpy(seed_layers):
        # Should not happen, but fallback
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    best_network = [list(layer) for layer in seed_layers]
    best_count = _count_comparators(best_network)
    
    # 2. Optimization Loop
    # Strategy: Greedy removal + Random Restart/Mutation
    
    # Try to remove comparators greedily from the seed
    current_network = [list(layer) for layer in seed_layers]
    current_count = best_count
    
    # Greedy Phase
    improved = True
    while improved and (time.time() - start_time) < TIME_LIMIT - 5:
        improved = False
        # Iterate through all comparators and try to remove them
        # We need to iterate in a way that we can revert if it fails.
        
        # Create a flat list of (layer_idx, comp_idx)
        # But removing one changes indices.
        
        # Instead, loop through layers, then through comparators in reverse order
        for l_idx in range(len(current_network) - 1, -1, -1):
            if not current_network[l_idx]:
                current_network.pop(l_idx)
                continue
            for c_idx in range(len(current_network[l_idx]) - 1, -1, -1):
                if (time.time() - start_time) > TIME_LIMIT - 5:
                    break
                
                # Temporarily remove
                comp_to_remove = current_network[l_idx].pop(c_idx)
                
                # Verify
                if _verify_network_numpy(current_network):
                    current_count -= 1
                    improved = True
                    # Keep removed
                    # Update best if better
                    if current_count < best_count:
                        best_count = current_count
                        best_network = [list(layer) for layer in current_network]
                else:
                    # Revert
                    current_network[l_idx].insert(c_idx, comp_to_remove)
                    # If we can't remove this one, try the next one
                    # But maybe removing a *different* one first would allow this one?
                    # Greedy is local.
            
            if (time.time() - start_time) > TIME_LIMIT - 5:
                break
    
    # If we have time, try some random mutations to escape local minima
    if (time.time() - start_time) < TIME_LIMIT - 5:
        # Simple local search
        # Take best_network, try mutations, if valid and better, keep.
        
        for _ in range(100): # Max 100 iterations or until time out
            if (time.time() - start_time) > TIME_LIMIT - 5:
                break
            
            candidate = _mutate_network(best_network)
            cand_count = _count_comparators(candidate)
            
            # Only consider if it's potentially better or equal (to explore)
            # If it's worse, it might be a step towards a better solution, 
            # but with 60s limit, we should be conservative.
            # Let's only accept if count is lower.
            # Or, accept if count is lower OR (count is same and depth is lower? No, score is count).
            
            if cand_count < best_count:
                if _verify_network_numpy(candidate):
                    best_count = cand_count
                    best_network = candidate
            # We could also do a "hill climbing" where we accept equal cost if it changes structure?
            # Probably not worth the verification cost.
            
            # Also, try to add a comparator to see if it allows removing two others?
            # That's complex.
            
            # Let's also try a "swap" mutation specifically on the best network
            # The _mutate_network function handles adding/removing/swapping.
            # We should bias towards removals if we are close to optimal?
            # The current _mutate_network is 50% add, 30% remove, 20% swap.
            # If we are at 136, we want to remove.
            # Let's force removals more often in the mutation function?
            # Or just rely on the greedy phase which is strong.
            
            # Let's add a specific "try to remove" loop for the candidate as well?
            # No, that's too slow.
            
            # Let's just run the mutation.
            pass

    # Final Verification
    if not _verify_network_numpy(best_network):
        # Fallback to seed if something went wrong
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    # Convert to list of tuples
    comparators = []
    for layer in best_network:
        for comp in layer:
            comparators.append(comp)
            
    return comparators

# EVOLVE-BLOCK-END
