"""Size-efficient sorting network on 18 wires via SAT-based synthesis and local search."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _verify_network(network):
    """
    Verify if the network sorts all 2^N Boolean inputs using the 0-1 principle.
    Uses NumPy vectorization for speed.
    Returns True if valid, False otherwise.
    """
    if len(network) == 0:
        return True
        
    n_comparators = len(network)
    if n_comparators == 0:
        return True
        
    # Pre-extract indices for speed
    comp_i = np.array([c[0] for c in network], dtype=np.int32)
    comp_j = np.array([c[1] for c in network], dtype=np.int32)
    
    total_inputs = 1 << N
    
    # Process in chunks to manage memory and potentially allow early exit
    # 2^18 is 262,144. 262,144 * 18 * 1 byte is ~4.7 MB. 
    # We can process all at once or in chunks. 
    # Let's try all at once first, it's simpler and often faster in NumPy if memory allows.
    # But to be safe against memory limits or cache issues, chunks are safer.
    # Let's use a chunk size that fits in L2/L3 cache well. 
    # 16384 inputs * 18 wires * 1 byte = ~288 KB. Very safe.
    # 65536 inputs * 18 wires * 1 byte = ~1.1 MB. Safe.
    
    chunk_size = 1 << 16  # 65536
    
    for start in range(0, total_inputs, chunk_size):
        end = min(start + chunk_size, total_inputs)
        masks = np.arange(start, end, dtype=np.int64)
        
        # Generate inputs for this chunk
        # Shape: (chunk_len, N)
        # We can use broadcasting
        # bits = (masks[:, None] >> np.arange(N)[None, :]) & 1
        # This creates a (chunk_len, N) array.
        inputs = (masks[:, None] >> np.arange(N)[None, :]) & 1
        inputs = inputs.astype(np.int8)
        
        # Apply network
        for k in range(n_comparators):
            i = comp_i[k]
            j = comp_j[k]
            # Compare
            # swap_mask = inputs[:, i] > inputs[:, j]
            # If we swap, we need to be careful with numpy in-place ops on views
            # temp = inputs[:, i].copy()
            # inputs[:, i] = np.where(swap_mask, inputs[:, j], inputs[:, i])
            # inputs[:, j] = np.where(swap_mask, temp, inputs[:, j])
            
            # Faster approach:
            # We only need to swap where inputs[i] > inputs[j].
            # Since values are 0/1, this means i=1, j=0.
            # So we set i=0 and j=1 where i was 1 and j was 0.
            
            # Let's stick to the standard swap logic which is robust.
            swap_mask = inputs[:, i] > inputs[:, j]
            if np.any(swap_mask):
                # Create a copy of the column i to use for j
                # We can't just swap views directly if they overlap, but they don't here.
                # However, numpy assignment with boolean indexing creates a copy internally.
                # temp = inputs[:, i][swap_mask]
                # inputs[:, i][swap_mask] = inputs[:, j][swap_mask]
                # inputs[:, j][swap_mask] = temp
                
                # This is slightly faster:
                # temp = np.take(inputs, i, axis=1)
                # np.take(inputs, j, axis=1) -> this is a view
                # Let's just use the temp variable approach, it's clear.
                temp = inputs[:, i][swap_mask].copy()
                inputs[:, i][swap_mask] = inputs[:, j][swap_mask]
                inputs[:, j][swap_mask] = temp
        
        # Check sorted
        # A 0-1 sequence is sorted if it is non-decreasing.
        # So we check if any element is greater than the next.
        # diff < 0 implies not sorted.
        if np.any(np.diff(inputs, axis=1) < 0):
            return False
            
    return True

def construct():
    """
    Construct a sorting network for N=18 wires.
    Strategy:
    1. Start with Odd-Even Transposition (153 comparators).
    2. Use a fast NumPy-based verifier.
    3. Perform greedy removal of comparators.
    4. Perform local search (mutations) to escape local minima if time permits.
    """
    start_time = time.time()
    # 90s limit. Let's aim for 85s to be safe.
    deadline = start_time + 85.0

    # --- Helper: Generate Seed ---
    def odd_even_transposition():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

    current_network = odd_even_transposition()
    best_network = current_network[:]
    best_size = len(current_network)
    
    # Verify seed (should be True, but good practice)
    if not _verify_network(current_network):
        # Fallback: return the seed anyway, as it should be valid
        return current_network

    # --- Phase 1: Greedy Removal ---
    # We try to remove comparators one by one.
    # We restart the scan from the beginning after each successful removal to find new redundancies.
    # We use a time budget for this phase, say 50% of the total time.
    greedy_deadline = start_time + (deadline - start_time) * 0.5
    
    while time.time() < greedy_deadline:
        removed_any = False
        n_comps = len(current_network)
        
        # Randomize the order of removal attempts to avoid bias towards specific indices
        # Or just go in order? Random might escape local minima better in the removal phase.
        # But systematic is faster if we find a removal early.
        # Let's try random order for the indices to check.
        indices = list(range(n_comps))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= greedy_deadline:
                break
            
            # Try removing comparator at idx
            trial_network = current_network[:idx] + current_network[idx+1:]
            
            # Verify
            if _verify_network(trial_network):
                # Success
                current_network = trial_network
                best_network = current_network[:]
                best_size = len(current_network)
                removed_any = True
                break # Restart the scan
        
        if not removed_any:
            # No more single removals found. 
            # We could try double removals here, but that's expensive.
            # Let's move to local search.
            break

    # --- Phase 2: Local Search / Simulated Annealing ---
    # If we have time left, we can try to mutate the network to find a smaller one.
    # Mutations:
    # 1. Remove a random comparator (if valid, keep; if invalid, discard or accept with low prob?)
    #    Since we want to minimize size, we only accept valid removals.
    #    But sometimes we need to add a comparator to enable a future removal?
    #    That's more complex. Let's stick to simple hill climbing with random restarts or swaps.
    #    
    #    Actually, "swapping" two comparators doesn't change size.
    #    "Replacing" a comparator (i,j) with (k,l) doesn't change size.
    #    The only way to reduce size is removal.
    #    
    #    If greedy removal is stuck, it means every single comparator is critical.
    #    To get a smaller network, we might need to restructure.
    #    This is hard with simple local search.
    #    
    #    However, we can try to "shuffle" the network (swap positions of comparators) 
    #    and then try removal again. Sometimes the order matters for redundancy.
    
    if time.time() < deadline:
        # Try a few iterations of:
        # 1. Randomly swap two comparators in the network
        # 2. Try greedy removal on the new network
        # 3. If we find a smaller network, update best.
        
        # We need to be careful not to spend too much time.
        # Let's allow a few such restarts.
        
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() >= deadline - 5.0: # Leave 5s buffer for final verification/return
                break
                
            # Create a mutated version of the current best (or current network?)
            # Let's mutate the current best network.
            mutated = best_network[:]
            n_mut = len(mutated)
            
            if n_mut < 2:
                break
                
            # Perform a few random swaps
            for _ in range(3):
                i, j = random.sample(range(n_mut), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
                
            # Try greedy removal on the mutated network
            # We do a limited greedy removal here to save time
            sub_deadline = time.time() + (deadline - time.time()) / (max_restarts - _ + 1)
            
            # Simple greedy removal loop
            removed_any = True
            while removed_any and time.time() < sub_deadline:
                removed_any = False
                indices = list(range(len(mutated)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() >= sub_deadline:
                        break
                    trial = mutated[:idx] + mutated[idx+1:]
                    if _verify_network(trial):
                        mutated = trial
                        removed_any = True
                        break
                        
            if len(mutated) < best_size:
                best_network = mutated
                best_size = len(mutated)

    return best_network

# EVOLVE-BLOCK-END
