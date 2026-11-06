import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072

def _build_initial_masks():
    """
    Builds bitmask representation for the 0-1 principle.
    mask[k] is an integer where bit m is set iff input m has a 1 on wire k.
    Input m is represented by its binary representation of m (0 to 2^N-1).
    """
    masks = [0] * N
    # Optimization: Instead of iterating all inputs and all wires (N * 2^N),
    # we can use bit manipulation or simple loops. 
    # For N=17, 17 * 131072 = 2.2M operations, which is fast enough in Python.
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks

def _simulate_tail(masks_before, tail_comparators):
    """
    Simulates the effect of 'tail_comparators' on 'masks_before'.
    Returns the final masks.
    """
    masks = masks_before[:]
    for i, j in tail_comparators:
        mi = masks[i]
        mj = masks[j]
        # Compare-exchange: min to i, max to j
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def _is_sorted(masks):
    """
    Checks if the output masks represent a sorted network.
    For a sorted network, if wire k has a 1, wire k+1 must also have a 1.
    So mask[k] & ~mask[k+1] must be 0.
    """
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True

def _is_valid_full(net):
    """
    Full validation of a network from scratch.
    """
    # Rebuild initial masks? No, we should pass them or have a global.
    # For efficiency in the main loop, we'll use incremental.
    # This is used for final verification or initialization.
    initial_masks = _build_initial_masks()
    masks = initial_masks
    for i, j in net:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return _is_sorted(masks)

def construct():
    start_time = time.time()
    # Safety margin: 50s deadline to avoid 60s kill.
    deadline = start_time + 50.0
    
    # 1. Generate initial network: Odd-Even Transposition
    # N phases. Phase p has comparators (p%2, p%2+1), (p%2+2, p%2+3), ...
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # Initial size is roughly N * (N-1)/2. For N=17, it's 17*8 = 136.
    
    # 2. Pre-compute initial masks
    initial_masks = _build_initial_masks()
    
    # 3. Pre-compute state after each comparator
    # states[k] = masks after applying network[0:k]
    states = [initial_masks]
    current_masks = initial_masks
    for comp in network:
        i, j = comp
        mi = current_masks[i]
        mj = current_masks[j]
        new_masks = current_masks[:]
        new_masks[i] = mi & mj
        new_masks[j] = mi | mj
        current_masks = new_masks
        states.append(current_masks)
        
    # Verify initial network is valid (it should be)
    if not _is_sorted(current_masks):
        # Fallback to a known valid smaller network if something went wrong?
        # Odd-even transposition is always valid.
        pass

    # 4. Greedy Deletion
    # We iterate through the network. For each comparator at index k, 
    # we check if removing it still results in a sorted network.
    # To check removal of comp[k], we look at states[k] (state before comp[k])
    # and simulate the tail network[k+1:].
    
    # To make this faster, we can iterate from end to start or start to end.
    # Removing from the end is generally easier to reason about for tail simulation?
    # Actually, removing comp[k] requires simulating network[k+1:].
    # If we remove from the end (k = len-1), tail is empty. Fast.
    # If we remove from the start (k = 0), tail is len-1. Slow.
    # So, iterate from end to start.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate backwards
        for k in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            # Check if we can remove network[k]
            # State before network[k] is states[k]
            # Tail is network[k+1:]
            
            # Optimization: If tail is empty, just check states[k]
            if k == len(network) - 1:
                if _is_sorted(states[k]):
                    network.pop(k)
                    # Rebuild states? 
                    # Since we only removed the last one, states[0..k] are still valid.
                    # states[k+1] is now the final state which is states[k].
                    # We can just truncate states.
                    states.pop()
                    changed = True
                    break # Restart loop to handle index shifts? Or continue?
                    # Actually, removing an element shifts indices.
                    # If we remove index k, network[k+1:] becomes network[k:].
                    # The states array is tied to indices.
                    # It is safer to break and restart the outer while loop 
                    # or rebuild states if we remove something not at the end.
                    # However, rebuilding states is O(N^2) bit ops, which is ~136*17*131072 ~ 300M ops?
                    # No, bit ops on big ints are fast.
                    # Let's just rebuild states if we remove anything other than the last one to be safe and simple.
                    # Or, even better: 
                    # If we remove index k, the new network is network[0:k] + network[k+1:].
                    # We can keep states[0:k+1] (which includes state before removed comp).
                    # Then we need to re-simulate the tail from state k.
                    
            else:
                # Simulate tail network[k+1:] starting from states[k]
                tail = network[k+1:]
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    # Remove network[k]
                    network.pop(k)
                    # Rebuild states for the new network
                    # This is expensive, so we break and let the outer loop handle it?
                    # Or we can update states incrementally.
                    # Let's just break and restart the while loop to rebuild states cleanly.
                    # This is a "break on first success" strategy for the inner loop.
                    changed = True
                    break
        
    # 5. Local Search: Try inserting and deleting to find better local minima
    # The greedy deletion above finds a locally minimal network under the "remove if possible" heuristic.
    # Sometimes, removing a comparator allows removing another one that was previously necessary, 
    # or adding a comparator allows removing two others.
    
    # Let's do a few rounds of "Try to insert a random comparator, then greedily delete".
    # This explores the neighborhood.
    
    best_network = list(network)
    best_len = len(network)
    
    # Pre-compute initial masks again for the search phase if we need to verify from scratch
    # But we can use the incremental method.
    
    # To keep it simple and fast, we will do a limited number of random restarts/insertions.
    # Each insertion adds 1 comparator. If we can delete 2 or more, we improve.
    
    for _ in range(50): # Limit iterations
        if time.time() >= deadline:
            break
        
        current_net = list(best_network)
        
        # Insert a random comparator
        # Random position
        pos = random.randint(0, len(current_net))
        # Random wires
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        current_net.insert(pos, (i, j))
        
        # Now try to greedily delete from this new network
        # We need to check validity.
        # We can use a quick full simulation to see if it's valid.
        # If valid, try to delete.
        
        # Build states for current_net
        states = [initial_masks]
        cur_masks = initial_masks
        for comp in current_net:
            x, y = comp
            mx = cur_masks[x]
            my = cur_masks[y]
            nm = cur_masks[:]
            nm[x] = mx & my
            nm[y] = mx | my
            cur_masks = nm
            states.append(cur_masks)
            
        if not _is_sorted(cur_masks):
            continue # Insertion broke validity, skip
            
        # Greedy delete on current_net
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for k in range(len(current_net) - 1, -1, -1):
                if time.time() >= deadline - 1:
                    break
                # Check if current_net[k] can be removed
                # State before is states[k]
                # Tail is current_net[k+1:]
                
                if k == len(current_net) - 1:
                    if _is_sorted(states[k]):
                        current_net.pop(k)
                        states.pop()
                        changed = True
                        break
                else:
                    tail = current_net[k+1:]
                    final_masks = _simulate_tail(states[k], tail)
                    if _is_sorted(final_masks):
                        current_net.pop(k)
                        # Rebuild states
                        states = [initial_masks]
                        cm = initial_masks
                        for comp in current_net:
                            x, y = comp
                            mx = cm[x]
                            my = cm[y]
                            nm = cm[:]
                            nm[x] = mx & my
                            nm[y] = mx | my
                            cm = nm
                            states.append(cm)
                        changed = True
                        break
                        
        if len(current_net) < best_len:
            best_network = current_net
            best_len = len(current_net)
            
    # Final verification
    # The best_network should be valid. 
    # Let's double check with a full simulation just in case.
    final_masks = initial_masks
    for i, j in best_network:
        mi = final_masks[i]
        mj = final_masks[j]
        final_masks[i] = mi & mj
        final_masks[j] = mi | mj
        
    if _is_sorted(final_masks):
        return best_network
    else:
        # Fallback to the original odd-even transposition if something went wrong
        return network
