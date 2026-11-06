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

def _build_states(network, initial_masks):
    """
    Builds the list of states (masks after each comparator).
    states[0] is initial_masks.
    states[k] is the state after network[0:k]
    """
    states = [initial_masks]
    cur = initial_masks
    for i, j in network:
        mi = cur[i]
        mj = cur[j]
        new_cur = cur[:]
        new_cur[i] = mi & mj
        new_cur[j] = mi | mj
        cur = new_cur
        states.append(cur)
    return states

def _greedy_delete(network, initial_masks, deadline, max_passes=100):
    """
    Greedily deletes comparators from the network.
    Returns the reduced network.
    """
    net = list(network)
    if not net:
        return net
    
    # Build initial states
    states = _build_states(net, initial_masks)
    
    # Check if current net is valid
    if not _is_sorted(states[-1]):
        return net # Should not happen if input is valid, but safety check
    
    changed = True
    passes = 0
    while changed and passes < max_passes and time.time() < deadline:
        changed = False
        passes += 1
        # Iterate backwards
        for k in range(len(net) - 1, -1, -1):
            if time.time() >= deadline:
                return net
            
            # Check if we can remove net[k]
            # State before net[k] is states[k]
            # Tail is net[k+1:]
            
            if k == len(net) - 1:
                # If it's the last one, just check if state before it is sorted
                if _is_sorted(states[k]):
                    net.pop(k)
                    states.pop()
                    changed = True
                    break # Restart loop to handle index shifts
            else:
                # Simulate tail net[k+1:] starting from states[k]
                # Optimization: if tail is short, it's fast. If long, it's slow.
                # But we have to do it to be sure.
                tail = net[k+1:]
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    net.pop(k)
                    # Rebuild states for the new network
                    # This is O(len(net)^2) if done naively, but len is small (~100)
                    # and bit ops are fast.
                    states = _build_states(net, initial_masks)
                    changed = True
                    break
        
    return net

def construct():
    start_time = time.time()
    # Safety margin: 85s deadline to avoid 90s kill.
    deadline = start_time + 85.0
    
    # 1. Generate initial network: Odd-Even Transposition
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # 2. Pre-compute initial masks
    initial_masks = _build_initial_masks()
    
    # 3. Initial Greedy Deletion
    # This should reduce 136 comparators significantly.
    network = _greedy_delete(network, initial_masks, deadline)
    
    best_network = list(network)
    best_len = len(best_network)
    
    # 4. Local Search: Insert-Delete Strategy
    # Try to insert a random comparator, then greedily delete.
    # If the result is smaller than best, update best.
    
    # To speed up, we can limit the number of insertions per "restart" or just loop until time.
    # We also want to try different random seeds/strategies.
    
    # Pre-compute all possible comparators for random selection
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # We will do a simple loop.
    # To avoid getting stuck in a local minimum, we might want to occasionally
    # restart from a slightly perturbed version of best_network or just keep going.
    
    # Optimization: Keep states of best_network? No, best_network changes.
    # We just rebuild states for each candidate.
    
    iteration = 0
    while time.time() < deadline - 2.0: # Leave 2s for final check
        iteration += 1
        
        # 1. Start with best_network
        current_net = list(best_network)
        
        # 2. Insert a random comparator
        # Strategy: 
        # - Sometimes insert 1 random comparator.
        # - Sometimes insert 2 random comparators (to escape deeper minima).
        # - Sometimes insert a comparator that is "near" existing ones?
        
        num_inserts = 1
        if random.random() < 0.2:
            num_inserts = 2
        elif random.random() < 0.1:
            num_inserts = 3
            
        for _ in range(num_inserts):
            pos = random.randint(0, len(current_net))
            comp = random.choice(all_comparators)
            current_net.insert(pos, comp)
            
        # 3. Greedy Delete on current_net
        # We need to be careful with time.
        # _greedy_delete has its own deadline check.
        # We pass a slightly tighter deadline to ensure we don't overshoot the main loop.
        current_net = _greedy_delete(current_net, initial_masks, deadline - 0.5)
        
        # 4. Check if improved
        if len(current_net) < best_len:
            best_network = current_net
            best_len = len(best_network)
            
            # Optional: If we found a new best, we can do a quick "polish" 
            # or just continue the loop. The loop naturally continues.
            
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
        # Re-generate it to be safe
        fallback = []
        for phase in range(N):
            start = phase % 2
            for i in range(start, N - 1, 2):
                fallback.append((i, i + 1))
        return fallback

