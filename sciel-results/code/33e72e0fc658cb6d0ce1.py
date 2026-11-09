import time
import random

N = 17
M = 1 << N  # 131072

# EVOLVE-BLOCK-START

# Precompute constants at module level to save time inside construct()
# S_init[w] is the bitmask of all inputs k such that wire w is 1.
# This is constant for the problem.
_S_init = [0] * N
for _w in range(N):
    _mask = 0
    # Efficiently construct the mask for bit _w
    # Pattern is blocks of 2^_w zeros followed by 2^_w ones.
    # Total length M = 2^17.
    # We can build this using shifts or just a loop since it's done once.
    # Loop is ~1.1M iterations total for all wires. Acceptable.
    for _k in range(M):
        if (_k >> _w) & 1:
            _mask |= (1 << _k)
    _S_init[_w] = _mask

# Targets: T[w] is bitmask of inputs k such that in sorted output, wire w is 1.
# Sorted output for input k (with c ones) has 1s in wires N-c to N-1.
# So wire w is 1 iff c >= N-w.
_Targets = [0] * N
for _k in range(M):
    _c = bin(_k).count('1')
    _start_w = N - _c
    for _w in range(_start_w, N):
        _Targets[_w] |= (1 << _k)

def _verify_fast(network):
    """
    Verify if the network sorts all boolean inputs.
    Uses bit-parallel simulation.
    Returns True if valid.
    """
    # Copy initial state
    S = _S_init[:]
    
    # Apply comparators
    for i, j in network:
        si = S[i]
        sj = S[j]
        # min goes to i, max goes to j
        # min(si, sj) = si & sj
        # max(si, sj) = si | sj
        S[i] = si & sj
        S[j] = si | sj
        
    # Check against targets
    for w in range(N):
        if S[w] != _Targets[w]:
            return False
    return True

def _odd_even_transposition_network():
    """Standard odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s wall-clock

    # 1. Start with Odd-Even Transposition
    current_network = _odd_even_transposition_network()
    
    # Verify initial network (should be True)
    if not _verify_fast(current_network):
        # Fallback: return as is if something went wrong, though it shouldn't
        return current_network

    best_network = current_network[:]
    best_count = len(current_network)

    # 2. Greedy Removal (Deterministic)
    # Iterate through the network, trying to remove each comparator.
    # If removal keeps the network valid, keep it removed.
    # Repeat until no more removals are possible.
    
    # We use a "pass" approach. In each pass, we scan the list.
    # If we remove an item, the list shrinks, and we re-scan from the start 
    # or continue carefully. A common strategy is to scan left-to-right.
    # If we remove index i, the next element shifts to i, so we don't increment i.
    # If we keep index i, we increment i.
    
    # However, a single pass might miss opportunities where removing a later 
    # comparator allows removing an earlier one that was previously essential.
    # So we loop passes.
    
    while time.time() < deadline:
        improved = False
        i = 0
        while i < len(current_network):
            if time.time() > deadline:
                break
            
            # Try removing current_network[i]
            # Optimization: Instead of creating a new list, we can simulate 
            # removal by checking if the network WITHOUT this comparator is valid.
            # But _verify_fast takes the whole list. 
            # Creating a new list of size ~100 is cheap.
            
            # Alternative: We can maintain the state S and "undo" the comparator?
            # Undoing a comparator (min, max) is not straightforward without storing previous state.
            # So we just re-verify.
            
            # Construct candidate network without i
            # To avoid list copying overhead in the inner loop, we can pass a slice?
            # No, _verify_fast iterates.
            # Let's just create the list. It's small.
            candidate = current_network[:i] + current_network[i+1:]
            
            if _verify_fast(candidate):
                # Removal successful
                current_network = candidate
                improved = True
                # Do not increment i, check the next element which is now at i
            else:
                i += 1
                
        if not improved:
            # No more improvements possible in this pass
            break

    best_network = current_network[:]
    best_count = len(current_network)

    # 3. Local Search / Random Mutation
    # If we still have time, try random mutations to find further reductions.
    # Mutations:
    # - Remove a random comparator (if valid, accept)
    # - Swap two comparators (if valid, accept - though size doesn't change, might help future removals? 
    #   Actually, size doesn't change, so we only care if it enables a removal later. 
    #   But _verify_fast is the bottleneck. Swapping doesn't reduce count directly.
    #   However, sometimes the order matters for the greedy removal algorithm.
    #   But since we do greedy removal on the current order, shuffling might help find a different local minimum.
    # - Insert a comparator? No, we want to minimize count.
    # - "Move" a comparator: remove and insert elsewhere. Same as swap-ish.
    
    # A better strategy for local search in size reduction:
    # Try removing a random comparator. If invalid, try removing two random comparators?
    # Or just stick to single removals but randomize the order of checks.
    
    # Let's try a randomized greedy removal.
    # Shuffle the indices, then try to remove in that order.
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0:  # Leave 5s buffer
        # Create a random permutation of indices
        indices = list(range(len(current_network)))
        rng.shuffle(indices)
        
        improved_in_pass = False
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            
            # The index 'idx' refers to the position in the ORIGINAL list before this pass?
            # No, if we remove elements, the list changes.
            # This makes random permutation tricky if we modify the list in place.
            
            # Better: Create a copy, try to remove one random element from the copy.
            # If valid, accept the copy as the new current_network.
            # This is O(N) for copy + O(Verify).
            
            # Actually, let's just pick a random index from the CURRENT list.
            # If valid, remove it.
            # This is simpler.
            
            if len(current_network) == 0:
                break
                
            rand_idx = rng.randint(0, len(current_network) - 1)
            
            candidate = current_network[:rand_idx] + current_network[rand_idx+1:]
            
            if _verify_fast(candidate):
                current_network = candidate
                improved_in_pass = True
                # We can continue trying to remove more
            # else: do nothing, try another random index? 
            # If we just try one random index and fail, we should probably try another.
            # But if we loop until no improvement, we might get stuck in a local minimum 
            # where no single random removal works, but a specific one does.
            # The deterministic pass already tried all. 
            # So random removal is only useful if the deterministic pass was interrupted 
            # or if we want to explore different "basins".
            
        if not improved_in_pass:
            # No improvement in this random pass. 
            # Maybe try shuffling the network order and running deterministic removal again?
            # Shuffling might allow a different set of removals.
            
            # Let's try a "restart" with a shuffled network.
            # Only do this if we haven't found a better solution recently?
            # For now, just break if no improvement, to avoid wasting time.
            # Or, try a few shuffles.
            
            # Let's try one shuffle and deterministic removal.
            if len(current_network) > 0:
                rng.shuffle(current_network)
                # Run a quick deterministic pass on the shuffled network
                improved = False
                i = 0
                while i < len(current_network) and time.time() < deadline - 5.0:
                    candidate = current_network[:i] + current_network[i+1:]
                    if _verify_fast(candidate):
                        current_network = candidate
                        improved = True
                    else:
                        i += 1
                
                if not improved:
                    break
            else:
                break

    return current_network

# EVOLVE-BLOCK-END
