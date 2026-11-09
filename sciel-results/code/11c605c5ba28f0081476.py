"""Size-efficient sorting networks on 17 wires."""
import time
import random
import sys

N = 17

# EVOLVE-BLOCK-START

# Precompute initial masks for fast verification
# Bit k of mask w is 1 if the k-th input (0 to 2^N-1) has a 1 on wire w.
_INITIAL_MASKS = None

def _get_initial_masks():
    global _INITIAL_MASKS
    if _INITIAL_MASKS is None:
        _INITIAL_MASKS = []
        for w in range(N):
            mask = 0
            # Efficiently construct the mask for wire w
            # Pattern: 2^w zeros, 2^w ones, repeated.
            # Total bits: 2^N
            # We can build this by iterating or using bit tricks.
            # Since N=17, 2^17 = 131072.
            # A simple loop is fast enough if done once.
            # But we can do it faster with shifts.
            # Let's stick to the loop for clarity and correctness, it's a one-time cost.
            # Actually, we can generate it recursively or with bit operations.
            # For w=0: 010101...
            # For w=1: 00110011...
            # mask = sum(1 << k for k in range(2**N) if (k >> w) & 1)
            # This is 131072 * 17 operations ~ 2M ops. ~0.1s in Python. Acceptable.
            for k in range(1 << N):
                if (k >> w) & 1:
                    mask |= (1 << k)
            _INITIAL_MASKS.append(mask)
    return _INITIAL_MASKS

def _verify_network_fast(network):
    """
    Verify if the network sorts all inputs using bitwise operations.
    """
    if not network:
        return N <= 1

    masks = _get_initial_masks()[:]
    
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj

    for w in range(N - 1):
        if masks[w] & ~masks[w + 1]:
            return False
    return True

def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _greedy_remove(network, deadline):
    """
    Attempt to remove comparators in a deterministic greedy fashion.
    Returns the modified network.
    """
    current = list(network)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            # Try removing idx
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                changed = True
                break
    # Forward pass
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(current)):
            if time.time() > deadline - 2:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                changed = True
                break
    return current

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    deadline = time.time() + 85  # Safety margin
    
    # 1. Seed
    current_network = _odd_even_transposition_network(N)
    best_network = list(current_network)
    best_size = len(current_network)
    
    # 2. Initial Greedy Pruning
    current_network = _greedy_remove(current_network, deadline)
    if len(current_network) < best_size:
        best_size = len(current_network)
        best_network = list(current_network)
        
    # 3. Randomized Local Search
    # We try to find better configurations by swapping or moving comparators.
    # If a move keeps the network valid, we accept it.
    # If it reduces size, we update best.
    # If it keeps size same, we might escape a local minimum for removal.
    
    random.seed(42)
    current = list(current_network)
    
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() < deadline - 5:
        if len(current) == 0:
            break
            
        # Choose a move
        # 70%: Try to remove a random comparator
        # 20%: Try to swap two random comparators
        # 10%: Try to move a random comparator to a random position
        
        move_type = random.random()
        candidate = None
        accepted = False
        size_changed = False
        
        if move_type < 0.7 and len(current) > 0:
            # Try removal
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                accepted = True
                size_changed = True
                
        elif move_type < 0.9 and len(current) > 1:
            # Try swap
            i1, i2 = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            if _verify_network_fast(candidate):
                current = candidate
                accepted = True
                
        else:
            # Try move (shift)
            if len(current) > 1:
                i1 = random.randint(0, len(current) - 1)
                i2 = random.randint(0, len(current) - 1)
                if i1 != i2:
                    candidate = list(current)
                    item = candidate.pop(i1)
                    candidate.insert(i2, item)
                    if _verify_network_fast(candidate):
                        current = candidate
                        accepted = True

        if accepted:
            if size_changed:
                if len(current) < best_size:
                    best_size = len(current)
                    best_network = list(current)
                no_improvement_count = 0
            else:
                # Size didn't change, but structure did.
                # We don't count this as "improvement" for the size, 
                # but it might lead to a removable comparator.
                # To prevent infinite loops on same size, we track "no size improvement"
                # but allow some wandering.
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    # Reset counter, maybe restart from best_network?
                    # Or just keep going.
                    no_improvement_count = 0
        else:
            # Move was invalid, do nothing
            pass

    # 4. Final Polish
    # Run greedy removal on the best network found so far
    final_candidate = _greedy_remove(best_network, deadline)
    if len(final_candidate) < best_size:
        best_network = final_candidate
        best_size = len(final_candidate)
        
    # Also try greedy removal on the current wandering network if it's smaller
    if len(current) < len(best_network):
        final_candidate = _greedy_remove(current, deadline)
        if len(final_candidate) < best_size:
            best_network = final_candidate
            best_size = len(final_candidate)

    return best_network

# EVOLVE-BLOCK-END
