import time
import random

N = 18
# Bit-parallel constants
# We represent the state of all 2^N inputs as a list of N big integers.
# Bit 'm' in integer 'wire_i' is 1 if the m-th input has a 1 on wire i.

_TARGETS = None
_INITIALS = None

def _precompute_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is not None:
        return

    # Precompute popcounts for all masks
    # popcount[m] is the number of 1s in the binary representation of m
    popcounts = [0] * (1 << N)
    for m in range(1, 1 << N):
        popcounts[m] = popcounts[m >> 1] + (m & 1)
    
    # masks_by_popcount[k] has bit m set if popcount(m) == k
    masks_by_popcount = [0] * (N + 1)
    for m in range(1 << N):
        k = popcounts[m]
        masks_by_popcount[k] |= (1 << m)
        
    # Construct Targets
    # Target for wire i: The sorted output for input m has 1s at the end.
    # If input m has k ones, the sorted output has 1s at positions N-k to N-1.
    # So wire i is 1 iff N-k <= i  => k >= N - i.
    _TARGETS = [0] * N
    for i in range(N):
        threshold = N - i
        target_mask = 0
        for k in range(threshold, N + 1):
            target_mask |= masks_by_popcount[k]
        _TARGETS[i] = target_mask
        
    # Construct Initials
    # Initial for wire i: Bit m is set if the i-th bit of m is 1.
    _INITIALS = [0] * N
    for i in range(N):
        mask = 0
        # Optimization: The i-th bit is 1 for m in [2^i, 2^(i+1)-1], [2^(i+2), 2^(i+3)-1], etc.
        # But simple loop is fast enough for 2^18 * 18 ~ 4.7M ops, which is < 0.5s in Py.
        # Let's use a slightly faster bit trick if possible, or just loop.
        # Looping is safer and clear.
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        _INITIALS[i] = mask

def _verify_network(net):
    """
    Verifies the network using bit-parallel simulation.
    Returns True if the network sorts all inputs.
    """
    # Copy initials to avoid modifying global state
    states = list(_INITIALS)
    
    for i, j in net:
        si = states[i]
        sj = states[j]
        # min goes to i, max goes to j
        # For bits: min(a,b) = a & b, max(a,b) = a | b
        states[i] = si & sj
        states[j] = si | sj
        
    # Check against targets
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _greedy_prune(net, deadline):
    """
    Deterministically remove redundant comparators from the end to the start.
    This is very effective because later comparators are often redundant 
    given the work done by earlier ones.
    """
    net = list(net)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Scan from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            # Try removing net[idx]
            candidate = net[:idx] + net[idx+1:]
            if _verify_network(candidate):
                net = candidate
                improved = True
                # Restart scan from end since indices shifted
                break
    return net

def _local_search(net, deadline):
    """
    Perform local search to escape local minima.
    Uses random swaps and insertions, followed by greedy pruning.
    """
    best = list(net)
    best_size = len(best)
    rng = random.Random(42)
    
    # We do a limited number of iterations to stay within time limits
    # Each iteration involves a mutation, verification, and pruning.
    # Pruning is the expensive part (O(N^2) verifications in worst case, 
    # but usually much less as it breaks early).
    
    while time.time() < deadline:
        mutated = list(best)
        
        # Choose a mutation strategy
        action = rng.random()
        if action < 0.5:
            # Swap two comparators
            if len(mutated) > 1:
                i, j = rng.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        elif action < 0.8:
            # Insert a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(mutated))
            mutated.insert(pos, (i, j))
        else:
            # Delete a random comparator (already handled by prune, but 
            # deleting a specific one might allow a different prune path)
            if len(mutated) > 1:
                idx = rng.randint(0, len(mutated) - 1)
                mutated.pop(idx)
        
        # Verify the mutated network
        if not _verify_network(mutated):
            continue
            
        # If valid, try to prune it
        pruned = _greedy_prune(mutated, deadline)
        
        if len(pruned) < best_size:
            best = pruned
            best_size = len(best)
            
    return best

def construct():
    _precompute_constants()
    
    # Deadline: 90s limit, keep 5s safety margin
    deadline = time.time() + 85.0
    
    # Start with seed network
    net = _seed_network()
    
    # 1. Initial Greedy Pruning
    net = _greedy_prune(net, deadline)
    
    # 2. Local Search for further improvements
    net = _local_search(net, deadline)
    
    # 3. Final Pruning pass to ensure no redundant comparators remain
    net = _greedy_prune(net, deadline)
    
    return net

if __name__ == "__main__":
    # Quick self-test
    _precompute_constants()
    net = construct()
    print(f"Network size: {len(net)}")
    if _verify_network(net):
        print("Network is valid.")
    else:
        print("Network is INVALID.")
