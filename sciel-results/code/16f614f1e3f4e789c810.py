import time
import random
import math
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.90

_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    Returns True if the network sorts all inputs.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        if S:
            wires[i] = wi ^ S
            wires[j] = wj ^ S
        # If S is 0, the comparator is a no-op for this specific set of wires
        # We don't track no-ops here, just simulate.
            
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _is_redundant(net, idx):
    """
    Check if removing comparator at index idx results in a valid network.
    This is the core of the pruning strategy.
    """
    if idx < 0 or idx >= len(net):
        return False
    # Quick check: simulate the network without this comparator
    # To speed this up, we can pre-compute the state up to idx,
    # skip idx, and continue. But for simplicity and robustness,
    # we use the full verification but with a slight optimization:
    # We check if the comparator is a no-op in the current context.
    
    # A comparator (i, j) is redundant if removing it still allows sorting.
    # We just verify the new network.
    new_net = net[:idx] + net[idx+1:]
    return _verify_network(new_net)

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    Returns a list of comparators (i, j) for n wires.
    """
    if n <= 1:
        return []
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    net = []
    
    # 1. Sort wires 0, 2, 4... (indices 0..n1-1 in sub-problem)
    sub_odd = _build_batcher_odd_merge(n1)
    for i, j in sub_odd:
        net.append((2*i, 2*j))
        
    # 2. Sort wires 1, 3, 5... (indices 0..n2-1 in sub-problem)
    sub_even = _build_batcher_odd_merge(n2)
    for i, j in sub_even:
        net.append((2*i+1, 2*j+1))
        
    # 3. Merge: compare (i, i+n2) for i in 0..n1-1
    for i in range(n1):
        net.append((i, i + n2))
        
    # 4. Sort even-indexed wires again (indices 0..n1-1)
    sub_even2 = _build_batcher_odd_merge(n1)
    for i, j in sub_even2:
        net.append((2*i, 2*j))
        
    # 5. Sort odd-indexed wires again (indices 0..n2-1)
    sub_odd2 = _build_batcher_odd_merge(n2)
    for i, j in sub_odd2:
        net.append((2*i+1, 2*j+1))
        
    return net

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _structured_pruning_search(net, deadline):
    """
    Strategy:
    1. Aggressive Pruning: Iterate through the network, removing any comparator
       that is redundant. Repeat until no more can be removed.
    2. Local Search: Perform swaps of adjacent comparators to escape local minima
       and find new redundancies. Also try single deletions.
    """
    current_net = net[:]
    best_net = current_net[:]
    best_size = len(current_net)
    
    # Phase 1: Aggressive Pruning
    # We loop until no changes are made or time is up.
    # To make this efficient, we check indices in reverse order to avoid
    # index shifting issues, or we restart the scan after a deletion.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Scan from end to start to delete safely
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                return best_net
            
            # Check if current comparator is redundant
            if _is_redundant(current_net, idx):
                current_net.pop(idx)
                improved = True
                if len(current_net) < best_size:
                    best_net = current_net[:]
                    best_size = len(current_net)
                # Restart scan from beginning as indices have shifted
                break
                
    # Phase 2: Local Search with Swaps and Deletions
    # We use a limited simulated annealing or hill climbing.
    # Moves:
    # 1. Delete a random comparator (if redundant)
    # 2. Swap two adjacent comparators (if resulting network is valid)
    # 3. Swap two comparators within a small window (e.g., distance < 5)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        T *= cooling
        
        move_type = random.random()
        
        if move_type < 0.5 and len(current_net) > 0:
            # Try deletion
            idx = random.randint(0, len(current_net) - 1)
            if _is_redundant(current_net, idx):
                new_net = current_net[:idx] + current_net[idx+1:]
                current_net = new_net
                if len(current_net) < best_size:
                    best_net = current_net[:]
                    best_size = len(current_net)
        elif move_type < 0.9 and len(current_net) >= 2:
            # Try swap
            # Prefer adjacent swaps to preserve depth structure
            if random.random() < 0.7:
                # Adjacent swap
                i = random.randint(0, len(current_net) - 2)
                j = i + 1
            else:
                # Close swap
                i = random.randint(0, len(current_net) - 1)
                j = i + random.randint(-3, 3)
                if j < 0: j = 0
                if j >= len(current_net): j = len(current_net) - 1
                if i == j: continue
                
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
            if _verify_network(new_net):
                # Accept move
                current_net = new_net
                # Check if any new deletions are possible after swap
                # Quick pass for deletions
                for idx in range(len(current_net) - 1, -1, -1):
                    if _is_redundant(current_net, idx):
                        current_net.pop(idx)
                        if len(current_net) < best_size:
                            best_net = current_net[:]
                            best_size = len(current_net)
                        break # Restart deletion pass
        else:
            # Random restart or large perturbation?
            # Maybe insert a random comparator to escape local minimum?
            # This increases size, so we only accept if it leads to a smaller network later.
            # For now, just shuffle a small segment.
            if len(current_net) > 5:
                start = random.randint(0, len(current_net) - 5)
                end = start + 5
                segment = current_net[start:end]
                random.shuffle(segment)
                new_net = current_net[:start] + segment + current_net[end:]
                if _verify_network(new_net):
                    current_net = new_net
                    # Prune again
                    for idx in range(len(current_net) - 1, -1, -1):
                        if _is_redundant(current_net, idx):
                            current_net.pop(idx)
                            if len(current_net) < best_size:
                                best_net = current_net[:]
                                best_size = len(current_net)
                            break

    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # Start with Batcher Odd-Merge
    best_net = _build_batcher_odd_merge(N)
    
    if not _verify_network(best_net):
        best_net = _build_seed()
        
    best_size = len(best_net)
    
    # Run structured pruning search
    result_net = _structured_pruning_search(best_net, deadline)
    
    if len(result_net) < best_size:
        best_net = result_net
        best_size = len(result_net)
        
    # Final check
    if not _verify_network(best_net):
        best_net = _build_seed()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
