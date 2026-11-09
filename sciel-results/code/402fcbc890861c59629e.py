import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.90

# Pre-compute initial and expected wires for verification
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
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    """
    if n <= 1:
        return []
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    net = []
    # 1. Recursively sort the two halves
    net.extend(_build_batcher_odd_merge(n1))
    if n2 > 1:
        sub_net = _build_batcher_odd_merge(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
            
    # 2. Compare (i, i + n2) for i in 0..n1-1
    for i in range(n1):
        net.append((i, i + n2))
        
    # 3. Recursively merge the odd and even subsequences
    if n1 > 1:
        sub_even = _build_batcher_odd_merge(n1)
        for i, j in sub_even:
            net.append((2*i, 2*j))
            
    if n2 > 1:
        sub_odd = _build_batcher_odd_merge(n2)
        for i, j in sub_odd:
            net.append((2*i+1, 2*j+1))
            
    return net

def _simulate(net):
    """
    Simulate the network and return the final wire states.
    """
    wires = _INITIAL_WIRES[:]
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
    return wires

def _is_sorted(wires):
    """
    Check if the wires are in the expected sorted state.
    """
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    rng = random.Random(42)
    
    # Start with Batcher Odd-Merge
    best_net = _build_batcher_odd_merge(N)
    if not _verify_network(best_net):
        best_net = _build_seed()
    
    best_size = len(best_net)
    best_wires = _simulate(best_net)
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.995
    
    # Pre-compute all possible comparators for random selection
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    current_net = best_net[:]
    current_wires = best_wires[:]
    current_size = best_size
    T = T_start
    
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Choose a random move
        # 1. Delete a random comparator
        # 2. Insert a random comparator at a random position
        # 3. Swap two comparators
        
        move_type = rng.random()
        
        if move_type < 0.4 and len(current_net) > 0:
            # Delete
            idx = rng.randint(0, len(current_net) - 1)
            removed_comp = current_net[idx]
            new_net = current_net[:idx] + current_net[idx+1:]
            new_wires = _simulate(new_net)
            new_size = current_size - 1
            # Cost difference: if it was necessary, it breaks sorting (infinite cost penalty via acceptance prob)
            # We just check if it's still sorted. If yes, delta_size = -1. If no, we reject unless T is high enough to accept invalid? 
            # No, we must maintain validity. So we only accept if valid.
            # To make SA work on a constrained space, we only propose valid moves or use a penalty.
            # Let's stick to valid moves only for simplicity and robustness, but that limits exploration.
            # Better: Propose move, if invalid, reject. If valid, accept based on size change (always accept if smaller, reject if larger? No, that's greedy).
            # SA allows larger moves (worse size) to escape minima. But we cannot have invalid networks.
            # So the "energy" is just the size. We want to minimize size.
            # If the move keeps it sorted, delta_size = -1 (good). Accept with prob 1.
            # If the move breaks sorting, reject.
            # This is effectively greedy descent with random restarts if we never accept worse size.
            # To escape, we need to accept "worse" size? But size is the objective. 
            # Usually in sorting networks, you can't have a "worse" valid network that leads to a better one easily without breaking validity.
            # However, you CAN replace a comparator with another one that keeps it valid but changes structure.
            
            if _is_sorted(new_wires):
                current_net = new_net
                current_wires = new_wires
                current_size = new_size
                if current_size < best_size:
                    best_net = current_net[:]
                    best_wires = current_wires[:]
                    best_size = current_size
            else:
                # Rejected
                pass
                
        elif move_type < 0.8 and len(current_net) < 4096:
            # Insert
            comp = rng.choice(all_comps)
            idx = rng.randint(0, len(current_net))
            new_net = current_net[:idx] + [comp] + current_net[idx:]
            new_wires = _simulate(new_net)
            new_size = current_size + 1
            
            # We only accept if it leads to a path that eventually reduces size?
            # In pure SA on size, we would accept larger size with prob exp(-delta/T).
            # But larger size is always worse. Why would we accept it?
            # Because it might allow a subsequent deletion that wasn't possible before.
            # So yes, we accept larger size with some probability.
            
            delta_size = 1
            if rng.random() < (1.0 / (1.0 + T * 100)): # Heuristic acceptance
                current_net = new_net
                current_wires = new_wires
                current_size = new_size
            # Note: We don't update best_net here because size increased.
            
        else:
            # Swap
            if len(current_net) >= 2:
                idx1 = rng.randint(0, len(current_net) - 1)
                idx2 = rng.randint(0, len(current_net) - 1)
                if idx1 != idx2:
                    new_net = current_net[:]
                    new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
                    new_wires = _simulate(new_net)
                    new_size = current_size
                    
                    if _is_sorted(new_wires):
                        # Same size, so we accept to explore the space
                        current_net = new_net
                        current_wires = new_wires
                        # current_size unchanged
                    else:
                        pass
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Periodic check for best
        if current_size < best_size:
            best_net = current_net[:]
            best_wires = current_wires[:]
            best_size = current_size

    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
