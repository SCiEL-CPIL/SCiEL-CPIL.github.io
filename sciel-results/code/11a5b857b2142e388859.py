import time
import random
import math
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.92

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

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    Returns a list of comparators (i, j) for n wires.
    Structure:
    1. Sort odd-indexed wires (0, 2, ...) recursively
    2. Sort even-indexed wires (1, 3, ...) recursively
    3. Merge comparators
    4. Sort even-indexed wires recursively
    5. Sort odd-indexed wires recursively
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

def _structured_local_search(net, deadline):
    """
    Local search that respects the Batcher structure.
    It attempts to remove comparators, prioritizing those that are likely redundant
    based on the structure, and uses random swaps within structural blocks to escape local minima.
    """
    current_net = net[:]
    current_size = len(current_net)
    best_net = current_net[:]
    best_size = current_size
    
    # Pre-compute structural groups for smarter swapping/mutation
    # For now, we use a hybrid: 
    # 1. Greedy deletion pass (fast)
    # 2. Randomized deletion with structural bias
    # 3. Swaps to escape local minima
    
    # Phase 1: Aggressive Greedy Deletion
    # Try to delete from the end of the network first, as later comparators are often redundant
    # if the earlier ones have already sorted the relevant subset.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate backwards
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_net = current_net[:idx] + current_net[idx+1:]
            if _verify_network(new_net):
                current_net = new_net
                current_size -= 1
                improved = True
                break # Restart loop to check from end again
        
    if current_size < best_size:
        best_net = current_net[:]
        best_size = current_size
        
    # Phase 2: Randomized Local Search with Structural Awareness
    # We will try random deletions and swaps.
    # To make it "structure aware", we bias swaps to occur between comparators
    # that are close in the network sequence or involve the same wires.
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    # List of all possible comparators for insertion (rarely needed, mostly for escape)
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline and T > T_min:
        T *= cooling
        
        move_type = random.random()
        
        if move_type < 0.7 and current_size > 0:
            # Deletion
            # Bias towards deleting comparators that are "far" from the start or end?
            # Actually, just random is fine, but let's try random index.
            idx = random.randint(0, current_size - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
            if _verify_network(new_net):
                current_net = new_net
                current_size -= 1
                if current_size < best_size:
                    best_net = current_net[:]
                    best_size = current_size
        elif move_type < 0.9 and current_size >= 2:
            # Swap
            # Pick two indices. Bias towards indices that are close to each other?
            # Or just random.
            i, j = random.sample(range(current_size), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
            if _verify_network(new_net):
                # Accept if valid. Since size is same, we accept to explore.
                # However, to prevent cycling, we could track history, but simple SA is okay.
                current_net = new_net
        else:
            # Insertion (Escape move)
            if current_size < 4096:
                comp = random.choice(all_comps)
                idx = random.randint(0, current_size)
                new_net = current_net[:]
                new_net.insert(idx, comp)
                
                if _verify_network(new_net):
                    # Size increased. Accept with prob exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        current_net = new_net
                        current_size += 1
                        
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Start with Batcher Odd-Merge, which is structurally superior to Odd-Even Transposition
    best_net = _build_batcher_odd_merge(N)
    
    if not _verify_network(best_net):
        # Fallback
        best_net = _build_seed()
        
    best_size = len(best_net)
    
    # Run structured local search
    # We can run multiple iterations if time permits, but the function above handles the loop internally.
    # To maximize exploration, we might want to run it multiple times with different random seeds
    # if the first run gets stuck. However, the internal loop runs until deadline.
    
    # Let's split the time: 
    # 50% for first run, 50% for second run with a perturbed start?
    # Or just one long run? 
    # Given the small N, one long run is usually better to converge.
    
    # However, to ensure we don't miss a better basin, we can do:
    # 1. Greedy deletion (fast)
    # 2. SA/Local Search on the result
    
    # The _structured_local_search does both.
    
    result_net = _structured_local_search(best_net, deadline)
    
    if len(result_net) < best_size:
        best_net = result_net
        best_size = len(result_net)
        
    # Final check
    if not _verify_network(best_net):
        # Should not happen, but safety net
        best_net = _build_seed()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
