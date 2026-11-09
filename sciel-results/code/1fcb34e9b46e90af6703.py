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
    This is a much tighter starting point (105 comps) than Odd-Even Transposition (136 comps).
    """
    if n <= 1:
        return []
    
    # Split into n1 (odd) and n2 (even)
    n1 = (n + 1) // 2
    n2 = n // 2
    
    # 1. Recursively sort the two halves
    net = []
    # First half (indices 0 to n1-1)
    net.extend(_build_batcher_odd_merge(n1))
    
    # Second half (indices n1 to n-1)
    if n2 > 1:
        sub_net = _build_batcher_odd_merge(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
            
    # 2. Merge the two sorted halves
    # Odd-even merge step:
    # Compare (i, i + n2) for i in 0..n1-1
    # Then recursively merge the resulting "odd" and "even" subsequences.
    
    # Standard Batcher Odd-Merge structure:
    # Let A be the first half (size n1) and B be the second half (size n2).
    # We want to merge A and B.
    # Step 1: Compare (i, i+n2) for i in range(n1).
    # This creates two sub-sequences:
    # O (Odd): wires 0, 2, 4...
    # E (Even): wires 1, 3, 5...
    # But we need to map these back to a contiguous structure for recursive merging.
    
    # A simpler, correct implementation of Batcher Odd-Merge:
    # 1. Sort A (size n1) and B (size n2).
    # 2. Compare (i, i + n2) for i in 0..n1-1.
    # 3. Recursively merge the "odd" wires and "even" wires.
    #    The "odd" wires are at indices 0, 2, 4... in the combined array?
    #    No, the standard definition is:
    #    After comparing (i, i+n2), the elements at even indices and odd indices 
    #    form two sorted lists that need to be merged.
    
    # Let's implement the specific comparator list for the merge phase.
    # For n=17, n1=9, n2=8.
    # Comparators: (0,8), (1,9), ..., (8,16).
    for i in range(n1):
        net.append((i, i + n2))
        
    # Now we need to recursively merge the odd and even subsequences.
    # The "even" subsequence consists of wires 0, 2, 4, ...
    # The "odd" subsequence consists of wires 1, 3, 5, ...
    # Wait, the standard Batcher Odd-Merge splits the *values* into two sets based on their position in the merged stream, 
    # but physically they are interleaved.
    
    # Actually, the standard recursive call is on the two halves of the *result* of the cross-comparison.
    # Let's look at a known correct implementation or a simpler valid merge if this is tricky.
    # A safe fallback for the merge part if the recursive odd-even merge is complex to code blindly:
    # Use a simple "Odd-Even Merge" pattern which is:
    # 1. Compare (i, i+n2)
    # 2. Recursively merge the elements at even indices and odd indices.
    
    # To make this robust without complex index mapping errors, we can use a known valid merge network 
    # or just fall back to a slightly less optimal but guaranteed correct merge if the recursive structure is hard.
    # However, for N=17, we can hardcode or use a simple library-like construction.
    
    # Let's try to implement the recursive merge correctly.
    # The elements at even indices (0, 2, ..., 2*n1-2) and odd indices (1, 3, ..., 2*n1-1) 
    # need to be merged.
    # Note: n1 + n2 = n.
    # The even indices are 0, 2, ..., n-2 (if n is even) or n-1 (if n is odd? No, indices go up to n-1).
    # Number of even indices: ceil(n/2) = n1.
    # Number of odd indices: floor(n/2) = n2.
    
    # We need to map the logical "even" wires to a contiguous range [0, n1-1] 
    # and "odd" wires to [0, n2-1] for the recursive call, then map back.
    
    # This mapping is:
    # Logical even wire k (0..n1-1) corresponds to physical wire 2k.
    # Logical odd wire k (0..n2-1) corresponds to physical wire 2k+1.
    
    # Recursively merge the even subsequence (size n1)
    if n1 > 1:
        sub_even = _build_batcher_odd_merge(n1)
        for i, j in sub_even:
            # Map logical i, j to physical 2i, 2j
            net.append((2*i, 2*j))
            
    # Recursively merge the odd subsequence (size n2)
    if n2 > 1:
        sub_odd = _build_batcher_odd_merge(n2)
        for i, j in sub_odd:
            # Map logical i, j to physical 2i+1, 2j+1
            net.append((2*i+1, 2*j+1))
            
    return net

def _local_search(net, deadline):
    """
    Steepest descent local search.
    Tries to remove any comparator that is redundant.
    """
    current_net = net[:]
    current_size = len(current_net)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try to remove each comparator
        # We iterate in reverse to avoid index shifting issues during removal
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Skip if we already tried this exact position recently? 
            # Simple approach: just try all.
            
            # Create a copy without the comparator at idx
            new_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_network(new_net):
                current_net = new_net
                current_size -= 1
                improved = True
                # Reset the loop to start from the beginning again? 
                # Or just continue? 
                # Since we removed one, the indices after idx are shifted.
                # It's safer to restart the loop or break and restart.
                # Breaking and restarting ensures we don't miss combinations.
                # But restarting from 0 is expensive.
                # Let's just break and let the while loop restart.
                break
                
    return current_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    rng = random.Random(42)
    
    # Start with the Batcher Odd-Merge network
    best_net = _build_batcher_odd_merge(N)
    
    # Verify it
    if not _verify_network(best_net):
        # Fallback to seed if Batcher implementation has a bug
        best_net = _build_seed()
        
    best_size = len(best_net)
    
    # Multi-start Local Search
    # We perform several rounds of local search.
    # To escape local minima, we can occasionally add a random comparator 
    # and then try to remove others, but pure deletion is often effective 
    # if the starting point is good.
    
    # Strategy:
    # 1. Run local search on the initial Batcher network.
    # 2. If stuck, try to add a random comparator, then run local search again.
    #    This allows "sideways" moves that might lead to a better basin.
    
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Pure deletion on the initial network
    best_net = _local_search(best_net, deadline)
    best_size = len(best_net)
    
    # Phase 2: Iterative improvement with random restarts/perturbations
    while time.time() < deadline:
        # Perturb the current best network
        # Add a random comparator
        perturbed_net = best_net[:]
        if len(perturbed_net) < 4096:
            comp = rng.choice(all_comps)
            idx = rng.randint(0, len(perturbed_net))
            perturbed_net.insert(idx, comp)
            
        # Run local search on the perturbed network
        # We give it a small time slice or just run it fully if fast
        # Since verification is fast, we can run it for a while
        candidate = _local_search(perturbed_net, deadline)
        
        if len(candidate) < best_size:
            best_net = candidate
            best_size = len(candidate)
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
