import time
import random
import sys

N = 17
DEADLINE_FRACTION = 0.90

def _build_seed():
    """
    Construct the Odd-Even Transposition Network for N wires.
    This is the standard seed with N^2 / 2 comparators (approx).
    For N=17, it has 17 * 17 = 289? No, it has N * (N-1) / 2 * 2? 
    Actually, Odd-Even Transposition has N phases.
    Phase 0: (0,1), (2,3), ...
    Phase 1: (1,2), (3,4), ...
    Total comparators = N * ceil(N/2) roughly.
    Let's just generate it correctly.
    """
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_by_simulation(net):
    """
    Fast verification using Python big integers (bitsets).
    Returns True if the network sorts all inputs.
    """
    # Pre-compute initial wires: wire i has bit k set if input k has bit i set.
    # We can compute this on the fly or cache it. Caching is better for repeated calls.
    # But for simplicity in this "different strategy" implementation, we'll use a 
    # slightly optimized simulation. 
    # Actually, the prompt says "Python standard library, NumPy and SciPy are available".
    # NumPy might be faster for the simulation if we vectorize? 
    # But 131072 inputs * 17 bits is small enough for Python ints.
    
    # Let's use the big int trick from the seed program but optimized.
    # Initial state:
    # wire[i] is an integer where bit k is 1 if the k-th input has a 1 in position i.
    # We need to check if after sorting, wire[i] matches the expected pattern.
    
    # Expected pattern for wire i:
    # The sorted output has the largest values at the end.
    # Input k (0..2^N-1) has popcount(k) ones.
    # After sorting, the wire at position i (0-indexed from smallest) should contain
    # the value from the input with the (i+1)-th smallest popcount?
    # No, the Zero-One Principle says we just need to check boolean inputs.
    # For a specific boolean input vector x, the sorted output y is the permutation of x
    # such that y is non-decreasing.
    # Since x is boolean, y is 0...0 1...1.
    # The number of 1s is popcount(x).
    # So wire i (0-indexed) is 1 iff i >= N - popcount(x).
    
    # Let's precompute the expected big ints.
    # This is a one-time cost.
    expected_wires = [0] * N
    for k in range(1 << N):
        pc = k.bit_count()
        # For this input k, wire i is 1 if i >= N - pc
        threshold = N - pc
        for i in range(threshold, N):
            expected_wires[i] |= (1 << k)
            
    # Initial wires
    initial_wires = [0] * N
    for k in range(1 << N):
        for i in range(N):
            if (k >> i) & 1:
                initial_wires[i] |= (1 << k)
                
    wires = initial_wires[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # Swap bits where wi=1 and wj=0
        # S = wi & ~wj
        # wires[i] = wi ^ S
        # wires[j] = wj ^ S
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    for i in range(N):
        if wires[i] != expected_wires[i]:
            return False
            
    return True

def _symbolic_prune(net):
    """
    Uses symbolic interval analysis to remove redundant comparators.
    
    State:
    For each wire w, we track:
    - min_rank[w]: The minimum possible rank (0 is smallest, N-1 is largest) 
                   of the value currently in wire w, assuming all previous 
                   comparators were applied.
    - max_rank[w]: The maximum possible rank.
    
    Initially, each wire w can hold any rank 0..N-1, so min=0, max=N-1.
    
    Comparator (i, j):
    - If max_rank[i] <= min_rank[j], then wire i is ALWAYS <= wire j.
      The comparator is redundant. We can remove it.
    - Otherwise, after compare-exchange:
      - The new value in i is min(old_i, old_j).
      - The new value in j is max(old_i, old_j).
      
      What are the new min/max ranks?
      This is tricky. The rank of the min of two values depends on the distribution.
      However, a known heuristic for sorting network optimization is:
      If we know that value in i is in [min_i, max_i] and value in j is in [min_j, max_j]:
      
      The min of two values from sets A and B has a rank bounded by:
      min_new_min = min(min_i, min_j)
      min_new_max = ? 
      
      Actually, a simpler and more robust symbolic check is often:
      If the *maximum* possible value in i is less than or equal to the *minimum* 
      possible value in j, the comparator does nothing.
      
      But we can also update the intervals.
      If we assume the values are distinct ranks 0..N-1:
      Let A be the set of possible ranks for i, B for j.
      New i gets min(a, b).
      New j gets max(a, b).
      
      The possible ranks for new i are { min(a,b) | a in A, b in B }.
      The possible ranks for new j are { max(a,b) | a in A, b in B }.
      
      Since we only track min/max, we can approximate:
      new_min_i = min(min_i, min_j)
      new_max_i = min(max_i, max_j)  # Is this correct?
      
      Counter-example: i can be 0 or 10. j can be 0 or 10.
      min(0,0)=0, min(0,10)=0, min(10,0)=0, min(10,10)=10.
      So new i can be 0 or 10.
      min(max_i, max_j) = min(10,10) = 10. Correct.
      min(min_i, min_j) = min(0,0) = 0. Correct.
      
      What about new j?
      max(0,0)=0, max(0,10)=10, max(10,0)=10, max(10,10)=10.
      New j can be 0 or 10.
      max(min_i, min_j) = max(0,0) = 0.
      max(max_i, max_j) = max(10,10) = 10.
      
      So the update rule is:
      new_min_i = min(min_i, min_j)
      new_max_i = min(max_i, max_j)
      new_min_j = max(min_i, min_j)
      new_max_j = max(max_i, max_j)
      
      Wait, is `new_max_i = min(max_i, max_j)` always valid?
      Let i be {0, 5}, j be {2, 3}.
      min(0,2)=0, min(0,3)=0, min(5,2)=2, min(5,3)=3.
      New i values: {0, 2, 3}.
      min(max_i, max_j) = min(5, 3) = 3. Correct.
      
      Let i be {0, 9}, j be {4, 5}.
      min(0,4)=0, min(0,5)=0, min(9,4)=4, min(9,5)=5.
      New i values: {0, 4, 5}.
      min(max_i, max_j) = min(9, 5) = 5. Correct.
      
      It seems the interval update [min(mins), min(maxs)] and [max(mins), max(maxs)] 
      is a valid over-approximation of the possible ranks.
      
      If this interval analysis says a comparator is redundant (max_i <= min_j), 
      it is DEFINITELY redundant for all inputs.
      
      If it says it's NOT redundant, it might still be redundant (false negative), 
      but we won't remove it. This is safe.
    """
    min_rank = [0] * N
    max_rank = [N - 1] * N
    
    new_net = []
    for i, j in net:
        # Check if redundant
        if max_rank[i] <= min_rank[j]:
            # Redundant, skip adding to new_net
            continue
        
        # Update intervals
        new_min_i = min(min_rank[i], min_rank[j])
        new_max_i = min(max_rank[i], max_rank[j])
        new_min_j = max(min_rank[i], min_rank[j])
        new_max_j = max(max_rank[i], max_rank[j])
        
        min_rank[i] = new_min_i
        max_rank[i] = new_max_i
        min_rank[j] = new_min_j
        max_rank[j] = new_max_j
        
        new_net.append((i, j))
        
    return new_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Start with seed
    current_net = _build_seed()
    
    # Verify seed just in case (should be valid)
    if not _verify_by_simulation(current_net):
        return current_net
        
    best_net = current_net[:]
    
    # Phase 1: Aggressive Symbolic Pruning
    # Repeat until no more removals are found by symbolic analysis
    while time.time() < deadline:
        old_len = len(current_net)
        current_net = _symbolic_prune(current_net)
        
        if len(current_net) == old_len:
            break
            
        if len(current_net) < len(best_net):
            best_net = current_net[:]
            
    # Phase 2: Local Search with Symbolic Pre-filtering
    # Now we are likely stuck at a local minimum for symbolic pruning.
    # We can try to remove comparators that are "almost" redundant or use 
    # full simulation to verify removals that symbolic analysis missed.
    
    # We will perform a steepest descent search.
    # At each step, we try to remove one comparator.
    # To speed up, we first use symbolic analysis to filter out candidates 
    # that are DEFINITELY not redundant (i.e., max_i > min_j).
    # Wait, if symbolic says it's not redundant, it MIGHT still be redundant.
    # So we should test the ones that symbolic says are "active".
    # Actually, if symbolic says max_i <= min_j, we would have removed it in Phase 1.
    # So in Phase 2, all remaining comparators have max_i > min_j.
    # They are potentially active.
    
    # We can try to remove them one by one and verify with simulation.
    # To make it faster, we can randomize the order.
    
    stuck_count = 0
    
    while time.time() < deadline:
        # Try to find a removable comparator
        n = len(current_net)
        if n == 0:
            break
            
        # Generate a random permutation of indices to try
        indices = list(range(n))
        random.shuffle(indices)
        
        found_removal = False
        
        for idx in indices:
            # Simulate removal of idx
            # Optimization: We can check if the network without idx is valid.
            # We can do this by running the simulation on the modified list.
            
            # To avoid creating a new list every time, we can just skip idx in the loop.
            # But _verify_by_simulation takes a list.
            # Let's create a view or just pass the list and the index to skip?
            # For simplicity, we'll just make a copy if we want to be safe, 
            # but slicing is O(N). N is small.
            
            # Actually, let's write a specialized verifier that takes a skip index.
            # Or just trust that the simulation is fast enough.
            # Simulation is ~0.1ms. 
            # If we have 70 comparators, and we try 70 candidates, that's 7ms.
            # We can do ~1000 such steps in 10s.
            
            # Let's try removing idx
            temp_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_by_simulation(temp_net):
                current_net = temp_net
                best_net = current_net[:]
                found_removal = True
                break # Restart the search with the new, shorter network
                
        if not found_removal:
            stuck_count += 1
            
            # If stuck, try a random swap of two comparators to escape local minimum
            # Swapping preserves the set of comparators but changes order.
            # This might make a previously non-removable comparator removable.
            
            if n > 1:
                i1 = random.randint(0, n - 1)
                i2 = random.randint(0, n - 1)
                if i1 != i2:
                    current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
                    
                    # Verify the swapped network
                    if _verify_by_simulation(current_net):
                        # Valid swap, keep it.
                        # Run symbolic pruning again? It might find new redundancies.
                        # Let's try a quick symbolic prune.
                        pruned = _symbolic_prune(current_net)
                        if len(pruned) < len(current_net):
                            current_net = pruned
                            best_net = current_net[:]
                    else:
                        # Swap broke it, revert
                        current_net[i1], current_net[i2] = current_net[i2], current_net[i1]
                        
            # If we've been stuck for a while, give up or try harder?
            # For now, if stuck_count > 50, we might be in a deep local minimum.
            # We can break or continue swapping.
            if stuck_count > 100:
                break

    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_by_simulation(net))
