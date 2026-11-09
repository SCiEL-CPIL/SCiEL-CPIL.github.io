import time
import random
import numpy as np

N = 18

def _odd_even_merge_sort(n):
    """
    Generate an Odd-Even-Merge sorting network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    def merge(n, offset=0):
        """
        Generate comparators to merge two sorted sequences of length n/2 
        (or n//2 and n-n//2) into one sorted sequence of length n.
        Assumes the two halves are already sorted.
        """
        if n <= 1:
            return []
        
        # Split n into n1 and n2
        n1 = n // 2
        n2 = n - n1
        
        # Recursively merge the interleaved parts?
        # Odd-Even-Merge step:
        # 1. Interleave the two halves: a0, b0, a1, b1, ...
        # 2. Compare (0,1), (2,3), ... (Even phase)
        # 3. Recursively merge the "even" subsequence (indices 0, 2, 4...) and "odd" subsequence (indices 1, 3, 5...)
        # 4. Compare (1,2), (3,4), ... (Odd phase)
        
        comps = []
        
        # 1. Interleave is implicit in the indexing for the next steps.
        # The inputs to this merge are assumed to be in wires:
        # First half: offset .. offset+n1-1
        # Second half: offset+n1 .. offset+n-1
        
        # We need to compare pairs (offset+2k, offset+2k+1) for the "Even" phase?
        # No, standard O-EM:
        # Let A be first half, B be second half.
        # Interleaved: A[0], B[0], A[1], B[1], ...
        # But in the wire array, they are just A[0], A[1], ..., B[0], B[1]...
        # So we need to physically move them? No, the network definition handles the logic.
        # The comparators are defined on the wire indices.
        
        # Standard construction:
        # Phase 1: Compare (offset + 2k, offset + 2k + 1) for k=0..n/2-1
        # This compares A[0] with B[0]? No.
        # If wires are [A0, A1, ..., B0, B1...], then:
        # (0,1) compares A0, A1. This is not interleaving.
        
        # Correct O-EM construction:
        # To merge two sorted lists of size m1, m2 into size n=m1+m2:
        # 1. Interleave them: C[0]=A[0], C[1]=B[0], C[2]=A[1], C[3]=B[1]...
        #    This requires a permutation network? No, we define the comparators on the *current* wire positions.
        #    If we assume the input to the merge is already interleaved, we can just compare.
        #    But the recursive sort produces sorted halves side-by-side.
        #    So we must first interleave.
        #    Interleaving is a permutation. Permutations can be done with comparators? No, permutations are not sorting networks in the sense of min/max, they are wires.
        #    Wait, a sorting network is a fixed set of comparators. We cannot permute wires dynamically.
        #    So, how is O-EM built as a static network?
        #    It uses the fact that you can compare non-adjacent elements.
        
        # Let's use the standard recursive definition for the comparators:
        # Merge(n, offset):
        #   if n==1: return []
        #   n1 = n//2, n2 = n-n1
        #   # The two halves are [offset, offset+n1-1] and [offset+n1, offset+n-1]
        #   # We want to merge them.
        #   # Step 1: Compare (offset+2k, offset+2k+1) for k=0 to n/2-1?
        #   # This compares elements within the first half? No.
        
        # Let's look at a known O-EM implementation.
        # It is easier to implement the **Bitonic** sort for powers of 2 and then adapt, 
        # or just use a known small network for N=18 if I can recall one.
        # 
        # Actually, a simpler structural change is to use a **Batcher Bitonic Sort** 
        # padded to 32, and then **prune** the network by removing comparators 
        # that only touch the "dummy" wires? 
        # No, as established, this is hard.
        
        # Let's stick to **Odd-Even Transposition** but start the search from a 
        # **randomized** or **shuffled** version? No, OET is fixed.
        
        # Let's use the **Batcher Bitonic Sort** for N=16 (53 comparators) 
        # and extend it to N=18?
        # 
        # Alternative: Use the **AKS** or just a good heuristic?
        # 
        # Let's implement **Bitonic Sort** for N=18 by padding to 32, 
        # generating the 32-wire network, and then **removing** any comparator 
        # (i,j) where i>=18 or j>=18? 
        # This results in a network on 18 wires. Is it valid?
        # If we remove comparators involving dummy wires, the dummy wires just sit there.
        # The comparators between valid wires (i<18, j<18) remain.
        # Does this sub-network sort the 18 inputs?
        # The 32-wire network sorts the 32 inputs (18 real + 14 dummy).
        # If we set the dummy wires to -infinity (or 0 if we use 0/1 logic and assume real inputs are 0/1, dummy 0), 
        # the real inputs will be sorted among themselves and pushed to the "end" (higher indices) or "start"?
        # If dummies are 0, and real inputs are 0/1, the 0s (dummies and real 0s) will be at the start.
        # The real 1s will be at the end.
        # The relative order of real inputs will be sorted.
        # So, if we take the 32-wire Bitonic network, keep only comparators (i,j) with i<18 and j<18, 
        # and assume inputs on wires 18..31 are 0, then the output on wires 0..17 will contain 
        # the sorted real inputs?
        # No, the 0s (dummies) will compete with the real 0s. 
        # The output will have 14 dummies (0s) and some real 0s at the front.
        # The real 1s will be at the back.
        # The real inputs will be sorted, but mixed with dummies.
        # We only care that the *real* inputs are sorted.
        # If the dummies are 0, and real inputs are 0 or 1:
        # The final state will have all 0s (dummy + real 0) in the first k positions, and real 1s in the last.
        # The real inputs are thus sorted (all 0s then all 1s).
        # So YES, this works!
        
        # So the strategy is:
        # 1. Generate Bitonic Sort Network for N=32.
        # 2. Filter comparators: keep only (i,j) where i < 18 and j < 18.
        # 3. Use this as the seed.
        # 4. Run fast local search.
        
        return comps

    # Implement Bitonic Sort for N=32
    def bitonic_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        n_half = n // 2
        # Sort first half and second half
        net1 = bitonic_sort(n_half)
        net2 = bitonic_sort(n_half)
        
        # Merge
        merge_net = bitonic_merge(n)
        
        # Combine
        # net1 is on wires 0..n_half-1
        # net2 is on wires n_half..n-1 (shifted)
        shifted_net2 = [(i + n_half, j + n_half) for i, j in net2]
        
        return net1 + shifted_net2 + merge_net

    def bitonic_merge(n):
        if n <= 1:
            return []
        
        comps = []
        # Bitonic merge step
        # Compare (i, i + n/2) for i=0..n/2-1
        # Then recursively merge the two halves
        n_half = n // 2
        
        # Phase 1: Compare (i, i + n_half)
        for i in range(n_half):
            comps.append((i, i + n_half))
            
        # Phase 2: Recursively merge
        # The first half (0..n_half-1) is now "sorted" in one direction?
        # In Bitonic Merge, after the first phase, the two halves are bitonic.
        # We need to merge them.
        # The standard recursion:
        # Merge(n) = 
        #   Compare (i, i+n/2)
        #   Merge(n/2) on first half
        #   Merge(n/2) on second half
        
        merge1 = bitonic_merge(n_half)
        merge2 = bitonic_merge(n_half)
        
        shifted_merge2 = [(i + n_half, j + n_half) for i, j in merge2]
        
        comps.extend(merge1)
        comps.extend(shifted_merge2)
        
        return comps

    # Generate for 32
    net32 = bitonic_sort(32)
    
    # Filter for 18 wires
    net18 = [(i, j) for i, j in net32 if i < 18 and j < 18]
    
    return net18

def _verify_network_fast(comparators):
    """
    Verify if the network sorts all 2^N Boolean inputs using NumPy.
    """
    n_inputs = 1 << N
    # Create input matrix: (2^N, N)
    # Each row is a binary representation of 0..2^N-1
    inputs = np.arange(n_inputs, dtype=np.uint32)
    # Unpack bits
    # We want state[k, i] = bit i of k
    # state = (inputs[:, None] >> np.arange(N)) & 1
    state = (inputs[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    
    # Apply comparators
    for i, j in comparators:
        # Extract columns
        col_i = state[:, i]
        col_j = state[:, j]
        
        # Compute min and max
        # min = col_i & col_j (if 0/1) ? No.
        # min(0,1)=0, min(1,0)=0, min(0,0)=0, min(1,1)=1 -> AND
        # max(0,1)=1, max(1,0)=1, max(0,0)=0, max(1,1)=1 -> OR
        new_i = col_i & col_j
        new_j = col_i | col_j
        
        state[:, i] = new_i
        state[:, j] = new_j
        
    # Check if sorted
    # A row is sorted if it is non-decreasing.
    # For 0/1, this means all 0s come before all 1s.
    # Check if there is any 1 followed by a 0.
    # diff = state[:, 1:] - state[:, :-1]
    # If any diff < 0, it's not sorted.
    # Since values are 0/1, diff can be -1, 0, 1.
    # We just need to check if any diff is -1.
    # Or simply: check if the sum of the first k elements is 0 for all k < count_ones?
    # Easier: check if `state` is equal to the sorted version.
    # Sorted version for a row with k ones is [0]*(N-k) + [1]*k.
    # We can compute the target for each row based on its popcount.
    
    popcounts = state.sum(axis=1)
    # Target: 0 if index < N - popcount, else 1
    indices = np.arange(N)
    # target[k, i] = 1 if i >= N - popcount[k] else 0
    # Condition: i + popcount[k] >= N
    targets = (indices[None, :] + popcounts[:, None]) >= N
    
    # Compare state and targets
    if not np.all(state == targets):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate baseline
    current_network = _odd_even_merge_sort(N)
    
    # Verify baseline
    if not _verify_network_fast(current_network):
        # Fallback to OET if something went wrong with the generator
        oet = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                oet.append((i, i + 1))
        current_network = oet
        if not _verify_network_fast(current_network):
            return current_network # Should not happen

    best_network = current_network
    best_count = len(current_network)

    # Local Search
    # Try to remove comparators
    # We'll do multiple passes until no improvement or time runs out.
    
    while time.time() < deadline:
        improved = False
        # Iterate in reverse to remove "later" comparators first
        # This is often effective.
        for i in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing comparator i
            trial_network = current_network[:i] + current_network[i+1:]
            
            if _verify_network_fast(trial_network):
                current_network = trial_network
                best_network = current_network
                best_count = len(current_network)
                improved = True
                # Do not decrement i, as the next element shifted down
                
        if not improved:
            # Local minimum reached.
            # Try a random restart or a more complex move?
            # For now, we can try to shuffle the order of comparators?
            # Shuffling doesn't change count, but might enable removals.
            # Let's try a random swap of two comparators.
            
            if len(current_network) < 2:
                break
                
            # Pick two random distinct indices
            idx1 = random.randint(0, len(current_network) - 1)
            idx2 = random.randint(0, len(current_network) - 1)
            if idx1 == idx2:
                continue
                
            # Swap them
            trial_network = current_network.copy()
            trial_network[idx1], trial_network[idx2] = trial_network[idx2], trial_network[idx1]
            
            if _verify_network_fast(trial_network):
                current_network = trial_network
                # Count is same, but structure changed.
                # Continue the loop to try removals again.
                # To avoid infinite loops of swaps, we should limit swaps or 
                # only swap if it leads to a removal?
                # Simpler: Just do removals. If stuck, break.
                # Given the time budget, getting stuck in a local minimum 
                # without removals is a risk.
                # Let's break out of the while loop if no removals were made 
                # in the last pass.
                pass 
            else:
                # If swap fails, we might be stuck.
                # Let's just break to save time for other potential strategies?
                # Or continue trying random swaps?
                # Let's limit the number of random swaps to avoid wasting time.
                # For now, break if no removals.
                break

    return best_network
