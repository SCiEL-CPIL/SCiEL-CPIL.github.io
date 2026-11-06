import time
import random

N = 16

def construct():
    # Precompute input masks for zero-one principle verification
    # mask[i] is a big integer where bit k is 1 if input i is 1 in test case k
    masks = [0] * N
    for i in range(N):
        m = 0
        # Bit i of k is 1 for k in [2^i, 2^(i+1)-1], etc.
        # Loop is 16 * 65536 = 1M iterations, very fast.
        for k in range(1 << 16):
            if (k >> i) & 1:
                m |= 1 << k
        masks[i] = m

    def verify(network):
        # Copy masks
        curr = masks[:]
        for u, v in network:
            a = curr[u]
            b = curr[v]
            curr[u] = a & b
            curr[v] = a | b
        
        # Check if sorted: curr[i] subset curr[i+1]
        # i.e., curr[i] & ~curr[i+1] == 0
        for i in range(N - 1):
            if curr[i] & ~curr[i + 1]:
                return False
        return True

    def bitonic_sort_network(n, offset=0):
        # Returns a list of comparators for bitonic sort of n wires starting at offset
        # Standard bitonic sort construction
        # 1. Sort first half increasing, second half decreasing
        # 2. Bitonic merge
        
        def sort_sub(n, offset, increasing):
            if n == 1:
                return []
            if n == 2:
                if increasing:
                    return [(offset, offset + 1)]
                else:
                    # To sort decreasing (max at offset, min at offset+1)
                    # We need comparator (offset+1, offset) ? No, indices must be i<j.
                    # Comparator (u, v) with u<v puts min at u, max at v.
                    # So (offset, offset+1) puts min at offset, max at offset+1 (Increasing).
                    # To get Decreasing, we need max at offset, min at offset+1.
                    # We can achieve this by swapping the roles? No, the hardware is fixed.
                    # In a sorting network, we can't "reverse" a comparator.
                    # However, bitonic sort usually assumes we can compare in either direction?
                    # No, the problem statement says: "Each compare-exchange writes min to wire i and max to j."
                    # And "0 <= i < j < N".
                    # This implies the comparator is FIXED: min to lower index, max to higher index.
                    # This is a "0-1" or "increasing" comparator.
                    # 
                    # If we only have increasing comparators, can we build a bitonic sort?
                    # Bitonic sort relies on sorting one half increasing and the other decreasing.
                    # If we can only sort increasing, we cannot directly sort a half decreasing 
                    # using the same network structure unless we permute the wires?
                    # 
                    # Wait. The standard "Bitonic Sort" network for N=16 uses 80 comparators.
                    # Does it require decreasing comparators?
                    # Yes, typically. 
                    # BUT, if the comparators are fixed to (i, j) -> min(i), max(j) with i<j,
                    # we are restricted to "0-1" comparators.
                    # 
                    # Let's check the Odd-Even Transposition network.
                    # It uses (0,1), (2,3)... then (1,2), (3,4)...
                    # All are (i, i+1) with i < i+1.
                    # So it only uses increasing comparators.
                    # And it sorts correctly.
                    # 
                    # Can we construct Bitonic Sort with ONLY increasing comparators?
                    # If we want to sort a sub-network decreasing, we can just reverse the wire indices 
                    # in the sub-network?
                    # If we have a network that sorts wires [0..7] increasing, 
                    # and we want to sort wires [0..7] decreasing, 
                    # we can map wire 0->7, 1->6, ... 7->0.
                    # A comparator (u, v) in the increasing network becomes (7-u, 7-v) in the mapped network?
                    # If u < v, then 7-u > 7-v.
                    # So the pair becomes (7-v, 7-u) to maintain i<j?
                    # Let's trace:
                    # Original: (0, 1) -> min(0), max(1).
                    # Mapped: We want max at 0, min at 1 (for decreasing order 0>1).
                    # If we use comparator (1, 0)? Not allowed (i<j).
                    # If we use comparator (0, 1), we get min(0), max(1). This is increasing.
                    # So we CANNOT create a decreasing sort on the same wires using only (i,j) i<j comparators 
                    # if the network structure is rigid?
                    # 
                    # Actually, we CAN. We just need to permute the inputs/outputs.
                    # But the network is fixed on wires 0..15.
                    # 
                    # Let's look at the "Akers" 60-comparator network. It is known to exist.
                    # It likely uses only (i,j) i<j comparators.
                    # 
                    # If Bitonic Sort requires decreasing comparators, it might not be directly implementable 
                    # with the "min to i, max to j" constraint if i<j strictly enforces increasing order on that pair.
                    # 
                    # However, we can simulate a "decreasing" comparator (v, u) (where v>u, so min to v, max to u) 
                    # by using a sequence of comparators? No, that's not a single comparator.
                    # 
                    # Let's re-read carefully: "Each compare-exchange writes min to wire i and max to j."
                    # This is a fixed direction.
                    # 
                    # If I want to sort 2 wires decreasing (wire 0 > wire 1), I need max at 0, min at 1.
                    # A comparator (0, 1) gives min at 0, max at 1.
                    # So a single comparator (0, 1) CANNOT sort 2 wires decreasing.
                    # 
                    # Therefore, standard Bitonic Sort (which relies on sorting halves in opposite directions) 
                    # CANNOT be constructed directly using only (i, j) with i<j comparators 
                    # unless we permute the wires before/after?
                    # 
                    # But the network is a fixed list of pairs.
                    # 
                    # Wait, Odd-Even Transposition works. Why?
                    # It doesn't sort halves in opposite directions. It interleaves.
                    # 
                    # So, the "Bitonic Sort" inspiration might be a trap if it assumes bidirectional comparators.
                    # 
                    # Let's stick to the current approach but improve the search.
                    # The current approach starts with Odd-Even (120).
                    # We can try to find a better starting point.
                    # 
                    # Is there a known 60-comparator network we can hardcode?
                    # The task says "A verified 60-comparator construction is the reference".
                    # If I can find the list of 60 pairs, I can just return it.
                    # 
                    # I don't have the list hardcoded.
                    # 
                    # Alternative: Use the "Shrinkage" method.
                    # Start with Odd-Even (120).
                    # Try to remove comparators.
                    # The current code does this but is slow/random.
                    # 
                    # Let's optimize the removal search.
                    # 1. Greedy pass: Check every comparator, if removable, remove.
                    # 2. Repeat greedy pass until no more removals.
                    # 3. Random search: Try removing random comparators.
                    # 
                    # Also, we can try to "swap" comparators? No, just removal is the goal.
                    # 
                    # Let's implement the Greedy + Random search robustly.
                    pass

    # Generate Odd-Even Transposition Network (120 comparators)
    initial_network = []
    for p in range(N):
        if p % 2 == 0:
            for i in range(0, N - 1, 2):
                initial_network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                initial_network.append((i, i + 1))
    
    # Verify initial
    if not verify(initial_network):
        return initial_network
        
    best_network = initial_network[:]
    best_count = len(best_network)
    
    start_time = time.time()
    limit = 55.0
    
    curr_network = best_network[:]
    
    # 1. Greedy Removal Pass
    # Iterate through the network, try to remove each comparator.
    # If valid, remove it and restart the pass (since indices shift).
    # This is O(N^2 * VerifyCost) in worst case, but usually fast.
    
    def greedy_reduce(network):
        changed = True
        while changed:
            changed = False
            for i in range(len(network)):
                # Try to remove network[i]
                new_net = network[:i] + network[i+1:]
                if verify(new_net):
                    network = new_net
                    changed = True
                    break # Restart loop
        return network

    curr_network = greedy_reduce(curr_network)
    if len(curr_network) < best_count:
        best_network = curr_network[:]
        best_count = len(best_network)

    # 2. Random Search with Additions
    # If stuck, add a random comparator to explore other local minima.
    
    no_improve_count = 0
    max_no_improve = 5000 # Increased threshold for random steps
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() - start_time < limit:
        if not curr_network:
            break
            
        # Try to remove a random comparator
        idx = random.randint(0, len(curr_network) - 1)
        new_network = curr_network[:idx] + curr_network[idx+1:]
        
        if verify(new_network):
            curr_network = new_network
            if len(curr_network) < best_count:
                best_network = curr_network[:]
                best_count = len(best_network)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Add a random comparator to escape
            present = set(curr_network)
            candidates = [p for p in all_pairs if p not in present]
            if candidates:
                curr_network.append(random.choice(candidates))
            else:
                curr_network.append((0, 1))
                
            no_improve_count = 0
            max_no_improve = random.randint(1000, 2000)
            
    return best_network

if __name__ == "__main__":
    net = construct()
    print(len(net))
