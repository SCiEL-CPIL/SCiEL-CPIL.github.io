import numpy as np
import time

N = 17

def construct():
    # Start with Odd-Even Transposition
    # Phase 0: (0,1), (2,3), ...
    # Phase 1: (1,2), (3,4), ...
    # ...
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Pre-generate inputs for verification
    # 2^17 inputs
    # We will create a function that verifies a given network
    # To speed up, we pass the input array
    
    all_inputs = (np.arange(2**N)[:, None] >> np.arange(N)[None, :]) & 1
    
    deadline = time.time() + 50.0
    
    def verify(net):
        # Make a copy to avoid modifying the original input
        # But we can just simulate in place on a copy
        # To save time, we can use a buffer
        # However, 131072 * 17 is ~2.2M integers. Copying is fast.
        state = all_inputs.copy()
        
        for i, j in net:
            a = state[:, i]
            b = state[:, j]
            # min to i, max to j
            # np.minimum and np.maximum
            min_ab = np.minimum(a, b)
            max_ab = np.maximum(a, b)
            state[:, i] = min_ab
            state[:, j] = max_ab
            
        # Check if sorted
        # A sequence is sorted if all diffs are >= 0
        # For boolean, sorted means 0s then 1s
        # So we just need to check if it's of the form 0...01...1
        # This is equivalent to: once a 1 is seen, no 0 can follow.
        # Or simply: np.all(np.diff(state, axis=1) >= 0)
        
        # Faster check for boolean:
        # Find the first 1. All before must be 0. All after must be 1.
        # Or just use diff.
        return np.all(np.diff(state, axis=1) >= 0)

    # Optimization
    # We will use a list for network to allow easy modification
    # But list of tuples is fine.
    
    # To speed up verify, we can try to break early?
    # No, we need to process all comparators.
    
    # Heuristic: Removal
    # We iterate through the network.
    # If removing a comparator keeps the network valid, we remove it.
    # We restart the scan from the beginning after a removal.
    
    # To avoid infinite loops or slow progress, we track if we made progress in a full pass.
    
    while time.time() < deadline:
        changed = False
        # Try removal
        for i in range(len(network)):
            if time.time() > deadline:
                break
            # Create new network without i
            # To avoid O(N) copy each time, we can just simulate skipping it?
            # But verify() takes a list.
            # We can create a view or just slice.
            # Slicing a list of 100 elements is fast.
            new_net = network[:i] + network[i+1:]
            
            if verify(new_net):
                network = new_net
                changed = True
                break # Restart loop
        
        if not changed:
            # Try swapping disjoint adjacent comparators
            # This is a heuristic to escape local minima
            swapped = False
            for i in range(len(network) - 1):
                if time.time() > deadline:
                    break
                c1 = network[i]
                c2 = network[i+1]
                # Check if disjoint
                if c1[0] != c2[0] and c1[0] != c2[1] and c1[1] != c2[0] and c1[1] != c2[1]:
                    # Swap them
                    network[i], network[i+1] = c2, c1
                    swapped = True
                    break
            
            if not swapped:
                # No removal, no swap possible. Local minimum.
                # We could try a different base or give up.
                # Given the time, we just return what we have.
                break
                
    return network
