import time
import random
import numpy as np

N = 18

def construct():
    # Deadline: 80 seconds to be safe within the 90s kill limit
    deadline = time.time() + 80.0
    
    # Generate all 2^18 boolean inputs
    # Shape: (262144, 18)
    # Using uint8 for memory efficiency and speed
    inputs = np.array([list(map(int, bin(i).zfill(N)[-N:])) for i in range(2**N)], dtype=np.uint8)
    
    def is_sorted_network(net):
        """
        Simulates the network on all boolean inputs and checks if output is sorted.
        Returns True if the network sorts all inputs.
        """
        if not net:
            # Empty network only sorts if inputs are already sorted, which they aren't for all
            # But for N=1, it is. For N=18, it's not.
            return False
        
        mat = inputs.copy()
        for i, j in net:
            # min to i, max to j
            mn = np.minimum(mat[:, i], mat[:, j])
            mx = np.maximum(mat[:, i], mat[:, j])
            mat[:, i] = mn
            mat[:, j] = mx
        
        # Check if all rows are sorted (non-decreasing)
        # mat[:, :-1] <= mat[:, 1:]
        return np.all(mat[:, :-1] <= mat[:, 1:])

    # Seed Network: Odd-Even Transposition
    # Phases: 0 to N-1
    # Phase k: compare (k%2, k%2+1), (k%2+2, k%2+3), ...
    oem_net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            oem_net.append((i, i + 1))
    
    best_net = oem_net[:]
    best_size = len(oem_net)
    
    # Greedy Removal Loop
    # We try to remove comparators one by one.
    # We restart the scan whenever a removal is successful to catch new redundancies.
    # We use random shuffling to explore different reduction paths.
    
    while time.time() < deadline:
        current_net = oem_net[:]
        
        # Multiple passes of greedy removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Create a test network without the comparator at idx
                test_net = current_net[:idx] + current_net[idx+1:]
                
                if is_sorted_network(test_net):
                    # Removal was successful
                    current_net = test_net
                    changed = True
                    break # Restart the inner while loop with a new shuffle
            
            # If no change in this full pass, we've reached a local minimum for this order
            # But since we shuffle, the next outer loop iteration might find more
            # Actually, if 'changed' is False, we break the inner while.
            # Then we check if this current_net is better than best_net.
            
        if len(current_net) < best_size:
            best_size = len(current_net)
            best_net = current_net[:]
            
        # Optional: If we are very fast, we could try to start from best_net and remove more?
        # No, best_net is a local minimum. Starting from OEM again with different shuffle is better.
        
    return best_net
