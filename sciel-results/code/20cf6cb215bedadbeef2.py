"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    # Precompute all 2^N Boolean inputs as a (N, 2^N) boolean matrix
    # Each column is one input vector
    total = 1 << N
    # Use efficient generation: for bit position i, the pattern is periodic
    # bit i toggles every 2^i entries
    all_inputs = np.zeros((N, total), dtype=np.uint8)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        # Create pattern: 0 for half, 1 for half, repeated
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, total // period)
    
    # Precompute sorted check: for a column to be sorted, all 1s must come after all 0s
    # i.e., once we see a 0 after a 1, it's unsorted
    # Equivalent: the column must be non-decreasing
    # For 0-1 inputs: sorted means there's no i<j where x[i]=1 and x[j]=0
    # We can check: for each wire i, if x[i]=1, then all wires j>i must also be 1
    # Simpler: compute cumulative sum from bottom; if any 1 appears above a 0...
    # Actually simplest: check if the column equals its sorted version
    # For 0-1: sorted means all 0s then all 1s. 
    # A column is sorted iff for all i, if x[i]=1 then x[i+1]=1 (for i < N-1)
    # i.e., no pattern (1,0) in adjacent wires
    
    def _verify(network):
        """Vectorized verification: returns True if network sorts all inputs."""
        # Check time
        if (time.time() - start_time) > deadline:
            return False  # timeout, assume invalid to be safe
        
        x = all_inputs.copy()
        for i, j in network:
            # Compare-exchange: min to i, max to j
            # swap where x[i] > x[j], i.e., x[i]=1 and x[j]=0
            mask = (x[i] > x[j])
            if np.any(mask):
                x[i][mask] = 0
                x[j][mask] = 1
        
        # Check if all columns are sorted (non-decreasing)
        # For 0-1: no adjacent (1,0) pair
        for i in range(N - 1):
            if np.any((x[i] == 1) & (x[i+1] == 0)):
                return False
        return True
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network works
    assert _verify(network), "Initial network is invalid"
    
    # Local search: iteratively try removing comparators
    # Multiple passes to find more removable comparators
    max_passes = 50
    for _pass in range(max_passes):
        if (time.time() - start_time) > deadline - 5:
            break
        
        improved = False
        # Try removing from the end first (often less critical), then all positions
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            if _verify(test_network):
                network = test_network
                improved = True
        
        if not improved:
            break
    
    # Second phase: try adding then removing (hill-climbing with moves)
    if (time.time() - start_time) < deadline - 10:
        existing_set = set(network)
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        for _round in range(20):
            if (time.time() - start_time) > deadline - 10:
                break
            
            found_improvement = False
            for add_pair in all_pairs:
                if add_pair in existing_set:
                    continue
                if (time.time() - start_time) > deadline - 10:
                    break
                
                # Add the new comparator, then try to remove others
                test_network = network + [add_pair]
                # Try removing each comparator (except the one we just added)
                for idx in range(len(test_network) - 1, -1, -1):
                    if idx == len(test_network) - 1:
                        continue  # skip the one we just added
                    if (time.time() - start_time) > deadline - 10:
                        break
                    
                    removed = test_network[:idx] + test_network[idx+1:]
                    if len(removed) < len(network) and _verify(removed):
                        network = removed
                        existing_set = set(network)
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                break
    
    return network
# EVOLVE-BLOCK-END
