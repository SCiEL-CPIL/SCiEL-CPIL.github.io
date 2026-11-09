import numpy as np
import time

N = 18

def generate_bitonic_sort(n):
    """
    Generates a Batcher's Bitonic Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    def gen_asc(k):
        if k <= 1:
            return []
        h = k // 2
        # Sort first h in ascending
        net1 = gen_asc(h)
        # Sort second k-h in descending
        # To sort in descending, we generate the ascending sort for k-h 
        # and then "reverse" the output? 
        # Actually, we can just generate the network for k-h and then 
        # the merge step assumes the second half is descending.
        # The standard recursive call for the second half is gen_desc(k-h).
        net2 = gen_desc(k - h)
        # Offset net2 indices by h
        net2 = [(i + h, j + h) for (i, j) in net2]
        
        merge = bitonic_merge(k)
        return net1 + net2 + merge

    def gen_desc(k):
        if k <= 1:
            return []
        h = k // 2
        # Sort first h in descending
        net1 = gen_desc(h)
        # Sort second k-h in ascending
        net2 = gen_asc(k - h)
        net2 = [(i + h, j + h) for (i, j) in net2]
        
        merge = bitonic_merge(k)
        return net1 + net2 + merge

    def bitonic_merge(k):
        # Merges a bitonic sequence of length k
        # Assumes k is a power of 2? 
        # If k is not a power of 2, the standard bitonic merge 
        # still works if we define the stages correctly.
        # The number of stages is floor(log2(k)).
        # For stage s (from floor(log2(k)) down to 1):
        #   step = 2^s
        #   for i in range(0, k - step, step * 2)? 
        #   No, the standard loop is:
        #   for j in range(1, k, 2):
        #       if j % (2 * step) < step: ...
        
        # Let's use the standard implementation for power of 2 and pad if necessary?
        # Or just use the general formula.
        # For N=18, we split into 9 and 9.
        # 9 is not a power of 2.
        # Bitonic sort works for any N, but the merge step is defined for powers of 2.
        # If N is not a power of 2, we can still use the recursive split, 
        # but the merge step `bitonic_merge(k)` must be valid for any k.
        
        # A simple valid merge for any k:
        # We can just use a series of "butterfly" exchanges.
        # However, for the purpose of this task, if we get stuck with non-powers of 2,
        # we can pad to the next power of 2 (32) and then remove the extra wires?
        # No, that increases comparators.
        
        # Let's assume the standard Bitonic Sort implementation 
        # handles non-powers of 2 by just running the merge stages 
        # up to the largest power of 2 less than k.
        
        comparators = []
        step = 1
        while step < k:
            # In each stage, we compare pairs (i, i+step)
            # for i in 0..k-1, if (i // (2*step)) % 2 == 0
            # This is the standard bitonic merge pattern.
            for i in range(k):
                if (i // (2 * step)) % 2 == 0:
                    j = i + step
                    if j < k:
                        comparators.append((i, j))
            step *= 2
        return comparators

    return gen_asc(n)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, leaving 10s margin
    
    # 1. Generate initial network
    network = generate_bitonic_sort(N)
    
    # 2. Prepare test vectors
    # Generate all 2^18 boolean inputs
    # Shape: (2^18, 18)
    # Using np.indices or similar
    # 2^18 = 262144
    # We can generate this as a binary counter.
    test_vectors = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        # Create a pattern of 0s and 1s
        # 2^(N-1-i) ones, then 2^(N-1-i) zeros, repeated
        period = 1 << (N - i)
        half = period // 2
        # Repeat [0]*half + [1]*half
        pattern = np.array([0]*half + [1]*half, dtype=np.uint8)
        reps = (1 << N) // period
        test_vectors[:, i] = np.tile(pattern, reps)
        
    # 3. Optimization Loop
    # We will try to remove redundant comparators.
    # We do this in multiple passes.
    
    def check_network(net):
        # Apply network to test_vectors
        # To avoid modifying the original, we copy
        # But copying 262k x 18 is cheap.
        current = test_vectors.copy()
        for i, j in net:
            min_val = np.minimum(current[:, i], current[:, j])
            max_val = np.maximum(current[:, i], current[:, j])
            current[:, i] = min_val
            current[:, j] = max_val
        # Check if sorted
        # diff along axis 1
        diffs = np.diff(current, axis=1)
        return np.all(diffs >= 0)
    
    # Initial check
    if not check_network(network):
        # Fallback to odd-even transposition if generation failed
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Greedy Removal
    # We iterate through the network multiple times.
    # In each pass, we try to remove each comparator.
    # If removal keeps it valid, we remove it permanently.
    
    # To speed up, we can randomize the order or just go reverse.
    # Reverse is often good because later comparators are more likely to be redundant 
    # if the earlier ones already did the work.
    
    changed = True
    passes = 0
    max_passes = 10
    
    while changed and passes < max_passes and time.time() < deadline:
        changed = False
        passes += 1
        # Iterate in reverse
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Check candidate
            if check_network(candidate):
                network = candidate
                changed = True
            # Else, keep it
    
    return network

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, leaving 10s margin
    
    # 1. Generate initial network
    network = generate_bitonic_sort(N)
    
    # 2. Prepare test vectors
    # Generate all 2^18 boolean inputs
    test_vectors = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        period = 1 << (N - i)
        half = period // 2
        pattern = np.array([0]*half + [1]*half, dtype=np.uint8)
        reps = (1 << N) // period
        test_vectors[:, i] = np.tile(pattern, reps)
        
    # 3. Optimization Loop
    def check_network(net):
        current = test_vectors.copy()
        for i, j in net:
            min_val = np.minimum(current[:, i], current[:, j])
            max_val = np.maximum(current[:, i], current[:, j])
            current[:, i] = min_val
            current[:, j] = max_val
        diffs = np.diff(current, axis=1)
        return np.all(diffs >= 0)
    
    # Initial check
    if not check_network(network):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Greedy Removal
    changed = True
    passes = 0
    max_passes = 10
    
    while changed and passes < max_passes and time.time() < deadline:
        changed = False
        passes += 1
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if check_network(candidate):
                network = candidate
                changed = True
    
    return network
# EVOLVE-BLOCK-END
