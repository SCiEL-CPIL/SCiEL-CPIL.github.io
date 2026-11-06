"""Simulated Annealing with Add-Remove kicks for 17-wire sorting networks."""
import time
import random
import numpy as np

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072

def _time_left(deadline):
    return deadline - time.time()

def _verify_network(comparators):
    """Verify network sorts all 2^N boolean inputs using zero-one principle.
    Returns True if valid, False otherwise."""
    n = N
    num_inputs = 1 << n
    # We'll process all inputs at once using bit operations
    # For each boolean input, simulate the network
    # Represent each wire's value as a bit in a numpy array
    
    # For efficiency, we process in batches
    # Each input is an n-bit number; we need to check if after sorting,
    # the result equals the sorted version (all 0s then all 1s)
    
    # The sorted version of a boolean input with k ones is:
    # first (n-k) wires are 0, last k wires are 1
    
    # We'll use a vectorized approach: represent all 2^N inputs as a 2D array
    # where each row is one input's wire values.
    
    # Actually, let's use a more memory-efficient approach.
    # Process inputs in chunks to avoid memory issues.
    
    chunk_size = 4096
    for start in range(0, num_inputs, chunk_size):
        end = min(start + chunk_size, num_inputs)
        # Create input matrix: shape (chunk, n)
        inputs = np.arange(start, end, dtype=np.uint8)
        # Extract bits
        # inputs[:, i] = (input >> i) & 1
        wires = np.zeros((end - start, n), dtype=np.uint8)
        for i in range(n):
            wires[:, i] = (inputs >> i) & 1
        
        # Simulate network
        for (a, b) in comparators:
            # Compare wires a and b, swap if needed
            wa = wires[:, a]
            wb = wires[:, b]
            # min goes to a, max goes to b
            wires[:, a] = np.minimum(wa, wb)
            wires[:, b] = np.maximum(wa, wb)
        
        # Check if sorted: for each row, all 0s should come before all 1s
        # i.e., the row should be non-decreasing
        # Check: no 1 before a 0
        # A row is sorted iff for all i < j, if wires[i]=1 then wires[j]=1
        # Equivalently: the row is of form [0,0,...,0,1,1,...,1]
        # Check: sum of first (n-k) elements = 0 and last k = 1 where k = sum
        
        # Simple check: the array should be non-decreasing
        # For uint8, check that no position has 1 followed by 0
        # diff = wires[:, 1:] - wires[:, :-1], sorted means all diffs >= 0
        diffs = wires[:, 1:].astype(np.int8) - wires[:, :-1].astype(np.int8)
        if np.any(diffs < 0):
            return False
    
    return True

def _compute_depth(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # depth[i] = longest path to wire i
    depth = [0] * N
    for (a, b) in comparators:
        new_depth_a = max(depth[a], depth[b]) + 1
        # Actually, both a and b get updated
        # The depth of the comparator is max(depth[a], depth[b]) + 1
        # Both outputs have this depth
        d = max(depth[a], depth[b]) + 1
        depth[a] = d
        depth[b] = d
    return max(depth) if depth else 0

def _make_seed():
    """Create the odd-even transposition network seed."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 85  # 85s to leave 5s safety margin
    
    # Start with seed network
    current = _make_seed()
    current_valid = _verify_network(current)
    current_len = len(current)
    
    best = current[:]
    best_len = current_len
    best_valid = current_valid
    
    # Simulated Annealing with Add-Remove moves
    # State: list of (i, j) comparators
    # Moves:
    #   Add: insert a random comparator (i, j) at a random position
    #   Remove: remove a random comparator
    #   Swap: swap two adjacent comparators
    #   Replace: change a comparator's endpoints
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    T = 1.0  # initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    iteration = 0
    while _time_left(deadline) > 2.0:
        iteration += 1
        
        if T < T_min:
            T = T_min
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.35:
            # Add a comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(current))
            new_net = current[:pos] + [(i, j)] + current[pos:]
        elif move_type < 0.65:
            # Remove a comparator
            if len(current) <= 1:
                continue
            pos = rng.randint(0, len(current) - 1)
            new_net = current[:pos] + current[pos+1:]
        elif move_type < 0.85:
            # Swap adjacent comparators
            if len(current) < 2:
                continue
            pos = rng.randint(0, len(current) - 2)
            new_net = current[:]
            new_net[pos], new_net[pos+1] = new_net[pos+1], new_net[pos]
        else:
            # Replace endpoints of a random comparator
            if len(current) == 0:
                continue
            pos = rng.randint(0, len(current) - 1)
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            new_net = current[:]
            new_net[pos] = (i, j)
        
        # Evaluate
        new_len = len(new_net)
        new_valid = _verify_network(new_net)
        
        # Objective: minimize length, but must be valid
        # If invalid, heavily penalize
        if new_valid:
            new_cost = new_len
        else:
            new_cost = new_len + 10000  # heavy penalty
        
        if current_valid:
            current_cost = current_len
        else:
            current_cost = current_len + 10000
        
        delta = new_cost - current_cost
        
        # Accept or reject
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = new_net
            current_len = new_len
            current_valid = new_valid
            
            # Update best
            if new_valid and new_len < best_len:
                best = new_net[:]
                best_len = new_len
                best_valid = True
        
        # Cool down
        T *= cooling
        
        # Periodic reheat to escape local minima
        if iteration % 500 == 0 and T < 0.1:
            T = 0.5
    
    return best

# EVOLVE-BLOCK-START
# (construct function is defined above; this block is kept for compatibility)
# EVOLVE-BLOCK-END
