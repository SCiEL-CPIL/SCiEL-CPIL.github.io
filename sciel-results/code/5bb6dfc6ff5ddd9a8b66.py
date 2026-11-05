import numpy as np
import time

N = 18

def construct():
    """
    Constructs a sorting network by starting with a known valid network 
    (Odd-Even Transposition) and greedily removing comparators while 
    maintaining correctness over all 2^N Boolean inputs using bit-parallel verification.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # 1. Generate Seed Network: Odd-Even Transposition
    # This is a valid sorting network for N wires.
    # Phase 0: (0,1), (2,3), ...
    # Phase 1: (1,2), (3,4), ...
    # ...
    # Total N phases for N wires? No, N phases is enough for N wires?
    # Actually, odd-even transposition takes N phases to sort N elements.
    # Let's verify: for N=2, phases: (0,1). 1 phase. N=2.
    # For N=4: 
    # P0: (0,1), (2,3)
    # P1: (1,2)
    # P2: (0,1), (2,3)
    # P3: (1,2)
    # Total 4 phases.
    # So yes, N phases.
    
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # 2. Prepare Bit-Parallel Verification Data
    # We need to test all 2^N inputs.
    # Instead of iterating 2^N times, we represent the state of all 2^N inputs 
    # simultaneously.
    # State: A 2D array of shape (2^N, N) would be 262144 * 18 * 8 bytes ~ 37 MB.
    # This is feasible in memory.
    # However, we can be even more efficient by representing the bits of each wire
    # across all inputs as a single integer or a small array of integers if N > 64.
    # Since N=18, we can store the state of all 2^18 inputs for a single wire as a 
    # single 2^18-bit integer? No, Python ints are arbitrary precision, so we can 
    # store a 262144-bit integer. 
    # Let's use Python integers where bit k represents the value of wire i for input k.
    
    # Initialize state for all inputs.
    # Input k (0 to 2^N - 1) has bit j set if the j-th wire of input k is 1.
    # Wait, standard binary representation: input k has bit j = (k >> j) & 1.
    # So for wire j, the value across all inputs is the bit j of the sequence 0, 1, ..., 2^N-1.
    # This pattern is a repeating sequence of 0s and 1s.
    # Specifically, wire j has 2^(N-j-1) zeros followed by 2^(N-j-1) ones, repeated.
    
    # Let's construct the initial state for each wire as a Python integer.
    # state[j] is an integer where bit k is 1 if wire j of input k is 1.
    
    states = [0] * N
    for j in range(N):
        # Pattern for wire j: period 2^(j+1), half 0, half 1.
        # We need to fill 2^N bits.
        # It's easier to just generate the mask.
        # Mask for wire j: bits k where (k >> j) & 1 == 1.
        # We can build this iteratively or use a lookup.
        # Since N=18, 2^18 = 262144.
        # Building a 262144-bit integer is fast.
        
        # Alternative: Use NumPy for speed in the verification loop?
        # Python big int bitwise ops are very fast in C.
        # Let's stick to Python big ints for the state of each wire.
        
        # Construct mask for wire j
        # The bit k of the mask is 1 if (k >> j) % 2 == 1
        # This is equivalent to: mask_j = sum(1 << k for k in range(2^N) if (k >> j) & 1)
        
        # Efficient construction:
        # The pattern is 2^j zeros, 2^j ones.
        # Block size = 1 << (j+1)
        # Half block = 1 << j
        # We need to repeat this block 2^(N-j-1) times.
        
        # Let's just use a simple loop to build the integer if N is small, 
        # but 2^18 is too big for a naive loop.
        # We can use the fact that the mask for wire j is related to wire j-1.
        # Actually, let's use NumPy to generate the initial states as uint64 arrays?
        # No, Python big ints are cleaner for bitwise ops on 2^18 bits.
        
        # Trick to build the big int mask:
        # Let's compute it using shifts.
        # mask = 0
        # for k in range(2**N):
        #    if (k >> j) & 1: mask |= (1 << k)
        # This is O(2^N) per wire, total O(N * 2^N) ~ 18 * 262144 ~ 4.7M ops. 
        # Python loop might be too slow (4.7M iterations ~ 0.5s). 
        # Let's do it once.
        
        pass # Placeholder, will implement efficient mask gen

    # Efficient mask generation using NumPy or bitwise tricks
    # Let's use a recursive or iterative bitwise construction.
    # Mask for j=0: 101010... (LSB is 1? No, k=0 -> 0, k=1 -> 1. So 010101... from MSB to LSB? 
    # Bit k corresponds to input k.
    # Input 0: ...0000 -> wire 0 is 0.
    # Input 1: ...0001 -> wire 0 is 1.
    # So bit 0 of mask is 0, bit 1 is 1, bit 2 is 0, bit 3 is 1.
    # Mask = 0xAAAA... (if LSB is bit 0).
    # Wait, 1<<0 is bit 0.
    # k=0: bit 0 is 0.
    # k=1: bit 1 is 1.
    # So mask has 1s at odd positions.
    
    # Let's generate masks using NumPy for speed, then convert to Python ints?
    # Or just use NumPy arrays for the state?
    # State: array of shape (N, 2^N) of uint8? 18 * 262144 * 1 byte = 4.7 MB.
    # This is very fast to manipulate with vectorized operations.
    
    # Let's use NumPy arrays for the state.
    # state[j] is an array of length 2^N, containing 0 or 1.
    
    size = 1 << N
    # Initial state:
    # state[j][k] = (k >> j) & 1
    # We can create a range array and extract bits.
    k_range = np.arange(size, dtype=np.uint64)
    states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    
    # 3. Greedy Removal
    # We will try to remove comparators from the end of the network backwards.
    # Why backwards? Because the last comparators are often redundant in transposition networks.
    # We'll iterate through the network in reverse order.
    
    # To make it faster, we can try to remove multiple at once or just iterate.
    # Verification cost: 
    # Applying a comparator (i, j) to states:
    # min_val = states[i] & states[j]
    # max_val = states[i] | states[j]
    # states[i] = min_val
    # states[j] = max_val
    # This is O(size) vectorized ops. 262144 elements. Very fast (~0.1 ms).
    
    # Check if sorted:
    # For all k, states[0][k] >= states[1][k] >= ... >= states[N-1][k]
    # Since values are 0/1, this means if states[j][k] is 0, all subsequent must be 0.
    # Or simply: 
    # diffs = states[:-1] - states[1:]
    # If any diff < 0, it's not sorted.
    # Vectorized check: np.any(states[:-1] < states[1:])
    
    current_network = network[:]
    
    # We'll try to remove comparators. 
    # Strategy: Iterate through the network from last to first.
    # For each comparator, try removing it, verify. If valid, keep it removed.
    # If invalid, put it back.
    
    # To be more aggressive, we can do multiple passes.
    
    # Optimization: Pre-calculate the effect? No, state changes.
    
    # Let's implement the removal loop.
    
    # We need to be careful with time.
    # Each verification is ~0.1-0.5 ms.
    # 153 comparators.
    # If we try to remove each one, that's 153 verifications.
    # If we do multiple passes, say 10 passes, that's 1530 verifications.
    # 1530 * 0.5ms = 0.75s. This is well within budget.
    
    # We can also try to remove more if we are fast.
    
    # Let's refine the removal strategy:
    # 1. Start with full network.
    # 2. Loop:
    #    a. For i from len(network)-1 down to 0:
    #       b. Remove network[i]
    #       c. Verify
    #       d. If valid, keep removed (don't add back)
    #       e. If invalid, add back
    #    b. If no removals happened in this pass, break.
    
    # This is standard greedy.
    
    # Let's implement this.
    
    def is_sorted(states):
        # states shape (N, size)
        # Check if states[0] >= states[1] ...
        # Since binary, states[j] >= states[j+1] is equivalent to ~states[j] | states[j+1] == 1?
        # No. If A >= B, then A=0, B=0 (ok), A=1, B=0 (ok), A=1, B=1 (ok).
        # Only bad case: A=0, B=1.
        # So we check if there is any k where states[j][k] == 0 and states[j+1][k] == 1.
        # This is (states[j] == 0) & (states[j+1] == 1)
        # Or simply: np.any(states[:-1] < states[1:])
        return not np.any(states[:-1] < states[1:])
    
    def apply_comparator(states, i, j):
        # In-place update? 
        # min = states[i] & states[j]
        # max = states[i] | states[j]
        # To avoid race conditions if i and j overlap? They don't in a single step.
        # But we need to store old values first.
        si = states[i].copy()
        sj = states[j].copy()
        states[i] = si & sj
        states[j] = si | sj
        
    # Main Loop
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        # Iterate backwards
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            comp = current_network[idx]
            i, j = comp
            
            # Save current states
            # Instead of copying entire states array (4.7MB) for every trial,
            # we can just apply, check, and revert if failed.
            # Reverting is just applying the same comparator again? 
            # No, compare-exchange is not its own inverse in terms of state restoration 
            # if we just re-apply. 
            # Wait, if we have state S, apply (i,j) -> S'. 
            # Apply (i,j) to S' -> S''?
            # min(Si, Sj), max(Si, Sj)
            # If Si < Sj: min=Si, max=Sj. State unchanged.
            # If Si > Sj: min=Sj, max=Si. State swapped.
            # So applying twice restores original order if it was swapped, or keeps it if not.
            # So yes, apply_comparator is an involution on the state! 
            # S --(i,j)--> S' --(i,j)--> S
            # So we can just apply, check, and if fail, apply again to revert.
            
            # Apply
            apply_comparator(states, i, j)
            
            # Check
            if is_sorted(states):
                # Valid, keep removed
                current_network.pop(idx)
                removed_any = True
                # Note: we don't revert, so states are now in the "removed" state.
                # We continue with this state.
            else:
                # Invalid, revert
                apply_comparator(states, i, j)
                
            # If we removed one, the indices shift? 
            # We are iterating backwards, so popping at idx doesn't affect indices < idx.
            # So the loop is safe.
            
    return current_network

# Initial test/verification logic to ensure correctness before returning
# The construct function above does the work.
# We just need to return the list of tuples.

# One detail: The problem says "Maximum 4096 comparators". 
# Our seed is 153. Greedy removal only reduces it. So we are safe.

# Let's double check the "involution" claim.
# Compare-exchange:
# a, b -> min(a,b), max(a,b)
# If we apply again:
# min(min(a,b), max(a,b)), max(min(a,b), max(a,b))
# Since min(a,b) <= max(a,b), 
# min(min(a,b), max(a,b)) = min(a,b)
# max(min(a,b), max(a,b)) = max(a,b)
# So yes, it is idempotent after the first application? 
# Wait.
# If a < b:
# Step 1: a, b -> a, b
# Step 2: a, b -> a, b
# It is idempotent.
# If a > b:
# Step 1: a, b -> b, a
# Step 2: b, a -> b, a
# It is idempotent.
# So applying the comparator twice does NOT revert the state if it was already sorted by that comparator.
# It DOES revert if we consider the "operation" as a swap? No, it's a sort.
# So if we apply (i,j) and the state changes (i.e., Si > Sj), then S' has Si' < Sj'.
# Applying (i,j) to S' results in S'' = S'.
# So S' is the fixed point.
# We CANNOT revert by applying again.
# We MUST save the state or calculate the inverse.
# Since it's a min/max, the inverse is not simply applying again.
# However, we can save the two wires' values before applying.
# si = states[i].copy()
# sj = states[j].copy()
# apply...
# if fail:
#   states[i] = si
#   states[j] = sj

# Let's update the code to use explicit save/restore for the two wires.
# This is cheap (2 * 262144 bytes ~ 0.5 MB copy).

# Revised construct:

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Seed Network
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    size = 1 << N
    k_range = np.arange(size, dtype=np.uint64)
    states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
        
    def is_sorted(states):
        return not np.any(states[:-1] < states[1:])
        
    def apply_comparator(states, i, j):
        si = states[i].copy()
        sj = states[j].copy()
        states[i] = si & sj
        states[j] = si | sj
        return si, sj
        
    def restore_comparator(states, i, j, si, sj):
        states[i] = si
        states[j] = sj

    current_network = network[:]
    
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            i, j = current_network[idx]
            
            # Save
            si, sj = states[i].copy(), states[j].copy()
            
            # Apply
            states[i] = si & sj
            states[j] = si | sj
            
            # Check
            if is_sorted(states):
                # Keep removed
                current_network.pop(idx)
                removed_any = True
            else:
                # Revert
                states[i] = si
                states[j] = sj
                
    return current_network

# We need to make sure we don't exceed the time limit during the final return or evaluation.
# The evaluator calls construct().
# The construct function itself takes time.
# The 80s deadline is inside construct.
# This is correct.

# One edge case: What if the seed network is not valid? 
# Odd-even transposition is valid.
# What if greedy removal gets stuck in a local minimum?
# It will return the best found so far.
# The score is based on the number of comparators.
# Fewer is better.
# So this is a valid strategy.

# Let's add a small optimization: 
# If we are very fast, we can do more passes.
# The current loop does as many passes as possible until no removals or time out.
# This is optimal for the given strategy.

# Final check on imports and structure.
# import numpy as np
# import time
# N = 18
# def construct(): ...

# The code looks solid.
