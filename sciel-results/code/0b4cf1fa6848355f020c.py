"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
NUM_BOOL = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on N=18 wires.
    Uses a greedy heuristic that selects comparators to maximize the reduction
    in the number of Boolean inputs that are not yet sorted.
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Precompute all Boolean inputs as bitmasks (0 to 2^N - 1)
    # For each input, we need to know the sorted order.
    # A Boolean input x (integer) has bits set where value is 1.
    # After sorting, the 1s should be on the right (higher indices) if we consider
    # standard sorting where smaller goes left.
    # Wait, the problem says: "compare-exchange writes min to wire i and max to j" with i < j.
    # So if wire i has 1 and wire j has 0, after compare-exchange, wire i gets 0 and wire j gets 1.
    # This means 0s move to lower indices and 1s move to higher indices.
    # So a sorted Boolean input has all 0s on the left and all 1s on the right.
    # i.e., the bitmask should be of the form (1 << k) - 1 << (N - k) ? No.
    # Let's trace:
    # Input: 1010 (binary, wire 0 is MSB? or LSB?)
    # Standard bit representation in Python: bit i is 2^i.
    # If wire i has value v_i, the integer is sum(v_i * 2^i).
    # Sorting means v_0 <= v_1 <= ... <= v_{N-1}.
    # So sorted integer has bits 0..N-1 such that if v_i=1 then v_j=1 for all j>i.
    # This means the set of 1s is a suffix of the bit positions.
    # e.g., 000111 (bits 0,1,2 are 1) -> value 7.
    # 001111 -> value 15.
    # So the target sorted value for an input with k ones is (1 << k) - 1.
    
    # We can represent the state of the network as a set of permutations? No, that's too big.
    # Instead, we track for each Boolean input whether it is sorted.
    # Initially, an input is sorted if it is already in the form (1<<k)-1.
    # When we add a comparator (i, j), it affects inputs where v_i != v_j.
    # Specifically, if v_i=1 and v_j=0, the comparator swaps them to v_i=0, v_j=1.
    # This changes the input x to x' = x - (1<<i) + (1<<j).
    # If v_i=0 and v_j=1, no change.
    
    # We want to minimize the number of unsorted inputs.
    # Let S be the set of unsorted inputs.
    # Initially, S = {x | x is not of the form (1<<k)-1}.
    
    # We can simulate the effect of a comparator on the set of unsorted inputs.
    # But 2^18 = 262144 is manageable.
    # We can maintain a boolean array `sorted_mask` of length NUM_BOOL, where
    # sorted_mask[x] is True if x is sorted.
    # When we add a comparator (i, j), we update the state of each input.
    # But we need to know the current value of each wire for each input, which depends
    # on the sequence of comparators applied so far.
    
    # This is equivalent to simulating the network for all 262144 inputs in parallel.
    # We can represent the state of all inputs as a matrix of shape (NUM_BOOL, N).
    # But that's 262144 * 18 bits, which is fine.
    # Actually, we can use numpy to simulate all inputs at once.
    
    # Let's use a numpy array `states` of shape (NUM_BOOL, N) with dtype uint8.
    # states[x, i] is the value of wire i for input x after applying the current network.
    
    # Initialize states:
    # For each x in 0..NUM_BOOL-1, states[x, i] = (x >> i) & 1.
    
    states = np.zeros((NUM_BOOL, N), dtype=np.uint8)
    for i in range(N):
        # Extract bit i from all x
        states[:, i] = ((np.arange(NUM_BOOL, dtype=np.uint32) >> i) & 1).astype(np.uint8)
    
    # Precompute the target sorted states.
    # target_sorted[x] is True if x is sorted.
    # x is sorted if its 1s are a suffix.
    # i.e., x & (x+1) == 0 ? No, that's for powers of 2.
    # x is of the form (1<<k)-1 iff x+1 is a power of 2, or x==0.
    # Let's just check: for each x, count trailing zeros? No.
    # x is sorted iff x == (1 << (x.bit_count())) - 1.
    
    # We can precompute a boolean array `is_sorted` of length NUM_BOOL.
    is_sorted = np.zeros(NUM_BOOL, dtype=bool)
    for x in range(NUM_BOOL):
        k = x.bit_count()
        is_sorted[x] = (x == (1 << k) - 1)
    
    # Now, we greedily select comparators.
    # We want to maximize the number of inputs that become sorted.
    # But an input might become unsorted? No, once sorted, it stays sorted?
    # Let's check: if x is sorted, all 0s are on the left, 1s on the right.
    # A comparator (i, j) with i < j:
    # If v_i=0, v_j=0 -> no change.
    # If v_i=1, v_j=1 -> no change.
    # If v_i=0, v_j=1 -> no change (already sorted order).
    # If v_i=1, v_j=0 -> this would mean 1 is left of 0, which contradicts sorted.
    # So if x is sorted, no comparator can change it to an unsorted state.
    # Wait, is that true?
    # Sorted means v_0 <= v_1 <= ... <= v_{N-1}.
    # So for i < j, v_i <= v_j.
    # So v_i=1 and v_j=0 is impossible.
    # Thus, sorted inputs remain sorted.
    
    # So we can just track the set of sorted inputs, and it only grows.
    # We want to pick the comparator that maximizes the number of new sorted inputs.
    
    # We maintain a boolean array `currently_sorted` of length NUM_BOOL.
    currently_sorted = is_sorted.copy()
    
    # We also need to know, for each input x, the current value of each wire.
    # This is given by `states`.
    
    # For each candidate comparator (i, j), we can compute how many new inputs become sorted.
    # An input x becomes sorted if:
    # 1. It is not currently sorted.
    # 2. After applying (i, j), it is sorted.
    
    # Applying (i, j) to x:
    # If states[x, i] == 0 and states[x, j] == 1: no change.
    # If states[x, i] == 1 and states[x, j] == 0: swap -> new_x = x - (1<<i) + (1<<j).
    # If states[x, i] == states[x, j]: no change.
    
    # So for each (i, j), we can compute the set of x that change, and check if new_x is sorted.
    
    # But we need to be careful: the `states` array is updated as we add comparators.
    # So we need to simulate the effect of (i, j) on the current `states`.
    
    # Let's define a function to evaluate a comparator (i, j) given the current `states`.
    # It returns the number of new sorted inputs.
    
    def evaluate_comparator(i, j):
        # Find all x where states[x, i] == 1 and states[x, j] == 0
        mask = (states[:, i] == 1) & (states[:, j] == 0)
        # For these x, the new value is x' = x - (1<<i) + (1<<j)
        # We need to check if x' is sorted.
        # But we don't have x' as an integer, we have the new state vector.
        # However, we can precompute a lookup table: for each possible state vector, is it sorted?
        # No, the state vector is N bits, so 2^N possibilities.
        # We can encode the new state vector as an integer and check is_sorted[new_x].
        
        # Let's compute new_x for all x where mask is True.
        # But we need to update states for these x.
        # Actually, we just need to count how many of these x become sorted.
        
        # For each x in mask:
        # new_state = states[x].copy()
        # new_state[i] = 0
        # new_state[j] = 1
        # new_x = sum(new_state[k] * (1<<k) for k in range(N))
        # if is_sorted[new_x] and not currently_sorted[x]:
        #     count += 1
        
        # This is slow in Python. Let's use numpy.
        
        # We can compute new_x for all x at once.
        # But we only care about x where mask is True.
        
        # Let's create a copy of states for the affected x.
        # Actually, we can compute the new integer value directly.
        
        # x_new = x - (1<<i) + (1<<j) for x where mask is True.
        # But x is not directly in our arrays. We have states.
        # We can reconstruct x from states: x = sum(states[x, k] * (1<<k)).
        
        # Let's precompute the integer values of all states.
        # This is expensive to do every time.
        
        # Alternative: maintain an array `int_states` of shape (NUM_BOOL,) with dtype uint32.
        # int_states[x] is the integer value of the state vector for input x.
        
        # Let's add this to the outer scope.
        
        # For now, let's just compute it.
        
        # This is getting complicated. Let's use a different approach.
        
        # We can maintain `int_states` as a numpy array.
        # When we apply a comparator (i, j), we update `int_states` for the affected x.
        
        # Let's restructure the code to use `int_states`.
        
        pass
    
    # Let's restart with a cleaner design using `int_states`.
    
    # Re-initialize
    int_states = np.zeros(NUM_BOOL, dtype=np.uint32)
    for i in range(N):
        int_states |= ((np.arange(NUM_BOOL, dtype=np.uint32) >> i) & 1).astype(np.uint32) << i
    # Actually, int_states[x] is just x.
    int_states = np.arange(NUM_BOOL, dtype=np.uint32)
    
    currently_sorted = is_sorted.copy()
    
    # Precompute bit masks
    bit_masks = [1 << i for i in range(N)]
    
    # List of all candidate comparators
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # We will greedily select comparators.
    # We stop when all inputs are sorted or when we run out of time.
    
    network = []
    
    while time.time() < deadline:
        # Check if all are sorted
        if currently_sorted.all():
            break
        
        best_comparator = None
        best_gain = -1
        
        for i, j in candidates:
            # Find x where int_states[x] has bit i set and bit j not set.
            mask = (int_states & bit_masks[i]) & ~(int_states & bit_masks[j])
            # For these x, the new value is x' = x - bit_masks[i] + bit_masks[j]
            # We need to check if x' is sorted and x is not currently sorted.
            
            # Get the indices of x where mask is True
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                continue
            
            # Compute new values
            new_vals = int_states[idx] - bit_masks[i] + bit_masks[j]
            
            # Check which of these new values are sorted
            newly_sorted = is_sorted[new_vals] & ~currently_sorted[idx]
            gain = newly_sorted.sum()
            
            if gain > best_gain:
                best_gain = gain
                best_comparator = (i, j)
        
        if best_comparator is None or best_gain <= 0:
            # No progress possible
            break
        
        # Apply the best comparator
        i, j = best_comparator
        mask = (int_states & bit_masks[i]) & ~(int_states & bit_masks[j])
        int_states[mask] = int_states[mask] - bit_masks[i] + bit_masks[j]
        
        # Update currently_sorted
        newly_sorted = is_sorted[int_states] & ~currently_sorted
        currently_sorted[newly_sorted] = True
        
        network.append((i, j))
    
    return network
# EVOLVE-BLOCK-END
