import numpy as np
import time
import random

N = 18
NUM_INPUTS = 2 ** N

def _generate_inputs():
    """Generate all 2^N boolean input vectors as a uint8 array."""
    x = np.arange(NUM_INPUTS, dtype=np.uint64)[:, None]
    # We want wire 0 to be LSB, wire N-1 to be MSB?
    # The comparator (i, j) operates on wires i and j.
    # The order of wires in the array doesn't matter as long as it's consistent.
    # Let's use bit 0 for wire 0.
    bits = (x >> np.arange(N, dtype=np.uint64)) & 1
    return bits.astype(np.uint8)

def _verify_network(network, inputs):
    """
    Check if the network sorts all inputs.
    inputs: array of shape (2^N, N)
    """
    if not network:
        return True
        
    # Copy inputs to state
    # Using uint8 for efficiency
    state = inputs.copy()
    
    for i, j in network:
        # min to i, max to j
        # For 0/1 values:
        # min = a & b
        # max = a | b
        # We need to be careful with in-place updates if i and j overlap?
        # They don't, i != j.
        # But we need the original values for both.
        vi = state[:, i]
        vj = state[:, j]
        
        # Calculate min and max
        # Since vi and vj are views, we can compute directly
        min_val = vi & vj
        max_val = vi | vj
        
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    # For binary, this means no 1 before 0.
    # Equivalent to: for each row, all 0s are before all 1s.
    # We can check if the array is non-decreasing along axis 1.
    # np.diff(state, axis=1) >= 0
    # Or simpler: 
    # A row is sorted if it matches the pattern of zeros then ones.
    # Since it's binary, sorted rows are exactly the rows where 
    # state[:, :-1] <= state[:, 1:]
    
    if N > 1:
        return np.all(state[:, :-1] <= state[:, 1:])
    return True

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    # The evaluator kills at 90s. 
    # The prompt says "Each evaluation must finish within 60 s".
    # But also "KILLS ... after 90s".
    # Let's aim for 55s to be safe against overhead.
    deadline = start_time + 55.0
    
    # Pre-compute inputs
    inputs = _generate_inputs()
    
    # Start with Odd-Even Transposition
    network = _odd_even_transposition()
    best_network = network.copy()
    
    # Local Search: Greedy Deletion
    # We try to remove random comparators.
    # If the network remains valid, we keep the removal.
    # This monotonically decreases the number of comparators.
    
    # To avoid getting stuck, we could randomize the order of comparators?
    # No, the list order defines the depth, but for validity, only the set matters?
    # Wait. 
    # "Construct a compare-exchange network... ordered list of integer pairs".
    # "Each compare-exchange writes min to wire i and max to j."
    # The sequence matters. 
    # If I swap two comparators in the list, I change the network.
    # My deletion algorithm removes a comparator from the sequence.
    # This is valid.
    
    # Optimization: 
    # If we fail to remove a comparator, we might try to remove a different one.
    # Simple random sampling works well.
    
    # We can also try to "shrink" the network by removing duplicates?
    # Odd-even transposition has no duplicate (i,j) pairs in the same phase,
    # but across phases, (0,1) appears multiple times.
    # Removing one instance is what we do.
    
    # To speed up convergence, we can try removing comparators that are "redundant" 
    # based on some heuristic? 
    # For now, random is robust.
    
    # Let's add a counter for consecutive failures to potentially restart or change strategy?
    # If we get stuck, we might just stop.
    
    # We can also try to randomize the initial network?
    # No, seed is fixed.
    
    # Let's run the loop
    # We will check time every 100 iterations to avoid overhead of time.time()
    # But time.time() is cheap.
    
    # To improve search, we can also try to swap comparators?
    # Swapping doesn't change the count, so it doesn't improve score directly,
    # but it might allow further deletions.
    # However, the prompt scores based on count only.
    # So we only care about deletion.
    
    # If we are stuck, we might try to add a comparator and then remove a different one?
    # This is "hill climbing" with moves of size 0 or -1.
    # But adding increases count. We want to minimize.
    # So we stick to deletion.
    
    # One potential issue: The initial network is 162.
    # If we can't get below 162, score is 1.0.
    # But we know we can.
    
    # Let's ensure we return the best (shortest) network found.
    # Since we only delete, the current network is always the best found so far
    # IF we only keep valid deletions.
    # If we restore, the count goes back up.
    # So `best_network` should be updated only when we successfully delete.
    
    # Wait, if I delete and it's valid, I update `best_network`.
    # If I delete and it's invalid, I restore.
    # So `network` length is non-decreasing?
    # No.
    # Start: 162.
    # Try delete -> 161. Valid? Yes. `network` is 161. `best` is 161.
    # Try delete -> 160. Valid? No. Restore -> 161.
    # Try delete -> 160. Valid? Yes. `network` is 160. `best` is 160.
    # So `network` length decreases over time.
    # `best_network` tracks the minimum length achieved.
    # Since we restore on failure, `network` always represents a valid state.
    # And its length is the current best.
    # So we can just return `network` at the end?
    # Yes, because we never accept an invalid state, and we never increase the size.
    # So `network` is always valid and its size is non-increasing.
    
    # However, if we want to be extra safe, we keep `best_network`.
    
    # Let's run
    while time.time() < deadline:
        if len(network) == 0:
            break
            
        idx = random.randint(0, len(network) - 1)
        comp = network.pop(idx)
        
        if _verify_network(network, inputs):
            # Success
            best_network = network.copy()
        else:
            # Failure, restore
            network.insert(idx, comp)
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_network
