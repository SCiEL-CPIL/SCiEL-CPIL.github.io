import time
import numpy as np

N = 18
NUM_MASKS = 1 << N

def _verify_network(network):
    """
    Verify if the sorting network sorts all 0-1 inputs using vectorized NumPy.
    Returns True if valid, False otherwise.
    """
    if not network:
        return True
        
    # Pre-compute indices for faster access
    # We simulate all 2^N inputs at once.
    # Input matrix shape: (NUM_MASKS, N)
    # mask i has bit j set if (i >> j) & 1
    
    # Generate all masks
    # Using bit operations to create the initial state matrix
    # This is a standard trick: create an array of all integers 0..2^N-1
    masks = np.arange(NUM_MASKS, dtype=np.uint8)
    
    # Create input matrix: X[m, k] = bit k of m
    # We can do this by reshaping and using broadcasting, or simply:
    # X = (masks[:, None] >> np.arange(N)) & 1
    # This creates an array of shape (NUM_MASKS, N)
    X = (masks[:, None] >> np.arange(N, dtype=np.uint8)) & 1
    
    # Apply network
    # For each comparator (i, j):
    #   X[:, i], X[:, j] = min(X[:, i], X[:, j]), max(X[:, i], X[:, j])
    # Since inputs are 0/1, min is AND, max is OR.
    # X[:, i] &= X[:, j]
    # X[:, j] |= X[:, i_old] -- wait, need to be careful with in-place ops if i, j overlap?
    # No, i != j.
    # But we need the original X[:, i] for the OR.
    # So:
    #   old_i = X[:, i].copy()
    #   X[:, i] = old_i & X[:, j]
    #   X[:, j] = old_i | X[:, j]
    
    # Optimization: Pre-allocate a temp array? Or just use two lines.
    # Since we are doing this in a loop, copies might be expensive.
    # Alternative:
    #   tmp = X[:, i]
    #   X[:, i] = np.minimum(tmp, X[:, j])
    #   X[:, j] = np.maximum(tmp, X[:, j])
    # np.minimum/maximum on uint8 is fast.
    
    for i, j in network:
        tmp = X[:, i]
        X[:, i] = np.minimum(tmp, X[:, j])
        X[:, j] = np.maximum(tmp, X[:, j])
        
    # Check if sorted: X[:, k] <= X[:, k+1] for all k
    # This is equivalent to: no 1 follows a 0 in each row.
    # Or: X is sorted if for all k, X[:, k] <= X[:, k+1].
    # We can check if any row violates this.
    # Violation if X[:, k] > X[:, k+1].
    # Since binary, this means X[:, k] == 1 and X[:, k+1] == 0.
    
    # We can check this by comparing adjacent columns
    # diffs = X[:, :-1] - X[:, 1:]
    # if np.any(diffs < 0): return False
    
    # More efficient:
    # Check if X[:, :-1] > X[:, 1:]
    if np.any(X[:, :-1] > X[:, 1:]):
        return False
        
    return True

def _initial_network():
    """
    Generate the odd-even transposition network.
    """
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with odd-even transposition
    network = _initial_network()
    
    # Verify initial network
    if not _verify_network(network):
        # Should not happen, but just in case
        return network
        
    best_network = list(network)
    best_len = len(network)
    
    # Local Search: Remove comparators
    # We iterate repeatedly until no improvement is found or time runs out.
    
    # To make the search more effective, we can randomize the order of removal attempts
    # or try specific heuristics.
    # Standard "remove one" pass:
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try removing from end to beginning to avoid index shifting issues if we modify in place,
        # but since we verify the whole network, we can just build a new list.
        # However, checking every index is O(L * V).
        # L is ~150. V is fast now.
        
        # We can try to remove comparators in reverse order (last added are often redundant first?)
        # Or just iterate all.
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            if _verify_network(test_network):
                network = test_network
                improved = True
                # Restart the loop to try removing more from the new network
                break
        
    # Add-Remove Cycle to escape local minima
    # We try to add a comparator, then remove one (possibly different) to reduce size.
    # If we add one and remove the same one, no change.
    # If we add one and remove a different one, we have effectively swapped them.
    # If we add one and remove a different one, and the new network is valid, we have the same size.
    # We want to find a sequence that leads to a smaller network.
    # Strategy:
    # 1. Pick a random (i, j) not in network.
    # 2. Add it.
    # 3. Try to remove ANY comparator from the extended network.
    # 4. If successful, we have a network of size L (same as before) but different structure.
    # 5. Then try to remove more.
    
    # Actually, a better "swap" move:
    # Pick two comparators A and B.
    # Try removing A. If valid, done.
    # If not, try adding B (if not present) and removing A.
    
    # Let's implement a simple random restart / perturbation if stuck.
    # But first, let's do a more exhaustive "swap" search.
    
    # We will try to find a pair (remove_idx, add_pair) such that:
    # Network with remove_idx removed and add_pair added is valid.
    # This keeps size constant but changes structure.
    # Then we run the "remove" pass again.
    
    # To save time, we limit the number of such cycles.
    
    # First, let's see if we can just do more "remove" passes with different starting points?
    # No, the "remove" pass is greedy.
    
    # Let's try a "2-opt" style move:
    # Remove one, add one.
    
    # Pre-compute all possible pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Current network set for fast lookup
    net_set = set(network)
    
    # Try to find a swap
    # We iterate through possible removals.
    # For each removal, we check if the resulting network is "close" to valid?
    # No, just check validity of (Network - C + D).
    
    # This is expensive if we do it for all D.
    # However, we can restrict D to "likely" candidates, e.g., comparators that are "far apart" or in "critical" positions.
    # Or just random sampling.
    
    # Given the time budget, let's try a limited number of random swap attempts.
    
    num_swap_attempts = 0
    max_swap_attempts = 5000 # Adjust based on speed
    
    while (time.time() - start_time) < deadline - 10:
        if not improved:
            # Try a swap
            # Pick a random comparator to remove
            if not network:
                break
            remove_idx = np.random.randint(0, len(network))
            remove_pair = network[remove_idx]
            
            # Try to add a random pair
            # To avoid duplicates, we can pick from all_pairs not in net_set
            # But net_set changes.
            
            # Let's just pick a random pair from all_pairs
            add_pair = all_pairs[np.random.randint(0, len(all_pairs))]
            
            if add_pair == remove_pair:
                continue
                
            # Construct test network
            # Remove remove_pair, add add_pair
            # Note: if add_pair is already in network, we are just swapping positions?
            # No, the order matters.
            
            test_network = [c for c in network if c != remove_pair]
            # Insert add_pair at the position of remove_pair? Or at the end?
            # Position matters.
            # Let's try inserting it at the same index.
            test_network.insert(remove_idx, add_pair)
            
            if _verify_network(test_network):
                network = test_network
                net_set = set(network)
                improved = True
                # After a successful swap, we might be able to remove more
                # So we break out of the swap loop and go back to the remove loop
                break
            else:
                num_swap_attempts += 1
                if num_swap_attempts > max_swap_attempts:
                    # If we've tried many swaps without success, maybe try a different strategy?
                    # For now, just stop swapping and return best
                    break
        else:
            # If we improved in the remove phase, reset the swap counter or continue
            # The while loop structure handles this: if improved is True, we don't enter the swap block
            # Wait, the logic above is:
            # while improved...:
            #    ... remove loop ...
            #    if not improved:
            #       ... swap loop ...
            
            # This is correct.
            pass
    
    return network

# To ensure the function is defined correctly for the evaluator
# The evaluator calls construct()

if __name__ == "__main__":
    # Simple test
    net = construct()
    print(f"Network size: {len(net)}")
    print(f"Valid: {_verify_network(net)}")
