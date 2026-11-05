import numpy as np
import time
import random

N = 16
MAX_INPUTS = 2 ** N
DEADLINE = 85.0  # Safety margin for 90s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # 1. Setup Bit-Parallel State
    # We represent the state of all 65536 inputs for all 16 wires.
    # W is shape (N, MAX_INPUTS) of uint8.
    # W[w, k] is the value of wire w for input k (where k is 0..2^N-1).
    
    inputs = np.arange(MAX_INPUTS, dtype=np.uint64)
    W = np.empty((N, MAX_INPUTS), dtype=np.uint8)
    for w in range(N):
        W[w] = (inputs >> w) & 1
    
    # Initial inversion count
    # An inversion is a pair (i, j) with i < j such that W[i, k] > W[j, k].
    # Since values are 0/1, W[i, k] > W[j, k] iff W[i, k]=1 and W[j, k]=0.
    # inv[i, j] = sum_k (W[i, k] * (1 - W[j, k]))
    
    def compute_inversions(W_state):
        # Returns a 2D array inv[i, j] for i < j
        # Or just total sum? 
        # For greedy, we need to know the marginal gain of adding (i, j).
        # Gain = inv[i, j] (inversions removed by sorting i, j)
        # But adding (i, j) might create inversions with other wires?
        # No, sorting (i, j) makes W[i] <= W[j]. 
        # It sets W[i] = min, W[j] = max.
        # This can only reduce the inversion count between i and j.
        # Does it affect inversions with other wires?
        # If we swap values, we might increase inversions with k.
        # Example: W = [1, 0, 1]. Inv(0,1)=1, Inv(1,2)=0, Inv(0,2)=0. Total=1.
        # Sort (0,1) -> [0, 1, 1]. Inv(0,1)=0, Inv(1,2)=0, Inv(0,2)=0. Total=0. Gain=1.
        # Example: W = [1, 0, 0]. Inv(0,1)=1, Inv(0,2)=1. Total=2.
        # Sort (0,1) -> [0, 1, 0]. Inv(0,1)=0, Inv(1,2)=1, Inv(0,2)=0. Total=1. Gain=1.
        # 
        # Actually, the standard greedy for sorting networks usually minimizes the sum of inversions.
        # The marginal benefit of adding comparator (i, j) is the number of inversions it resolves 
        # minus the number it creates. 
        # However, it is known that adding a comparator (i, j) to a network state 
        # reduces the total inversion count by exactly the number of inversions between i and j 
        # in the current state, PROVIDED that we consider the total sum of inversions over all pairs.
        # Let's verify.
        # Total Inv = sum_{a<b} Inv(a,b).
        # Adding (i, j) changes W[i] and W[j].
        # Inv(i, j) becomes 0.
        # Inv(k, i) and Inv(k, j) for k != i, j might change.
        # This makes pure greedy on total inversion count tricky because of side effects.
        
        # Alternative Metric: 
        # Many successful greedy algorithms for sorting networks use "minimum entropy" or 
        # simply pick the comparator that maximizes the number of "misplaced" bits it fixes 
        # relative to a sorted state? No, sorted state is different for each input.
        
        # Let's stick to the simplest robust greedy: 
        # At each step, pick (i, j) that minimizes the total inversion count of the resulting state.
        # To speed this up, we can approximate or use a faster metric.
        # Given 16 wires, there are 120 pairs. 
        # Checking 120 candidates * 16*65536 ops is ~125M ops per step. 
        # If we do 60 steps, that's 7.5B ops. Too slow for Python/NumPy in 90s? 
        # NumPy is fast, but 125M ops per step might take ~50-100ms. 
        # 60 steps * 100ms = 6s. This is feasible!
        
        # Let's implement a function to calculate the total inversion count for a given W_state.
        # Total Inv = sum_{i<j} sum_k (W[i,k] & ~W[j,k])
        
        # We can vectorize this.
        # For a fixed W_state, we can compute the inversion matrix.
        
        pass

    def total_inversions(W_state):
        # Efficient calculation
        # W_state is (N, MAX_INPUTS)
        # We want sum_{i<j} sum_k W_state[i,k] * (1 - W_state[j,k])
        
        # Let's compute the matrix M where M[i,j] = sum_k W[i,k] * (1 - W[j,k])
        # This is (W @ (1-W).T)? No.
        # W is (N, K). 1-W is (N, K).
        # We want sum_k W[i,k] * (1-W[j,k]).
        # This is the element-wise product summed over k.
        # Let A = W (N, K). B = 1-W (N, K).
        # We want C[i,j] = sum_k A[i,k] * B[j,k].
        # This is A @ B.T.
        
        # But we only need i < j.
        # And we need to do this for many candidate states? 
        # No, we need to evaluate the TOTAL inversion count of the state AFTER applying (i,j).
        
        # Applying (i,j) to W_state:
        # W_new[i] = min(W[i], W[j])
        # W_new[j] = max(W[i], W[j])
        # Other wires unchanged.
        
        # So we can compute the new inversion count efficiently by updating only the rows/cols involving i and j.
        # Current Total Inv = T.
        # New Inv = T - Inv(i,j) - (Changes in Inv(k,i) and Inv(k,j))
        
        # This is complex. Let's just recompute the total inversion count for the candidate state.
        # Optimization: Precompute the "potential" of each pair?
        
        # Let's try a simpler heuristic first: 
        # Pick the pair (i, j) that has the highest current inversion count Inv(i, j).
        # This is a "local" greedy. It might not be globally optimal but is very fast.
        # Inv(i, j) = sum_k W[i,k] * (1 - W[j,k]).
        
        # Let's compute the full inversion matrix once per step.
        # W is (N, K).
        # Inv_matrix[i, j] = sum_k W[i,k] * (1 - W[j,k])
        # This is W @ (1-W).T.
        # Shape (N, N).
        
        # Then pick the max off-diagonal element (i < j).
        # Apply it.
        # Repeat.
        
        return None # Placeholder

    # Implementation of the Greedy Strategy
    
    current_W = W.copy()
    network = []
    
    # Pre-compute 1 - W? 
    # We can just use (1 - W) or (~W) & 1.
    
    max_comparators = 100 # Limit to avoid infinite loop if it doesn't converge fast
    # Actually, we should stop when sorted or when no improvement is possible.
    
    for step in range(max_comparators):
        if time.time() > DEADLINE - 5:
            break
            
        # 1. Compute Inversion Matrix
        # Inv[i, j] = number of inputs where W[i]=1 and W[j]=0
        # W is uint8.
        # W_one_hot = W
        # W_zero_hot = 1 - W
        
        # Using matrix multiplication:
        # Inv = W @ (1-W).T
        # (N, K) @ (K, N) -> (N, N)
        
        W_zero = 1 - current_W
        Inv = current_W @ W_zero.T
        # Inv[i, j] is the count.
        
        # We want to pick (i, j) with i < j that maximizes Inv[i, j].
        # Note: Inv[i, j] is the number of inversions between i and j.
        # If Inv[i, j] is 0, adding the comparator does nothing to the inversion count between i and j.
        # It might still help by fixing inversions with other wires? 
        # As discussed, it's safer to assume it only helps if Inv[i, j] > 0.
        
        # Mask out i >= j and i == j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        Inv_masked = np.where(mask, Inv, 0)
        
        max_inv = np.max(Inv_masked)
        
        if max_inv == 0:
            # No inversions left? It should be sorted.
            break
            
        # Find the pair
        max_idx = np.unravel_index(np.argmax(Inv_masked), Inv_masked.shape)
        i, j = max_idx
        
        # Apply comparator (i, j)
        a = current_W[i].copy()
        b = current_W[j].copy()
        current_W[i] = np.minimum(a, b)
        current_W[j] = np.maximum(a, b)
        
        network.append((i, j))
        
        # Check if sorted
        # If sorted, we can stop.
        # Check: all current_W[i] <= current_W[i+1]
        # We can check this periodically to save time, or every step.
        # Checking every step is 15 comparisons of 65k arrays. Very fast.
        is_sorted = np.all(current_W[:-1] <= current_W[1:])
        if is_sorted:
            break

    # Verify the constructed network
    # It's possible the greedy got stuck or didn't finish.
    # We must verify correctness.
    
    def verify_network(net):
        test_W = W.copy()
        for comp in net:
            i, j = comp
            a = test_W[i].copy()
            b = test_W[j].copy()
            test_W[i] = np.minimum(a, b)
            test_W[j] = np.maximum(a, b)
        return np.all(test_W[:-1] <= test_W[1:])
    
    if not verify_network(network):
        # Fallback to a known valid network if greedy failed
        # Use Odd-Even Transposition
        fallback = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    fallback.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    fallback.append((i, i + 1))
        return fallback

    # Optional: Local Search to reduce size further
    # Try removing comparators from the end.
    # This is similar to the current program's strategy but applied to the greedy result.
    
    # Re-compute states for the greedy network
    states = []
    current_W_state = W.copy()
    for comp in network:
        states.append(current_W_state.copy())
        i, j = comp
        a = current_W_state[i].copy()
        b = current_W_state[j].copy()
        current_W_state[i] = np.minimum(a, b)
        current_W_state[j] = np.maximum(a, b)
    
    changed = True
    while changed:
        changed = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > DEADLINE - 2:
                break
            
            # Try removing network[i]
            # Start from states[i] and apply network[i+1:]
            test_W = states[i].copy()
            for comp in network[i+1:]:
                k, l = comp
                a = test_W[k].copy()
                b = test_W[l].copy()
                test_W[k] = np.minimum(a, b)
                test_W[l] = np.maximum(a, b)
            
            if np.all(test_W[:-1] <= test_W[1:]):
                del network[i]
                changed = True
                # states[i] is valid for the new index i as reasoned before
    
    return network

# EVOLVE-BLOCK-END
