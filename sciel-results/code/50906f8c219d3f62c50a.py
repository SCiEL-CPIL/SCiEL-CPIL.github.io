import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55

# Precompute full matrix of all 2^N boolean inputs
# We store the target sorted state for each input.
# Input x (0 to 2^N - 1) has k ones.
# Output is 0...0 1...1 (k ones at the end).
num_inputs = 1 << N
W_target = np.zeros((num_inputs, N), dtype=np.uint8)

# Calculate popcount for each mask to determine number of 1s
# np.bitwise_count is not available in all numpy versions, use bin().count or lookup
# For 2^18, we can compute popcount efficiently.
popcounts = np.zeros(num_inputs, dtype=np.uint8)
for i in range(num_inputs):
    # Efficient popcount for small integers in a loop might be slow in Python
    # Use vectorized approach or lookup
    pass

# Faster popcount generation
# We can generate W_target directly from the input masks
masks = np.arange(num_inputs, dtype=np.int32)
# For each mask, count bits
# Using a lookup table for 16-bit chunks
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
# Split 18-bit masks into low 8, mid 8, high 2
low = masks & 0xFF
mid = (masks >> 8) & 0xFF
high = (masks >> 16) & 0x3
popcounts = popcount_table[low] + popcount_table[mid] + popcount_table[high]

# Construct W_target
# For input index idx, k = popcounts[idx]
# W_target[idx, :N-k] = 0
# W_target[idx, N-k:] = 1
for k in range(N + 1):
    # Find all indices where popcount is k
    idxs = np.where(popcounts == k)[0]
    if len(idxs) > 0:
        W_target[idxs, N-k:] = 1

# Precompute initial W0 (input states)
W0 = np.zeros((num_inputs, N), dtype=np.uint8)
for i in range(N):
    W0[:, i] = (masks >> i) & 1

# Subset for fast rejection (optional, but full check with array_equal is fast)
# Let's use full check for simplicity and correctness, relying on speed of array_equal.

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    """
    if len(net) > MAX_COMPARATORS:
        return False
    
    W = W0.copy()
    for i, j in net:
        # min = AND, max = OR
        min_val = W[:, i] & W[:, j]
        max_val = W[:, i] | W[:, j]
        W[:, i] = min_val
        W[:, j] = max_val
        
    return np.array_equal(W, W_target)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy Removal
    # We check validity on the FULL set to be safe, but this is expensive.
    # We can check on a subset first?
    # Let's just do full check, it's only 262k rows.
    # 153 comparators * 262k rows * 18 cols ~ 700M ops.
    # In numpy, this is vectorized. 153 steps.
    # Each step is 2 vector ops on 262k uint8s.
    # This is very fast, likely < 1 second for one full verification.
    
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 5:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 5:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Simulated Annealing with In-Place Updates
    if time.time() < deadline - 3:
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        current_count = len(net)
        
        # Precompute W for current net
        W = W0.copy()
        for i, j in net:
            min_val = W[:, i] & W[:, j]
            max_val = W[:, i] | W[:, j]
            W[:, i] = min_val
            W[:, j] = max_val
        
        # Check if current is valid (should be)
        # np.array_equal(W, W_target)
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 2:
            if T < min_T or no_improve_count > max_no_improve:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
                no_improve_count = 0
                # Rebuild W
                W = W0.copy()
                for i, j in net:
                    min_val = W[:, i] & W[:, j]
                    max_val = W[:, i] | W[:, j]
                    W[:, i] = min_val
                    W[:, j] = max_val

            # Generate move
            r = random.random()
            if r < 0.4:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                new_net = net[:idx] + net[idx+1:]
                move_type = 'remove'
                move_idx = idx
                move_i, move_j = i, j
            elif r < 0.7:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
                move_type = 'insert'
                move_idx = idx
                move_i, move_j = i, j
            else:
                # Mutate (Rewire)
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
                move_type = 'mutate'
                move_idx = idx
                move_i, move_j = ni, nj
                old_i, old_j = i, j

            # Apply move to W in-place
            # We need to reverse the effect of the move to rollback?
            # Or just save the old state of the wires involved.
            
            if move_type == 'remove':
                # We need to "undo" the removal? No, we are testing the new_net.
                # The current W corresponds to `net`.
                # `new_net` is `net` without (i,j).
                # We cannot easily "un-apply" a comparator to get the state of `new_net` from `net`'s state.
                # We must apply the network from scratch or maintain a stack?
                # Applying from scratch is O(L * N * 2^N).
                # L is ~100. 100 * 18 * 262k ~ 470M ops.
                # This is too slow for SA steps.
                
                # Alternative:
                # If we remove a comparator, the state of `new_net` is NOT simply derived from `W` by reversing the comparator.
                # Because the comparator $(i,j)$ was applied AFTER previous comparators.
                # To get the state of `new_net`, we would need to re-run the network up to `idx`.
                
                # This implies In-Place SA for *removal* is hard if we don't store intermediate states.
                # However, we can use a "Forward" check for Insert/Mutate if we can derive the new state from the old?
                # If we Insert at end: Yes.
                # If we Insert in middle: No.
                # If we Mutate at end: Yes.
                # If we Mutate in middle: No.
                
                # Given the complexity, let's stick to the **Copy** method but optimize the verification.
                # The copy of 262k * 18 uint8 is ~4.7MB.
                # Copying 4.7MB is ~0.001s?
                # 1000 steps/sec is possible.
                # 60s * 1000 = 60k steps.
                # This should be enough to find a good solution.
                
                # Let's revert to the copy-based SA but ensure it's efficient.
                
                pass

            # Re-implementing SA with Copy for correctness and simplicity
            # We'll just use the _is_sorted_network which copies W0.
            
            if not _is_sorted_network(new_net):
                no_improve_count += 1
                continue
            
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                no_improve_count = 0
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling

    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()
        
    return best_net
