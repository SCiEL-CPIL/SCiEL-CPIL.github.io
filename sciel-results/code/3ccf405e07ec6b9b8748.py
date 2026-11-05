"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    # Safety margin: Evaluator kills at 90s, task requires <60s.
    # We set deadline to 50s to ensure we return well in time.
    deadline = time.time() + 50.0
    
    # Generate all 2^18 boolean inputs efficiently
    # Using bitwise operations to generate the truth table faster than itertools.product
    # We want rows where each row is a binary representation of i
    # i ranges from 0 to 2^N - 1
    # We can create an array of indices and shift right to get bits
    idx = np.arange(1 << N, dtype=np.uint32)
    # Create a mask for each bit position
    # bits[:, j] = (idx >> j) & 1
    # We can do this in a loop or using broadcasting
    # Broadcasting: idx[:, None] >> np.arange(N) & 1
    inputs = ((idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(np.uint8)
    
    # Pre-allocate a buffer for state to avoid repeated allocation in verify
    state_buffer = np.empty_like(inputs)
    
    def verify(net):
        # Copy inputs to buffer
        np.copyto(state_buffer, inputs)
        
        # Apply network
        # Optimized loop for numpy operations
        for i, j in net:
            # min is AND, max is OR for booleans
            # We need to update state_buffer in place
            # To avoid race conditions in numpy vectorization if we did it in one go, 
            # but since we iterate sequentially, it's fine.
            # However, we must be careful: state_buffer[:, i] and state_buffer[:, j] are views.
            # mi = A & B, ma = A | B
            # We need to compute mi and ma before assigning to avoid overwriting one before the other is read.
            col_i = state_buffer[:, i]
            col_j = state_buffer[:, j]
            state_buffer[:, i] = col_i & col_j
            state_buffer[:, j] = col_i | col_j
        
        # Check sortedness: state[:, k] <= state[:, k+1]
        # This is equivalent to no "10" pattern in the row.
        # state[:, :-1] <= state[:, 1:]
        return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])

    # Start with a known good network (Odd-Even Transposition)
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    if not verify(best_network):
        return best_network

    current = list(best_network)
    current_count = len(current)
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        removed = False
        
        # 1. End-biased Greedy Removal
        # Try to remove comparators, prioritizing those earlier in the network.
        # We scan from the front. If we can remove one, we do it and restart the scan.
        # This is more aggressive than random removal for finding local optima.
        # To avoid O(N^2) verification if the network is large, we limit the scan depth or use randomization.
        # For 18 wires, the network size is small (<100), so systematic scan is feasible.
        
        # We try to remove from index 0 upwards.
        # If we remove one, we restart from 0 because indices shift.
        # To prevent getting stuck in a specific local minimum, we occasionally randomize the start index.
        # But let's try pure front-to-back first.
        
        # Optimization: Only do full scan if we haven't improved recently, or just always do it?
        # Full scan is O(L * V) where L is length and V is verify cost.
        # V is ~262k * 18 ops. L is ~80. 80 * 262k * 18 ~ 376M ops. Might be slow in Python loop?
        # The verify function is vectorized numpy, so it's fast. The loop is Python.
        # 80 iterations of numpy ops on 262kx18 arrays. Should be fast enough.
        
        # Let's try a hybrid: 
        # 1. Try random removals (fast to check, might find easy wins)
        # 2. If no random win, try systematic front-to-back removal.
        
        # Random phase
        for _ in range(5):
            if current_count == 0:
                break
            idx = np.random.randint(current_count)
            candidate = current.copy()
            candidate.pop(idx)
            if verify(candidate):
                current = candidate
                current_count -= 1
                removed = True
                if current_count < best_count:
                    best_network = current
                    best_count = current_count
                    last_improvement_time = time.time()
                break
        
        if removed:
            continue
            
        # Systematic Front-to-Back Removal
        # We iterate through the list. If we remove an item, we restart the loop from 0.
        # This ensures we always try to remove the earliest possible comparator.
        # This can be expensive if we verify many times. 
        # Let's limit the number of verifications in this phase to avoid timeout.
        # Or just do it. The network is small.
        
        i = 0
        while i < current_count:
            if time.time() > deadline:
                break
            candidate = current.copy()
            candidate.pop(i)
            if verify(candidate):
                current = candidate
                current_count -= 1
                # Restart from 0 because indices shifted
                i = 0
                if current_count < best_count:
                    best_network = current
                    best_count = current_count
                    last_improvement_time = time.time()
            else:
                i += 1
        
        if current_count < best_count:
            continue
            
        # If we are stuck (no removals), try mutation
        # Targeted mutation: change a comparator to a nearby one
        if current_count > 0:
            # Pick a random index
            idx = np.random.randint(current_count)
            old_i, old_j = current[idx]
            
            # Generate a candidate mutation
            # Bias towards keeping the same wire or adjacent wires
            new_i = old_i
            new_j = old_j
            
            # With 50% probability, change one of the wires to an adjacent one
            if np.random.rand() < 0.5:
                # Change i
                delta = np.random.choice([-1, 1])
                new_i = old_i + delta
                if new_i < 0 or new_i >= N:
                    new_i = old_i
            else:
                # Change j
                delta = np.random.choice([-1, 1])
                new_j = old_j + delta
                if new_j < 0 or new_j >= N:
                    new_j = old_j
            
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
                
            candidate = current.copy()
            candidate[idx] = (new_i, new_j)
            
            if verify(candidate):
                current = candidate
                # Count doesn't change, but structure does
                # Update best if we want to track structure? 
                # Usually we only care about count. But a better structure might allow future removals.
                # We don't update best_network here unless count improves, 
                # but we keep 'current' as the active search space.
                
        # If we haven't improved in a while, consider a random restart from best_network
        # This helps escape deep local minima
        if time.time() - last_improvement_time > 5.0:
            # Reset to best network
            current = list(best_network)
            current_count = len(current)
            # Maybe perturb it?
            # For now, just reset to let the systematic removal run again on a fresh copy
            # (Though it's the same network, the random mutations might have led us astray)
            # Actually, if we just reset, we'll do the same systematic removal.
            # Let's apply a random shuffle to the comparators in current? 
            # No, order matters for sorting networks? 
            # Wait, sorting networks are usually defined by the set of comparators and their parallel steps.
            # The problem says "ordered list of integer pairs". 
            # Does the order matter? 
            # "Each compare-exchange writes min to wire i and max to j."
            # If the comparators are in parallel steps, the order within a step doesn't matter, 
            # but the order of steps does.
            # The input is a list of pairs. The evaluator applies them sequentially.
            # So the order DOES matter.
            # Shuffling the list would likely break the network.
            # So we cannot just shuffle.
            # We can only remove or change specific comparators.
            
            # So if we reset to best_network, we are back to the known good state.
            # The systematic removal will find the same local optimum.
            # To escape, we need a different mutation strategy.
            # Let's try a "double mutation": change two comparators.
            # Or insert a new comparator? 
            # Inserting increases count, which is bad for the score, but might allow removal of 2 later?
            # Unlikely to be worth it.
            
            # Let's just continue with random mutations. The 5s timeout reset is a safety valve.
            pass

    return best_network

def _odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
