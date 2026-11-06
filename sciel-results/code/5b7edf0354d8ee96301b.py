import time
import numpy as np

N = 16
MAX_TIME = 58.0

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs (65536 x 16 matrix)
    # This is the bottleneck for memory, but 65536*16*8 bytes ~ 8MB, which is fine.
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    # bits[i, j] is the j-th bit of i. 
    # We want wire 0 to be MSB or LSB? 
    # Standard: wire 0 is usually the first input. 
    # Let's define wire 0 as the MSB for the index, so input 1 is [1, 0, ..., 0].
    # Actually, it doesn't matter for sorting as long as it's consistent.
    # Let's use: bit 0 is LSB of index? No, let's stick to standard:
    # index 0: 0000...0
    # index 1: 0000...1 (wire 15 if LSB is wire 15?)
    # Let's define: wire i corresponds to bit i of the index.
    # So index = sum(bit_i * 2^i).
    # This means wire 0 is the least significant bit.
    bits = (indices[:, None] >> np.arange(N)) & 1
    
    # A network is valid if for all inputs, the output is sorted (non-decreasing).
    # i.e., out[:, i] <= out[:, i+1] for all i.
    # Violation: count of (i, input) such that out[:, i] > out[:, i+1].
    
    def count_violations(comps):
        """
        Returns the number of sortedness violations across all inputs.
        0 means valid.
        """
        if not comps:
            # Initial state: unsorted
            # Count violations in raw bits
            diffs = np.diff(bits, axis=1)
            return np.sum(diffs < 0)
        
        vec = bits.copy()
        for i, j in comps:
            min_val = np.minimum(vec[:, i], vec[:, j])
            max_val = np.maximum(vec[:, i], vec[:, j])
            vec[:, i] = min_val
            vec[:, j] = max_val
        
        diffs = np.diff(vec, axis=1)
        return np.sum(diffs < 0)
    
    # Greedy Construction
    # Start with an empty network or a very small one?
    # Starting with empty is slow because violations are high and many moves reduce them.
    # Starting with a "bad" network (e.g. random) might be faster to converge?
    # Let's start with a known good structure but broken?
    # Actually, let's start with an empty list and add comparators.
    # But 60 steps of greedy selection over 120 possible comparators = 7200 evaluations.
    # Each evaluation is O(N * 65536) ~ 1M ops.
    # 7200 * 1M = 7.2B ops. Too slow in Python/Numpy loop?
    # Numpy is fast, but 7200 calls to count_violations might be too many.
    
    # Optimization: 
    # 1. Use a subset of inputs for the greedy selection (e.g., 1024 random inputs).
    # 2. Only check full validity at the end of a phase or periodically.
    
    # Let's use a sample of 2048 inputs for the greedy guide.
    sample_size = 2048
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_inputs, sample_size, replace=False)
    sample_bits = bits[sample_indices]
    
    def count_violations_sample(comps):
        if not comps:
            diffs = np.diff(sample_bits, axis=1)
            return np.sum(diffs < 0)
        
        vec = sample_bits.copy()
        for i, j in comps:
            min_val = np.minimum(vec[:, i], vec[:, j])
            max_val = np.maximum(vec[:, i], vec[:, j])
            vec[:, i] = min_val
            vec[:, j] = max_val
        
        diffs = np.diff(vec, axis=1)
        return np.sum(diffs < 0)
    
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    current_comps = []
    current_violations = count_violations_sample(current_comps)
    
    # Greedy Loop
    while time.time() < deadline - 5.0: # Leave 5s for final checks and potential local search
        # Try to add a comparator that reduces violations the most
        best_comp = None
        best_violations = current_violations
        best_delta = 0
        
        # We can vectorize the evaluation of all possible comparators?
        # For each comp (i, j), apply it to sample_bits, count violations.
        # This is 120 * (2048 * 16) ops. ~ 4M ops. Very fast.
        
        # However, applying comps sequentially in a loop is slow.
        # Let's just iterate through all_comps.
        
        # Optimization: If we have many comps, the "current" state is complex.
        # We must apply all current comps + candidate comp.
        # This is O(len(comps) * N * sample_size).
        # If len(comps) is 30, that's 30 * 16 * 2048 ~ 1M ops per candidate.
        # 120 candidates * 1M = 120M ops per step.
        # 60 steps * 120M = 7.2B ops. Might be too slow.
        
        # Alternative: 
        # 1. Generate a baseline network (Batcher) quickly.
        # 2. Use the sample-based local search to reduce it.
        # This is safer.
        
        break # Skip greedy from scratch, use local search on Batcher
    
    # Re-evaluating strategy:
    # Greedy from scratch is risky on time.
    # Local search on Batcher (63 comps) is proven to work (score 1.39).
    # How to make it structurally different and better?
    # 1. Use the sample-based violation count for the search moves, not just full validity.
    #    This allows accepting moves that reduce violations even if they don't immediately
    #    lead to a valid network, or rejecting moves that increase violations.
    #    Actually, for reduction (removing comps), we need the network to REMAIN valid.
    #    So we can't accept invalid intermediates.
    #    
    #    BUT, if we use a "repair" strategy:
    #    1. Remove a comp.
    #    2. If invalid, try to find a replacement comp that fixes the violations with minimal cost.
    #    
    #    Let's try a different structural change: 
    #    **Simulated Annealing on the Violation Count**
    #    Instead of requiring validity for every step, we minimize the violation count.
    #    We start with a random network (or Batcher) and try to drive violations to 0.
    #    Moves:
    #    - Add a comp
    #    - Remove a comp
    #    - Swap a comp
    #    Acceptance: 
    #    - If violations decrease, accept.
    #    - If violations increase, accept with probability exp(-delta / T).
    #    This allows escaping local minima in the "valid" space by going through invalid states.
    #    
    #    This is a significant structural change: optimizing a continuous (integer) error surface
    #    rather than a binary constraint satisfaction problem.
    
    # Implementation of SA on Violations
    
    # Start with Batcher
    def _generate_batcher_network():
        def merge(n, offset):
            if n == 1:
                return [(offset, offset + 1)]
            
            h = n // 2
            step1 = [(offset + i, offset + i + n) for i in range(n)]
            
            def strided_merge(n, offset, stride):
                if n == 1:
                    return [(offset, offset + stride)]
                
                h = n // 2
                step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
                even = strided_merge(h, offset, 2 * stride)
                odd = strided_merge(h, offset + stride, 2 * stride)
                return step1 + even + odd
            
            even_merge = strided_merge(h, offset, 2)
            odd_merge = strided_merge(h, offset + 1, 2)
            
            return step1 + even_merge + odd_merge

        def sort(n, offset):
            if n <= 1:
                return []
            h = n // 2
            left = sort(h, offset)
            right = sort(h, offset + h)
            m = merge(h, offset)
            return left + right + m

        return sort(N, 0)

    initial_comps = _generate_batcher_network()
    
    # Verify initial
    if count_violations(initial_comps) != 0:
        # Fallback to OET
        initial_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    initial_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    initial_comps.append((j, j + 1))
    
    current_comps = initial_comps[:]
    current_violations = count_violations_sample(current_comps)
    
    best_comps = current_comps[:]
    best_violations = current_violations
    best_len = len(best_comps)
    
    T = 10.0 # Temperature
    T_min = 0.1
    alpha = 0.99
    
    rng = np.random.default_rng(123)
    
    # Main SA Loop
    while time.time() < deadline - 2.0:
        if T < T_min:
            T = T_min
            
        # Generate a move
        move_type = rng.random()
        
        new_comps = current_comps[:]
        
        if move_type < 0.5:
            # Remove a random comparator
            if len(new_comps) > 10: # Don't go below 10
                idx = rng.integers(0, len(new_comps))
                new_comps.pop(idx)
        elif move_type < 0.8:
            # Add a random comparator
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp not in new_comps:
                idx = rng.integers(0, len(new_comps) + 1)
                new_comps.insert(idx, new_comp)
            else:
                # If it exists, do nothing or swap
                continue
        else:
            # Swap a random comparator
            if len(new_comps) > 1:
                idx = rng.integers(0, len(new_comps))
                old_comp = new_comps[idx]
                new_comp = all_comps[rng.integers(0, len(all_comps))]
                if new_comp != old_comp:
                    new_comps[idx] = new_comp
                else:
                    continue
        
        # Evaluate new state
        new_violations = count_violations_sample(new_comps)
        
        delta = new_violations - current_violations
        
        if delta < 0:
            # Improvement
            current_comps = new_comps
            current_violations = new_violations
            
            # Check if it's a valid network (full check)
            # We only do full check if sample violations are 0
            if new_violations == 0:
                if count_violations(new_comps) == 0:
                    if len(new_comps) < best_len:
                        best_comps = new_comps[:]
                        best_len = len(new_comps)
        else:
            # Accept with probability
            if delta == 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            
            if rng.random() < prob:
                current_comps = new_comps
                current_violations = new_violations
        
        T *= alpha
        
    # Final Polish: Try to remove comparators from the best found valid network
    # using the fast full check
    if time.time() < deadline - 0.5:
        polish_comps = best_comps[:]
        for _ in range(100): # Max 100 attempts
            if time.time() > deadline - 0.5:
                break
            if len(polish_comps) <= 50: # Don't go too low
                break
            idx = rng.integers(0, len(polish_comps))
            removed = polish_comps.pop(idx)
            if count_violations(polish_comps) == 0:
                best_comps = polish_comps[:]
                best_len = len(best_comps)
            else:
                polish_comps.insert(idx, removed)
    
    return best_comps

# EVOLVE-BLOCK-END
