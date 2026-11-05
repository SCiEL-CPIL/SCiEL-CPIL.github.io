"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute mapping from sum value to index in target_sums
    # sum range is [0, 256], so we can use a simple array lookup
    sum_to_idx = np.full(257, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def get_counts(A_set):
        """Compute representation counts for target sums given set A."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = list(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            # Pair (ai, ai)
            s = 2 * ai
            idx = sum_to_idx[s]
            if idx >= 0:
                counts[idx] += 1
            # Pairs (ai, aj) for j > i
            for j in range(i + 1, m):
                aj = A_list[j]
                s = ai + aj
                idx = sum_to_idx[s]
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval."""
        if np.any(counts == 0):
            # Invalid if any sum is not covered.
            # Penalty to guide search away from invalid states.
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    def initial_solution():
        """Create a valid solution using a dense interval and greedy reduction."""
        # Start with a set that guarantees coverage: [0, 128] covers [0, 256]
        # But we want to minimize peak. A balanced set is better.
        # Let's start with a random set of moderate size and ensure validity.
        # A set of size ~60-80 is usually enough to cover [128, 256].
        
        # Strategy: Start with a random subset, check validity.
        # If invalid, add elements greedily to cover uncovered sums.
        
        A = set()
        # Randomly initialize with ~50 elements
        size_init = 50
        A = set(random.sample(range(129), size_init))
        
        counts = get_counts(A)
        if np.any(counts == 0):
            uncovered_indices = np.where(counts == 0)[0]
            # Map indices back to sums
            uncovered_sums = target_sums[uncovered_indices]
            
            candidates = [x for x in range(129) if x not in A]
            random.shuffle(candidates)
            
            for x in candidates:
                if not np.any(counts == 0):
                    break
                    
                # Check if adding x covers any uncovered sum
                # We need to check pairs (x, a) for a in A and (x, x)
                new_counts = counts.copy()
                # Add x to A temporarily for calculation
                # Pairs involving x:
                # 1. (x, x)
                s_xx = 2 * x
                idx_xx = sum_to_idx[s_xx]
                if idx_xx >= 0 and new_counts[idx_xx] == 0:
                    # This helps
                    pass
                
                # Efficient check: does x help cover any 0?
                helps = False
                # Check (x, x)
                s = 2 * x
                idx = sum_to_idx[s]
                if idx >= 0 and new_counts[idx] == 0:
                    helps = True
                
                # Check (x, a) for a in A
                if not helps:
                    for a in A:
                        s = x + a
                        idx = sum_to_idx[s]
                        if idx >= 0 and new_counts[idx] == 0:
                            helps = True
                            break
                
                if helps:
                    A.add(x)
                    # Update counts incrementally? For simplicity in init, recompute if not too slow.
                    # 50 elements -> 1250 pairs. 129 candidates. 129*1250 = 160k ops. Fast.
                    counts = get_counts(A)
        
        return A, calculate_peak(get_counts(A))

    # Find a good starting point
    best_A = set()
    best_peak = 10000
    current_A = set()
    current_peak = 10000
    
    # Try a few random restarts for initialization
    for _ in range(10):
        A_try, peak_try = initial_solution()
        if peak_try < best_peak:
            best_peak = peak_try
            best_A = set(A_try)
            current_A = A_try
            current_peak = peak_try
            
    # If still invalid (shouldn't happen), fallback
    if best_peak > 1000:
        current_A = set(range(129))
        current_peak = calculate_peak(get_counts(current_A))
        best_A = set(current_A)
        best_peak = current_peak

    current_counts = get_counts(current_A)
    
    # Simulated Annealing with Delta Updates
    # Cost: Peak multiplicity.
    # Moves: Add, Remove, Swap.
    
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9995
    
    def get_cost(counts):
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return np.max(counts)
    
    current_cost = get_cost(current_counts)
    
    T = T_start
    steps = 0
    
    # Precompute list of all elements
    all_elements = list(range(129))
    
    while time.time() < deadline - 2.0 and T > T_end:
        steps += 1
        if steps % 100 == 0:
            if current_cost < best_peak:
                best_peak = current_cost
                best_A = set(current_A)
            T *= alpha
            
        # Choose move type
        move_type = random.random()
        new_A = set(current_A)
        
        if move_type < 0.25: # Add
            # Pick random element not in A
            # Optimization: pick from a small subset of candidates to speed up?
            # No, random is fine.
            candidates = [x for x in all_elements if x not in new_A]
            if candidates:
                x = random.choice(candidates)
                new_A.add(x)
        elif move_type < 0.5: # Remove
            # Only remove if set is large enough
            if len(new_A) > 20:
                x = random.choice(list(new_A))
                new_A.remove(x)
        else: # Swap
            if len(new_A) > 1:
                x = random.choice(list(new_A))
                new_A.remove(x)
                candidates = [y for y in all_elements if y not in new_A]
                if candidates:
                    y = random.choice(candidates)
                    new_A.add(y)
        
        if new_A == current_A:
            continue
            
        # Calculate new counts and cost
        # To speed up, we could do delta updates, but get_counts is O(M^2).
        # M is ~50-80. M^2 is ~4000-6400.
        # 50k steps * 6400 ops = 320M ops. Might be tight in 60s in Python.
        # Let's try to optimize get_counts or use delta updates if possible.
        # For now, let's keep get_counts but ensure it's as fast as possible.
        
        new_counts = get_counts(new_A)
        new_cost = get_cost(new_counts)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Improvement
            current_A = new_A
            current_counts = new_counts
            current_cost = new_cost
            if current_cost < best_peak:
                best_peak = current_cost
                best_A = set(current_A)
        else:
            # Accept with probability
            # If delta is large, prob is low
            prob = np.exp(-delta_cost / T)
            if random.random() < prob:
                current_A = new_A
                current_counts = new_counts
                current_cost = new_cost
        
        T *= alpha

    # Final check
    if current_cost < best_peak:
        best_peak = current_cost
        best_A = set(current_A)
        
    # Verify best_A
    final_counts = get_counts(best_A)
    if np.any(final_counts == 0):
        # Fallback
        return list(range(129))
        
    return list(best_A)
# EVOLVE-BLOCK-END
