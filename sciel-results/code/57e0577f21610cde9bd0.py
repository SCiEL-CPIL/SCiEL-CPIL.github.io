"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 128
    min_sum = 128
    max_sum = 256
    
    # Precompute target range
    target_sums = range(min_sum, max_sum + 1)
    
    # Helper to get counts of pair sums
    def get_counts(A):
        """
        Returns a numpy array of counts for sums in [0, max_sum].
        A is a sorted list of integers.
        """
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = list(A)
        k = len(A_list)
        
        # Vectorized approach for speed? 
        # For k ~ 20-40, loops are fine. 
        # Let's stick to efficient loops or numpy tricks if possible.
        # Actually, for k=30, 435 pairs. 129 targets.
        
        for i in range(k):
            ai = A_list[i]
            # Self sum
            s = ai + ai
            if s <= max_sum:
                counts[s] += 1
            
            # Cross sums
            if i + 1 < k:
                # Sums with all j > i
                # A_list[i+1:] + ai
                sums = A_list[i+1:] + ai
                # Filter valid sums
                valid_sums = sums[(sums <= max_sum) & (sums >= min_sum)]
                if len(valid_sums) > 0:
                    # np.bincount is fast
                    # But we need to add to existing counts
                    # We can use np.add.at or just loop if small
                    # Given k is small, a simple loop might be faster than numpy overhead for small arrays
                    # But let's try numpy
                    # Actually, simple loop is safer and likely fast enough for k<50
                    pass 
        
        # Let's use a pure python loop but optimized
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for i in range(k):
            ai = A_list[i]
            s_ii = ai + ai
            if s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, k):
                s_ij = ai + A_list[j]
                if s_ij <= max_sum:
                    counts[s_ij] += 1
        return counts

    def get_cost(A):
        """
        Returns (is_valid, max_multiplicity).
        """
        counts = get_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        
        # Check coverage
        if np.any(target_counts == 0):
            return False, float('inf')
        
        max_mult = np.max(target_counts)
        return True, max_mult

    def greedy_cover(A):
        """
        Add elements to A until all target sums are covered.
        Returns the modified set A.
        """
        A = set(A)
        while time.time() < deadline - 5.0:
            counts = get_counts(sorted(A))
            target_counts = counts[min_sum:max_sum+1]
            uncovered = [s for s in target_sums if target_counts[s - min_sum] == 0]
            
            if not uncovered:
                break
            
            # Find best element to add
            best_x = None
            best_gain = -1
            uncovered_set = set(uncovered)
            
            # Candidate elements: 0 to 128
            # We can restrict candidates to those that might help
            # But iterating 0-128 is cheap (129 iterations * |A| ops)
            for x in range(0, N + 1):
                if x in A:
                    continue
                gain = 0
                for y in A:
                    s = x + y
                    if s in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                # Stuck, pick random
                candidates = [x for x in range(0, N + 1) if x not in A]
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break
            
            A.add(best_x)
            
            # If A gets very large, we might want to prune later
            if len(A) > 50:
                break
                
        return A

    best_A = None
    best_peak = float('inf')
    
    # Strategy:
    # 1. Try to find a valid set with low peak.
    # 2. Start with small sets, increase size if coverage fails or peak is high.
    
    # Theoretical lower bound on size k:
    # Number of pairs = k*(k+1)/2.
    # We need to cover 129 sums.
    # If peak is M, then 129*M >= k*(k+1)/2.
    # For M=2, k*(k+1)/2 <= 258 => k ~ 22.
    # For M=3, k*(k+1)/2 <= 387 => k ~ 27.
    # For M=4, k*(k+1)/2 <= 516 => k ~ 32.
    
    # Let's try sizes from 20 to 40.
    # For each size, run simulated annealing / local search.
    
    random.seed(42)
    
    for k in range(20, 45):
        if time.time() > deadline - 10.0:
            break
            
        # Number of restarts depends on time remaining
        time_left = deadline - time.time() - 10.0
        if time_left <= 0:
            break
            
        # Estimate time per restart
        # get_cost is O(k^2). k=30 -> 900 ops. Very fast.
        # Local search steps: 1000 steps * O(k^2) = 900k ops. ~1ms.
        # So we can do many restarts.
        
        num_restarts = int(min(50, max(5, time_left / 0.1))) # At least 5, at most 50, or based on time
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize A
            # Random sample
            A = set(random.sample(range(0, N + 1), k))
            
            # Greedy cover if needed
            # But if k is small, greedy might add too many.
            # Let's just start with random and use local search that allows size changes?
            # Or fix size k and use swaps.
            
            # If we fix size k, we need to ensure coverage is possible.
            # Let's use a hybrid: 
            # 1. Greedy cover to get a valid set (size might grow).
            # 2. Then local search to minimize peak and shrink size.
            
            # Actually, let's just do local search on the current set A (size k).
            # If it doesn't cover, penalize heavily.
            
            current_A = list(A)
            current_valid, current_peak = get_cost(current_A)
            
            if current_valid:
                best_A = set(current_A)
                best_peak = current_peak
            
            # Local Search / Simulated Annealing
            T = 10.0
            T_min = 0.1
            alpha = 0.99
            
            for step in range(1000):
                if time.time() > deadline - 5.0:
                    break
                    
                # Pick a move
                # 1. Swap
                # 2. Add (if len < 50)
                # 3. Remove (if len > 15)
                
                move_type = random.random()
                
                if move_type < 0.8:
                    # Swap
                    if len(current_A) == 0:
                        continue
                    i = random.randint(0, len(current_A) - 1)
                    old_elem = current_A[i]
                    
                    # Pick new elem
                    new_elem = random.randint(0, N)
                    if new_elem in current_A:
                        continue
                    
                    current_A[i] = new_elem
                    new_valid, new_peak = get_cost(current_A)
                    
                    # Cost function:
                    # If invalid, cost = 1000 + num_uncovered
                    # If valid, cost = peak
                    # We need a comparable cost.
                    # Let's define:
                    # cost = peak if valid, else 100 + (129 - covered_count) * 10
                    
                    # Recalculate cost properly
                    def calc_cost(A_list):
                        counts = get_counts(A_list)
                        target_counts = counts[min_sum:max_sum+1]
                        if np.any(target_counts == 0):
                            uncovered_count = np.sum(target_counts == 0)
                            return 1000 + uncovered_count * 10
                        return np.max(target_counts)
                    
                    old_cost = calc_cost([x for x in current_A if x != new_elem] + [old_elem]) # This is expensive
                    # Better to store old_cost from previous iteration
                    
                    # Let's refactor to store current_cost
                    
                # Refactor loop to store current_cost
                
            # Let's rewrite the inner loop properly
            
            current_A = list(A)
            current_cost, current_valid, current_peak = None, None, None
            
            def update_state(A_list):
                counts = get_counts(A_list)
                target_counts = counts[min_sum:max_sum+1]
                if np.any(target_counts == 0):
                    uncovered_count = np.sum(target_counts == 0)
                    cost = 1000 + uncovered_count * 10
                    return cost, False, float('inf')
                else:
                    peak = np.max(target_counts)
                    return peak, True, peak
            
            current_cost, current_valid, current_peak = update_state(current_A)
            
            if current_valid and current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
            
            for step in range(2000):
                if time.time() > deadline - 5.0:
                    break
                    
                # Pick move
                r = random.random()
                if r < 0.7:
                    # Swap
                    if len(current_A) == 0:
                        continue
                    i = random.randint(0, len(current_A) - 1)
                    old_elem = current_A[i]
                    new_elem = random.randint(0, N)
                    if new_elem in current_A:
                        continue
                    
                    current_A[i] = new_elem
                    new_cost, new_valid, new_peak = update_state(current_A)
                    
                    # Acceptance
                    if new_cost < current_cost or (random.random() < np.exp(-(new_cost - current_cost) / T)):
                        current_cost = new_cost
                        current_valid = new_valid
                        current_peak = new_peak
                        if new_valid and new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(current_A)
                    else:
                        current_A[i] = old_elem
                        
                elif r < 0.85 and len(current_A) < 50:
                    # Add
                    new_elem = random.randint(0, N)
                    if new_elem in current_A:
                        continue
                    current_A.append(new_elem)
                    new_cost, new_valid, new_peak = update_state(current_A)
                    
                    if new_cost < current_cost or (random.random() < np.exp(-(new_cost - current_cost) / T)):
                        current_cost = new_cost
                        current_valid = new_valid
                        current_peak = new_peak
                        if new_valid and new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(current_A)
                    else:
                        current_A.pop()
                        
                elif r >= 0.85 and len(current_A) > 15:
                    # Remove
                    i = random.randint(0, len(current_A) - 1)
                    old_elem = current_A[i]
                    current_A.pop(i)
                    new_cost, new_valid, new_peak = update_state(current_A)
                    
                    if new_cost < current_cost or (random.random() < np.exp(-(new_cost - current_cost) / T)):
                        current_cost = new_cost
                        current_valid = new_valid
                        current_peak = new_peak
                        if new_valid and new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(current_A)
                    else:
                        current_A.insert(i, old_elem)
                        
                T *= alpha
                if T < T_min:
                    T = T_min
                    
            # If we found a very good solution, we can stop early
            if best_peak <= 2:
                return list(best_A)
                
    if best_A is None:
        # Fallback: return a valid set (e.g. range(64,129))
        return list(range(64, 129))
        
    return list(best_A)

# EVOLVE-BLOCK-END
