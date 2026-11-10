import random
import numpy as np

def construct():
    # Parameters
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGETS = range(TARGET_MIN, TARGET_MAX + 1)
    
    # Helper to compute representation counts and validity
    def compute_stats(A_set):
        # A_set is a set of ints
        # Returns (is_valid, max_count, total_count)
        counts = [0] * (TARGET_MAX + 1)
        A_list = list(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]: # Unordered pairs, a <= b
                s = a + b
                if TARGET_MIN <= s <= TARGET_MAX:
                    counts[s] += 1
        
        is_valid = all(counts[s] > 0 for s in TARGETS)
        max_count = max(counts[s] for s in TARGETS)
        total_count = sum(counts[s] for s in TARGETS)
        return is_valid, max_count, total_count

    # 1. Greedy Initialization
    # We want to start with a valid set.
    # Greedy: Add elements that cover the most uncovered sums.
    A_set = set()
    uncovered = set(TARGETS)
    
    # Candidate pool is [0, 128]
    candidates = list(range(129))
    
    while uncovered:
        best_candidate = None
        best_coverage = -1
        
        for x in candidates:
            if x in A_set:
                continue
            
            # Calculate how many uncovered sums x would cover
            # A sum s is covered by x if s - x is in A_set OR s - x == x (i.e., 2x=s)
            # Actually, if we add x, new pairs are (x, a) for a in A_set and (x, x).
            covered_now = 0
            for s in uncovered:
                other = s - x
                if other == x:
                    # Pair (x,x)
                    covered_now += 1
                elif other in A_set:
                    # Pair (x, other)
                    covered_now += 1
                # Note: if other < 0 or other > 128, it's not in A_set anyway.
                
            if covered_now > best_coverage:
                best_coverage = covered_now
                best_candidate = x
        
        if best_candidate is None or best_coverage == 0:
            # Greedy failed to find a single element that covers any new sum?
            # This shouldn't happen if we have enough candidates, but just in case.
            # Force add a random candidate
            best_candidate = random.choice(candidates)
            
        A_set.add(best_candidate)
        
        # Update uncovered
        new_uncovered = set()
        for s in uncovered:
            other = s - best_candidate
            if other == best_candidate or other in A_set:
                # s is now covered
                pass
            else:
                new_uncovered.add(s)
        uncovered = new_uncovered
        
        # Optional: Prune candidates that are no longer useful? No, keep all.
        
    # Now A_set is valid.
    is_valid, current_peak, current_total = compute_stats(A_set)
    
    # 2. Local Search / Simulated Annealing
    # Try to reduce the peak multiplicity.
    # Moves: Swap an element in A with one not in A.
    # We also try removing elements if possible (size reduction helps reduce counts).
    
    all_elements = set(range(129))
    
    best_peak = current_peak
    best_A = A_set.copy()
    best_total = current_total
    
    # Simulated Annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    steps = 20000 # Adjust based on time budget
    
    current_A = A_set.copy()
    current_peak, current_total = compute_stats(current_A)[1], compute_stats(current_A)[2]
    
    for step in range(steps):
        if T < T_min:
            break
            
        # Choose a move
        # 1. Swap: pick x in current_A, y in all_elements - current_A
        # 2. Remove: pick x in current_A (only if valid after removal)
        # 3. Add: pick y in all_elements - current_A
        
        move_type = random.random()
        
        if move_type < 0.7: # Swap
            if not current_A: continue
            x = random.choice(list(current_A))
            y = random.choice(list(all_elements - current_A))
            
            new_A = current_A.copy()
            new_A.remove(x)
            new_A.add(y)
            
            is_valid, new_peak, new_total = compute_stats(new_A)
            
            if not is_valid:
                # Reject invalid
                continue
                
            delta_peak = new_peak - current_peak
            if delta_peak < 0:
                # Improvement
                current_A = new_A
                current_peak = new_peak
                current_total = new_total
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = current_A.copy()
                    best_total = current_total
            else:
                # Worse or equal peak
                # Accept with probability exp(-delta / T) if delta is small
                # We care about peak, so if peak increases, reject mostly.
                # If peak is same, accept if total decreases.
                if delta_peak == 0 and new_total < current_total:
                    current_A = new_A
                    current_peak = new_peak
                    current_total = new_total
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = current_A.copy()
                        best_total = current_total
                elif delta_peak == 0:
                    # Neutral move, accept with small prob to explore
                    if random.random() < 0.1:
                        current_A = new_A
                        current_total = new_total
                # else reject
                
        elif move_type < 0.85: # Remove
            if not current_A: continue
            x = random.choice(list(current_A))
            
            new_A = current_A.copy()
            new_A.remove(x)
            
            is_valid, new_peak, new_total = compute_stats(new_A)
            
            if not is_valid:
                continue
                
            delta_peak = new_peak - current_peak
            if delta_peak <= 0:
                current_A = new_A
                current_peak = new_peak
                current_total = new_total
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = current_A.copy()
                    best_total = current_total
            else:
                # Removing an element that reduces peak? Unlikely if valid.
                # But if it increases peak, reject.
                pass
                
        else: # Add
            y = random.choice(list(all_elements - current_A))
            
            new_A = current_A.copy()
            new_A.add(y)
            
            is_valid, new_peak, new_total = compute_stats(new_A)
            
            if not is_valid:
                continue
                
            delta_peak = new_peak - current_peak
            if delta_peak < 0:
                current_A = new_A
                current_peak = new_peak
                current_total = new_total
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = current_A.copy()
                    best_total = current_total
            else:
                # Adding usually increases or keeps peak.
                # Only accept if peak stays same and total decreases? No, adding increases total.
                # So reject unless it helps future swaps? 
                # For now, reject if peak doesn't decrease.
                pass

        T *= cooling
        
    # Return the best set found
    return list(best_A)
