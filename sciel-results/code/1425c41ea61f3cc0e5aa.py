import time
import numpy as np
import random

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Initialize A with a valid set (e.g., all even numbers or a dense random set).
    2. Compute initial counts and peak.
    3. Simulated Annealing:
       - Propose a move: add an element, remove an element, or swap an element.
       - Incrementally update counts to calculate the new peak and validity.
       - Accept move if it improves the score (lower peak) or with probability exp(-delta/T) if it worsens.
       - Cool down T over time.
    4. Return the best set found.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    n_elements = 129  # 0..128
    
    # Helper to calculate counts for a given set A (as a boolean array)
    def get_counts(A_bool):
        """
        Returns an array of counts for sums 0..256.
        count[s] = number of unordered pairs {a,b} in A such that a+b=s.
        """
        conv = np.convolve(A_bool, A_bool, mode='full')
        sums = np.arange(257)
        even_mask = (sums % 2 == 0)
        mids = sums[even_mask] // 2
        diag_terms = np.zeros(257, dtype=np.int32)
        diag_terms[even_mask] = A_bool[mids]
        counts = (conv + diag_terms) // 2
        return counts

    # Helper to check validity
    def is_valid(counts):
        return np.all(counts[target_min:target_max+1] > 0)

    # Helper to get peak
    def get_peak(counts):
        return np.max(counts[target_min:target_max+1])

    # --- Phase 1: Initialization ---
    # We need a valid starting set.
    # A simple valid set: All even numbers from 0 to 128?
    # Let's check: Sums of two evens are even. We need to cover odd sums in [128, 256].
    # So we need at least one odd number.
    # Actually, a dense set is safer. Let's start with a random valid set.
    # Or, construct a valid set greedily first to ensure we have a baseline.
    
    A_bool = np.zeros(n_elements, dtype=np.int32)
    
    # Quick greedy construction to get a valid start
    A_set = set()
    # 128 is essential for sum 256
    A_set.add(128)
    A_bool[128] = 1
    counts = get_counts(A_bool)
    
    # Greedy add to cover all sums
    candidates = list(range(129))
    while not is_valid(counts) and len(A_set) < 129:
        best_x = -1
        best_gain = -1
        current_uncovered = np.sum(counts[target_min:target_max+1] == 0)
        
        for x in candidates:
            if x in A_set: continue
            A_bool[x] = 1
            new_counts = get_counts(A_bool)
            new_uncovered = np.sum(new_counts[target_min:target_max+1] == 0)
            gain = current_uncovered - new_uncovered
            if gain > best_gain:
                best_gain = gain
                best_x = x
            A_bool[x] = 0
            
        if best_x == -1:
            break
        A_set.add(best_x)
        A_bool[best_x] = 1
        counts = get_counts(A_bool)
        
    if not is_valid(counts):
        # Fallback if greedy fails (shouldn't happen)
        A_bool[:] = 1
        return list(np.where(A_bool > 0)[0])

    # Now we have a valid set A_bool and counts.
    current_peak = get_peak(counts)
    best_peak = current_peak
    best_A_bool = A_bool.copy()
    
    # --- Phase 2: Simulated Annealing ---
    # We want to minimize the peak.
    # Objective function: Peak multiplicity.
    # We can also penalize invalid sets heavily.
    
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute indices for faster access
    all_indices = np.arange(n_elements)
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = min_T
            
        # Generate a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick a random element not in A
            not_in_A = all_indices[A_bool == 0]
            if len(not_in_A) == 0: continue
            x = random.choice(not_in_A)
            A_bool[x] = 1
            # Update counts incrementally?
            # Adding x adds pairs {x, y} for y in A_bool (including x itself if we just set it).
            # Actually, it's easier to just recompute counts for the affected sums or use the full conv if |A| is small.
            # But for speed, let's use the incremental logic.
            # New pairs involving x:
            # {x, y} where y in A_bool (before adding x) and x <= y? No, unordered.
            # Let's just recompute the whole count array. It's 257 length.
            # np.convolve is very fast.
            new_counts = get_counts(A_bool)
            
            new_peak = get_peak(new_counts)
            valid = is_valid(new_counts)
            
            # Cost function
            if not valid:
                cost = 1000 + new_peak
            else:
                cost = new_peak
                
            old_cost = 1000 + current_peak if not is_valid(counts) else current_peak
            
            delta = cost - old_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                counts = new_counts
                current_peak = new_peak
                if valid and new_peak < best_peak:
                    best_peak = new_peak
                    best_A_bool = A_bool.copy()
            else:
                A_bool[x] = 0
                # counts unchanged

        elif move_type == 'remove':
            in_A = all_indices[A_bool == 1]
            if len(in_A) <= 1: continue
            x = random.choice(in_A)
            A_bool[x] = 0
            new_counts = get_counts(A_bool)
            
            new_peak = get_peak(new_counts)
            valid = is_valid(new_counts)
            
            if not valid:
                cost = 1000 + new_peak
            else:
                cost = new_peak
                
            old_cost = 1000 + current_peak if not is_valid(counts) else current_peak
            
            delta = cost - old_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                counts = new_counts
                current_peak = new_peak
                if valid and new_peak < best_peak:
                    best_peak = new_peak
                    best_A_bool = A_bool.copy()
            else:
                A_bool[x] = 1

        elif move_type == 'swap':
            in_A = all_indices[A_bool == 1]
            not_in_A = all_indices[A_bool == 0]
            if len(in_A) == 0 or len(not_in_A) == 0: continue
            x_out = random.choice(in_A)
            x_in = random.choice(not_in_A)
            
            A_bool[x_out] = 0
            A_bool[x_in] = 1
            
            new_counts = get_counts(A_bool)
            
            new_peak = get_peak(new_counts)
            valid = is_valid(new_counts)
            
            if not valid:
                cost = 1000 + new_peak
            else:
                cost = new_peak
                
            old_cost = 1000 + current_peak if not is_valid(counts) else current_peak
            
            delta = cost - old_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                counts = new_counts
                current_peak = new_peak
                if valid and new_peak < best_peak:
                    best_peak = new_peak
                    best_A_bool = A_bool.copy()
            else:
                A_bool[x_out] = 1
                A_bool[x_in] = 0

        T *= cooling

    # Return the best set found
    return list(np.where(best_A_bool > 0)[0])

# EVOLVE-BLOCK-END
