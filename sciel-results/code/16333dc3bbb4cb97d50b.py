import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 128
    target_lo = 128
    target_hi = 256
    
    def compute_peak(A_set):
        """Compute representation counts for all targets and return peak."""
        reps = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                s = a + b
                if target_lo <= s <= target_hi:
                    reps[s - target_lo] += 1
        return int(np.max(reps))
    
    def compute_reps(A_set):
        """Compute representation counts array."""
        reps = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:
                s = a + b
                if target_lo <= s <= target_hi:
                    reps[s - target_lo] += 1
        return reps
    
    def is_valid(A_set):
        """Check all targets have at least one representation."""
        reps = compute_reps(A_set)
        return bool(np.all(reps > 0))
    
    # Initial candidate: use a structured set
    # Try different sizes and structures
    
    best_A = None
    best_peak = float('inf')
    
    # Strategy: try several random initializations, then local search
    random.seed(42)
    
    sizes_to_try = [70, 75, 80, 85, 90]
    
    for size in sizes_to_try:
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        A_init = set(random.sample(range(0, n + 1), size))
        
        # Ensure validity: if not valid, try to fix by adding elements
        # For now, just evaluate
        if is_valid(A_init):
            peak = compute_peak(A_init)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A_init)
        
        # Local search: try swapping elements to reduce peak
        A = set(A_init)
        reps = compute_reps(A)
        current_peak = int(np.max(reps))
        
        if not is_valid(A):
            # Try to make it valid by adding elements
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                # Find targets with 0 reps
                zero_targets = [target_lo + i for i in range(len(reps)) if reps[i] == 0]
                if not zero_targets:
                    break
                # Add a random element that helps
                improved = False
                for _ in range(50):
                    new_el = random.randint(0, n)
                    if new_el in A:
                        continue
                    A_new = A | {new_el}
                    if len(A_new) > size + 10:
                        continue
                    reps_new = compute_reps(A_new)
                    if np.all(reps_new > 0):
                        A = A_new
                        reps = reps_new
                        current_peak = int(np.max(reps))
                        improved = True
                        break
                if not improved:
                    break
        
        # Now do local search: swap operations
        A_list = sorted(A)
        peak = int(np.max(reps))
        
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            # Try random swaps
            if len(A) < 2:
                break
            
            a_idx = random.randint(0, len(A) - 1)
            # Pick a new element not in A
            new_el = random.randint(0, n)
            if new_el in A:
                continue
            
            A_old = A_list[a_idx]
            A_new_set = (A - {A_old}) | {new_el}
            
            if not is_valid(A_new_set):
                continue
            
            reps_new = compute_reps(A_new_set)
            new_peak = int(np.max(reps_new))
            
            if new_peak < peak:
                A = A_new_set
                reps = reps_new
                peak = new_peak
                A_list = sorted(A)
                if peak < best_peak:
                    best_peak = peak
                    best_A = set(A)
            elif new_peak == peak and random.random() < 0.3:
                # Accept with some probability for exploration
                A = A_new_set
                reps = reps_new
                A_list = sorted(A)
        
        if time.time() > deadline - 5:
            break
    
    # If no valid set found, fall back to seed
    if best_A is None or not is_valid(best_A):
        return list(range(64, 129))
    
    return sorted(best_A)
