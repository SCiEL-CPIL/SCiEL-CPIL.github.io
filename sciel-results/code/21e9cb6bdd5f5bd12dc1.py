import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s kill

    N = 1000

    def sum_counts_from_mask(mask):
        # mask: bool array length N+1 (index 0 unused)
        # returns counts of a+b with a<=b, for sums 2..2N
        m = mask.astype(np.int32)
        c = np.convolve(m, m, mode='full')  # length 2N+1, index s
        s = np.arange(2 * N + 1)
        diag = np.zeros(2 * N + 1, dtype=np.int32)
        even = (s % 2 == 0)
        half = s // 2
        # mask index must be within [0,N]
        valid = even & (half <= N)
        diag[valid] = m[half[valid]]
        counts = (c + diag) // 2
        return counts

    def is_valid(mask):
        counts = sum_counts_from_mask(mask)
        return np.max(counts) <= 2

    # 1. Deterministic Greedy Construction: Start with empty set, add smallest valid number
    # This provides a very strong, consistent baseline.
    def deterministic_greedy():
        mask = np.zeros(N + 1, dtype=bool)
        for x in range(1, N + 1):
            mask[x] = True
            if not is_valid(mask):
                mask[x] = False
        return mask

    best_mask = deterministic_greedy()
    best_size = int(np.sum(best_mask))
    
    # 2. Simulated Annealing
    # Objective: Maximize size. 
    # Moves: Add, Remove, Swap.
    # We track the current mask and its size.
    
    current_mask = best_mask.copy()
    current_size = best_size
    current_counts = sum_counts_from_mask(current_mask)
    
    # Parameters for Simulated Annealing
    T_start = 1.0
    T_end = 0.01
    # We will run until deadline
    # We need to estimate steps per second to set total steps, or just loop until time.
    
    rng = random.Random(42)
    
    # Precompute present list for faster sampling? 
    # Using np.where is O(N). With N=1000 it's fast.
    
    t0 = time.time()
    total_time_budget = deadline - t0
    
    # We don't know exactly how many steps we can take, so we loop while time remains.
    # To make SA effective, we need a schedule. 
    # Let's assume we can do ~1000-5000 steps per second.
    # Let's just loop and decrement T based on elapsed time fraction.
    
    while time.time() < deadline - 2:
        elapsed = time.time() - t0
        progress = min(1.0, elapsed / total_time_budget)
        # Exponential decay of temperature
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a move
        move_type = rng.random()
        if move_type < 0.4:
            # Add move
            # Pick a random number not in mask
            # To pick efficiently, we can just pick random 1..N and check if in mask.
            # Or maintain a list of absent numbers.
            # Given N=1000, picking random and checking is O(1) on average if density is ~0.3-0.5.
            b = rng.randint(1, N)
            if current_mask[b]:
                continue
            # Try adding b
            current_mask[b] = True
            new_counts = sum_counts_from_mask(current_mask)
            if np.max(new_counts) <= 2:
                # Valid move
                new_size = current_size + 1
                # Always accept valid moves that increase size (greedy hill climb)
                # But in SA, we might want to accept size-decreasing moves?
                # Here, "size" is the objective. 
                # If we add, size increases. If we remove, size decreases.
                # If we swap, size stays same.
                # We want to maximize size.
                # So, adding is always good if valid.
                # Removing is bad (decreases size) but might allow future adds.
                # Swapping is neutral in size, but might allow future adds.
                
                # Accept if it improves or is neutral? 
                # Actually, if we add, size increases by 1. This is strictly better.
                # So we always accept valid adds.
                current_size = new_size
                current_counts = new_counts
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask.copy()
            else:
                # Invalid, revert
                current_mask[b] = False
                
        elif move_type < 0.7:
            # Remove move
            # Pick a random number in mask
            present = np.where(current_mask)[0]
            present = present[present >= 1]
            if len(present) == 0:
                continue
            a = present[rng.randrange(len(present))]
            current_mask[a] = False
            new_counts = sum_counts_from_mask(current_mask)
            new_size = current_size - 1
            
            # This move decreases size. We only accept if it helps escape local optima.
            # Probability of acceptance: exp(-delta_E / T)
            # Here delta_E = -1 (size decreases by 1). 
            # So we accept with prob exp(1/T). Since T < 1, exp(1/T) is large? 
            # Wait, standard SA minimizes energy. Here we maximize size.
            # Let's define Energy = -Size.
            # Move reduces size by 1 -> Energy increases by 1.
            # Accept with prob exp(-1/T).
            
            if rng.random() < np.exp(-1.0 / T):
                current_size = new_size
                current_counts = new_counts
            else:
                # Revert
                current_mask[a] = True
                # current_counts stays same as before move? 
                # We calculated new_counts, but we revert, so we keep old current_counts.
                # Actually, we didn't update current_counts if we revert.
                # So we just need to revert the mask.
                
        else:
            # Swap move
            # Pick a in mask, b not in mask
            present = np.where(current_mask)[0]
            present = present[present >= 1]
            if len(present) == 0:
                continue
            a = present[rng.randrange(len(present))]
            
            # Pick b not in mask
            # Retry a few times if we pick one in mask
            b = rng.randint(1, N)
            attempts = 0
            while current_mask[b] and attempts < 10:
                b = rng.randint(1, N)
                attempts += 1
            if current_mask[b]:
                continue
                
            # Try swap
            current_mask[a] = False
            current_mask[b] = True
            new_counts = sum_counts_from_mask(current_mask)
            if np.max(new_counts) <= 2:
                # Valid swap. Size unchanged.
                # Accept with prob 1.0? Or explore?
                # Since size is same, we accept to explore landscape.
                # This allows us to move to a different configuration of same size,
                # which might allow future adds.
                current_counts = new_counts
                # Size is same, so best_size doesn't change unless we later add.
            else:
                # Invalid, revert
                current_mask[a] = True
                current_mask[b] = False

    # Final check
    if best_mask is None:
        return []
    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
