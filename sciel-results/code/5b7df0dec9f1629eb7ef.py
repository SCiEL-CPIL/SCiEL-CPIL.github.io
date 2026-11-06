# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def _check_validity(mask):
        """
        Vectorized check for B2[2] condition.
        mask: boolean array of size 1001 (index 1..1000)
        Returns True if valid, False otherwise.
        """
        # Get indices of selected elements
        idx = np.nonzero(mask)[0]
        if len(idx) < 2:
            return True
            
        # Compute all pair sums (including i==j)
        # idx[:, None] + idx[None, :] gives a (N, N) matrix of sums
        # We only need upper triangle (i <= j) to match "a <= b" representation
        # However, the count of representations a+b = s with a<=b is exactly the number of pairs (i,j) in upper triangle with idx[i]+idx[j]=s
        
        # To be efficient, we can just compute the full outer sum and count, 
        # but we must be careful: 
        # Rep(s) = count of (i,j) in upper triangle such that idx[i] + idx[j] = s.
        
        # Using np.add.outer on the full set creates a N x N matrix.
        # For N=300, this is 90,000 operations. Very fast.
        sums = np.add.outer(idx, idx)
        
        # We need to count occurrences of each sum s in the upper triangle.
        # Instead of building a 2D histogram which might be slow in Python loops,
        # we can use a 1D histogram of the upper triangle elements.
        
        # Extract upper triangle (including diagonal)
        # np.triu_indices
        iu, ju = np.triu_indices(len(idx))
        upper_sums = sums[iu, ju]
        
        # Count occurrences
        # sums range from 2 to 2000
        # Use np.bincount
        counts = np.bincount(upper_sums, minlength=2001)
        
        return np.all(counts <= 2)

    # Initial Solution: Greedy construction to get a decent start
    # This ensures we start with a valid set, maximizing initial size.
    def _greedy_init():
        mask = np.zeros(1001, dtype=bool)
        counts = np.zeros(2001, dtype=np.int32)
        
        # Standard ascending greedy
        for x in range(1, 1001):
            # Check if adding x is valid
            # 1. 2*x count < 2
            # 2. x+y count < 2 for all y in set
            
            # Efficient check using current counts
            if counts[2*x] >= 2:
                continue
            
            valid = True
            # Iterate through currently chosen elements
            # To speed this up, we can maintain a list of chosen elements
            # But for init, simple loop is fine as N is small initially
            chosen_list = np.nonzero(mask)[0]
            for y in chosen_list:
                if counts[x + y] >= 2:
                    valid = False
                    break
            
            if valid:
                mask[x] = True
                counts[2*x] += 1
                for y in chosen_list:
                    counts[x + y] += 1
        return mask

    # Start with greedy
    current_mask = _greedy_init()
    current_size = np.sum(current_mask)
    best_mask = current_mask.copy()
    best_size = current_size
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    # We will run until deadline
    # Estimate number of steps possible
    # Each step: vectorized check ~ 1ms? 
    # Let's assume ~1000 steps/sec. 
    # Time available: ~130s. So ~130,000 steps.
    
    T = T_start
    # Cooling rate: T_end = T_start * (cooling_rate)^steps
    # Let's target 50,000 steps as a baseline, adjust if too slow/fast
    
    # To be safe, let's just loop until time runs out, with a fixed cooling rate per step.
    # We want to cool down reasonably.
    # If we do 100,000 steps, cooling rate = (0.001/1.0)^(1/100000) ~ 0.99993
    
    cooling_rate = 0.99995
    
    # Precompute candidate indices for random selection
    all_indices = np.arange(1, 1001)
    
    steps = 0
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random element
            x = random.choice(all_indices)
            if current_mask[x]:
                continue
            new_mask = current_mask.copy()
            new_mask[x] = True
        elif move_type < 0.6:
            # Remove a random element
            chosen = np.nonzero(current_mask)[0]
            if len(chosen) == 0:
                continue
            x = random.choice(chosen)
            new_mask = current_mask.copy()
            new_mask[x] = False
        else:
            # Swap: Remove one, Add one
            chosen = np.nonzero(current_mask)[0]
            if len(chosen) == 0:
                continue
            x_remove = random.choice(chosen)
            x_add = random.choice(all_indices)
            if current_mask[x_add]:
                continue
            
            new_mask = current_mask.copy()
            new_mask[x_remove] = False
            new_mask[x_add] = True
            
        # Check validity
        if _check_validity(new_mask):
            new_size = np.sum(new_mask)
            delta_size = new_size - current_size
            
            if delta_size > 0:
                # Always accept improvement
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask.copy()
            else:
                # Accept worse with probability exp(delta/T)
                # delta is negative
                prob = np.exp(delta_size / T)
                if random.random() < prob:
                    current_mask = new_mask
                    current_size = new_size
        
        # Cool down
        T *= cooling_rate
        
        steps += 1
        
        # Safety check: if T is too low, we might get stuck. 
        # But we also want to explore. 
        # If we hit deadline, break.
        
        # Optional: Restart if stuck for too long? 
        # For now, simple SA.
        
        # Check time more frequently to avoid overshooting
        if time.time() > deadline - 5:
            break

    # Convert best_mask to list of integers
    result = np.nonzero(best_mask)[0].tolist()
    
    return result

# EVOLVE-BLOCK-END
