import numpy as np
import random
import time
import math

def construct():
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    n_classes = len(divisors)
    
    # Pre-slice indices for each divisor to speed up access?
    # Slicing in numpy is O(1) to create the view, but O(N/m) to iterate.
    # We can't precompute the views easily because the residue changes.
    # But we can precompute the step size.
    
    def init_random():
        return [random.randint(0, m - 1) for m in divisors]

    def simulate_annealing(time_limit=110):
        residues = init_random()
        
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(divisors):
            r = residues[i]
            counts[r::m] += 1
            
        unc = int(np.sum(counts == 0))
        best_unc = unc
        best_residues = residues.copy()
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.9999
        start_time = time.time()
        
        while (time.time() - start_time) < time_limit:
            # Pick a random class
            i = random.randint(0, n_classes - 1)
            m = divisors[i]
            r_old = residues[i]
            
            # Pick a random new residue different from old
            if m <= 1:
                continue
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            # Calculate delta
            # Remove old
            old_slice = slice(r_old, None, m)
            new_slice = slice(r_new, None, m)
            
            counts[old_slice] -= 1
            d1 = int(np.sum(counts[old_slice] == 0))
            
            counts[new_slice] += 1
            d2 = int(np.sum(counts[new_slice] > 0)) # Wait, d2 is zeros removed?
            # Before add: counts[new_slice] has some zeros.
            # After add: those zeros become 1.
            # So d2 should be the number of zeros in new_slice BEFORE adding.
            # I added 1 already. So I need to check if it was 0 before.
            # It was 0 before if it is 1 now and was 0? No.
            # Let's revert the logic.
            
            # Correct Delta Logic:
            # 1. counts[old_slice] -= 1
            # 2. zeros_gained = sum(counts[old_slice] == 0)
            # 3. counts[new_slice] += 1
            # 4. zeros_lost = sum((counts[new_slice] - 1) == 0)  # i.e. was 0, now 1
            #    OR simpler: count zeros in new_slice BEFORE increment.
            
            # Let's redo the update carefully.
            # Current state: counts reflects residues.
            
            # Step 1: Tentative remove old
            counts[old_slice] -= 1
            zeros_gained = int(np.sum(counts[old_slice] == 0))
            
            # Step 2: Tentative add new
            # We need to know how many zeros are in new_slice in the state AFTER removing old but BEFORE adding new.
            # Let's call this state S_mid.
            # zeros_lost = sum(counts[new_slice] == 0) in S_mid.
            zeros_lost = int(np.sum(counts[new_slice] == 0))
            
            counts[new_slice] += 1
            
            delta_unc = zeros_gained - zeros_lost
            new_unc = unc + delta_unc
            
            # Acceptance
            if delta_unc < 0 or random.random() < math.exp(-delta_unc / T):
                # Accept
                unc = new_unc
                residues[i] = r_new
                if unc < best_unc:
                    best_unc = unc
                    best_residues = residues.copy()
            else:
                # Revert
                counts[old_slice] += 1
                counts[new_slice] -= 1
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_residues, best_unc

    best_global_unc = base
    best_global_res = []
    
    # Run multiple restarts to find a good basin
    # Estimate time per run
    # Let's run for 10 seconds each, max 10 runs?
    # Or just one long run? SA benefits from restarts.
    # Let's do 5 runs of 20s each.
    
    for _ in range(5):
        res, unc = simulate_annealing(time_limit=20)
        if unc < best_global_unc:
            best_global_unc = unc
            best_global_res = res
    
    classes = [[divisors[i], best_global_res[i]] for i in range(n_classes)]
    return classes
