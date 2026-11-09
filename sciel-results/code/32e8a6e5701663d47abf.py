import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s margin for 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        # A is a list or np.array of integers
        if len(A) == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        A_arr = np.array(A, dtype=np.int32)
        # Create outer sum matrix
        # sums[i, j] = A[i] + A[j]
        sums = np.add.outer(A_arr, A_arr)
        
        # We only care about i <= j for unordered pairs with repetition
        # The upper triangle (including diagonal) contains each pair once
        # Note: np.triu_indices returns indices for i <= j if we want upper? 
        # Actually, standard convention: 
        # Pair (a, b) is same as (b, a). 
        # If we take upper triangle of the sum matrix, we get each unique combination {a, b} exactly once.
        # Let's verify: 
        # A = [1, 2]. sums = [[2, 3], [3, 4]].
        # Upper triangle (i<=j): (0,0)->2, (0,1)->3, (1,1)->4.
        # Pairs: (1,1)->2, (1,2)->3, (2,2)->4. Correct.
        
        i_idx, j_idx = np.triu_indices(len(A_arr))
        valid_sums = sums[i_idx, j_idx]
        
        # Filter sums in range [128, 256]
        mask = (valid_sums >= N_LOW) & (valid_sums <= N_HIGH)
        valid_sums = valid_sums[mask]
        
        # Count occurrences
        # Map to 0..127
        indices = valid_sums - N_LOW
        if len(indices) == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
            
        reps = np.bincount(indices, minlength=RANGE_SIZE)
        return reps.astype(np.int32)

    def get_objective(reps):
        # Objective: Minimize peak, penalize uncovered
        # If all covered, penalty is 0.
        # If some uncovered, we want to fix them.
        # A large penalty ensures coverage is prioritized, but allows movement.
        uncovered_count = np.sum(reps == 0)
        if uncovered_count == 0:
            peak = int(np.max(reps))
            # Small tie-breaker: minimize total representations? Or just peak.
            # Let's stick to peak.
            return peak
        else:
            # Heavy penalty for uncovered sums
            # Peak might be 0 or small if uncovered.
            # We want to drive uncovered to 0.
            # Return a large number so it's worse than any valid peak.
            # Max possible peak for K=100 is ~5000.
            # Let's return 1000 + uncovered_count * 100
            return 1000 + uncovered_count * 100

    # Phase 1: Initialize
    best_A = None
    best_obj = float('inf')
    best_peak = float('inf')
    
    # Try a few random initializations
    for _ in range(20):
        # Random set size 40-80
        k = random.randint(40, 80)
        A_init = set(random.sample(range(0, 129), k))
        # Force 0 and 128 to help with edges
        A_init.add(0)
        A_init.add(128)
        
        reps = calc_reps(list(A_init))
        obj = get_objective(reps)
        
        if obj < best_obj:
            best_obj = obj
            best_A = list(A_init)
            if obj < 1000: # Valid
                best_peak = int(np.max(reps))

    if best_A is None:
        # Fallback
        return list(range(64, 129))

    current_A = set(best_A)
    current_reps = calc_reps(list(current_A))
    current_obj = get_objective(current_reps)
    
    # Simulated Annealing
    T = 100.0
    cooling_rate = 0.995
    min_T = 1.0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if T < min_T or no_improve_count > max_no_improve:
            break
            
        # Generate move
        # 1. Swap
        # 2. Remove
        # 3. Add
        
        move_type = random.random()
        
        new_A = set(current_A)
        
        if move_type < 0.7:
            # Swap
            if len(new_A) == 0:
                continue
            item_to_remove = random.choice(list(new_A))
            new_item = random.randint(0, 128)
            new_A.remove(item_to_remove)
            new_A.add(new_item)
        elif move_type < 0.85:
            # Remove
            if len(new_A) == 0:
                continue
            item_to_remove = random.choice(list(new_A))
            new_A.remove(item_to_remove)
        else:
            # Add
            new_item = random.randint(0, 128)
            new_A.add(new_item)
            
        if new_A == current_A:
            continue
            
        new_reps = calc_reps(list(new_A))
        new_obj = get_objective(new_reps)
        
        delta_obj = new_obj - current_obj
        
        accept = False
        if delta_obj < 0:
            accept = True
        else:
            # Accept worse with probability exp(-delta/T)
            # If T is small, exp(-large/T) is ~0
            prob = np.exp(-delta_obj / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_A = new_A
            current_reps = new_reps
            current_obj = new_obj
            
            # Update best
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = list(current_A)
                if current_obj < 1000:
                    best_peak = int(np.max(current_reps))
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        T *= cooling_rate
        
    # Final check: if best_A is not covered, try to repair or fallback?
    # The objective function penalizes uncovered, so best_A should be covered if possible.
    # But if it's not covered, it's an invalid solution for the task "Every integer... must be a sum".
    # We must return a valid A.
    # If best_A is invalid, we need a valid one.
    # Let's check validity of best_A.
    final_reps = calc_reps(best_A)
    if np.any(final_reps == 0):
        # Fallback to a known valid structure or try to repair quickly
        # Simple repair: Add elements to cover missing sums?
        # Or just return a dense set like range(0, 129) which is definitely valid (peak will be high, but valid).
        # However, we want to minimize peak. 
        # If SA failed to find a valid one, it's unlikely. 
        # Let's try to force a valid solution from best_A if it's close.
        # Or just use a pre-computed valid set if we have one? 
        # We don't have a pre-computed one here.
        # Let's try a greedy repair on best_A.
        missing = [i for i in range(RANGE_SIZE) if final_reps[i] == 0]
        # For each missing sum, add a pair?
        # This might increase peak significantly.
        # Given the SA usually finds valid solutions, this is a safety net.
        # Let's just return the best valid one found during SA? 
        # We didn't track "best valid" separately from "best objective".
        # But objective < 1000 implies valid.
        # So if best_obj >= 1000, we didn't find a valid one.
        # In that case, we should probably return a valid set.
        # Let's generate a valid set quickly: A = [0, 128] + random middle?
        # Actually, range(64, 129) is valid?
        # Sums: 128 (64+64) to 256 (128+128).
        # Does it cover all?
        # 128=64+64. 129=64+65. ... 192=64+128.
        # 193=65+128. ... 256=128+128.
        # Yes, range(64, 129) covers [128, 256].
        # Peak for range(64, 129) (size 65):
        # Total pairs ~ 2145. Avg rep ~ 16.
        # It's valid.
        if best_obj >= 1000:
             return list(range(64, 129))
             
    return best_A
