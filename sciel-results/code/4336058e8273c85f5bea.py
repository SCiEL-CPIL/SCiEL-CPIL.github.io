"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_N = 4096
    MAX_SUM = 3 * MAX_N
    TARGET_SIZE = 80
    
    best_set = []
    best_size = 0

    # Vectorized check function
    def can_add_correct(S_np, x, sum_flags):
        k = len(S_np)
        
        # 1. 3x
        if sum_flags[3*x]:
            return False
        # We will build the full array of new sums to check both against existing and internal collisions.
        # To save memory/time in the check, we can do it in parts if needed, but for k<=80 it's fine.
        
        # 2. 2x + S
        sums_2x_S = 2*x + S_np
        if np.any(sum_flags[sums_2x_S]):
            return False
            
        # 3. x + S_i + S_j
        # We generate the k x k matrix of sums
        sums_x_S_S = x + S_np[:, None] + S_np[None, :]
        
        # Check against existing sums
        if np.any(sum_flags[sums_x_S_S]):
            return False
        
        # Check internal collisions among new sums
        # The set of new sums is:
        # {3x} U {2x+s | s in S} U {x+s1+s2 | s1,s2 in S}
        # Note: {x+s1+s2} includes s1=s2, which is x+2s. This is distinct from 2x+s unless x=s, which is not the case as x is new.
        # Also, 3x is distinct from others if S is valid and x is new? 
        # We must ensure that no two *distinct* multisets produce the same sum.
        # The multiset of new sums has size 1 + k + k*(k+1)/2.
        # If we flatten the k x k matrix, we have k^2 elements.
        # The number of unique sums in the k x k matrix should be k*(k+1)/2.
        # The total number of unique sums in the combined set should be 1 + k + k*(k+1)/2.
        
        # Instead of checking unique counts which requires np.unique (slower), 
        # we can check if any sum appears more than expected?
        # Or simpler: just use np.unique on the concatenated array.
        # Size is ~ 1 + 80 + 6400 = 6481.
        # np.unique on 6481 ints is very fast.
        
        all_new_sums = np.concatenate([np.array([3*x]), sums_2x_S, sums_x_S_S.flatten()])
        unique_new_sums = np.unique(all_new_sums)
        
        expected_count = 1 + k + k*(k+1)//2
        
        if len(unique_new_sums) != expected_count:
            return False
            
        return True

    def add_correct(S_np, x, sum_flags):
        sum_flags[3*x] = True
        sum_flags[2*x + S_np] = True
        sum_flags[x + S_np[:, None] + S_np[None, :].flatten()] = True

    # Main Loop
    # Strategy: Random restarts with greedy construction.
    # To improve on the current 1.778 (which likely corresponds to size 16 if seed is 9, or size 16 if seed is 1? 
    # Actually 1.778 = 16/9. So seed likely finds 9. Current finds 16.
    # Wait, if seed is 1, score 1.778 means size 1.778? No, size is integer.
    # If seed is 9, size 16 gives 1.778.
    # If seed is 1, size 1 gives 1.0.
    # Let's assume the seed finds a small set, and we are doing better.
    # We want to maximize size.
    
    # We can also try a "local search" or "repair" phase.
    # But first, let's optimize the restart loop.
    
    # Optimization: 
    # Instead of shuffling the whole list every time, we can use different random seeds or different orderings.
    # Also, we can try to bias the selection.
    # For B_h sets, spreading out the elements is often good.
    # Random permutation is a good baseline.
    
    # Let's run as many restarts as possible.
    # Each restart takes some time. 
    # With vectorized checks, each step is fast.
    # Let's estimate time per restart.
    # If we find size 16 quickly, the loop breaks early? No, it continues to see if it can go higher.
    # But if we hit a dead end, it scans the rest of the candidates (4096 - 16 = 4080).
    # Each candidate check is O(k^2) vectorized.
    # For k=16, k^2=256. 4080 * 256 ~ 1M ops. Very fast.
    # So one restart is very fast.
    # We can do thousands of restarts.
    
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Use different random seeds for diversity
        random.seed(restart)
        candidates = list(range(1, MAX_N + 1))
        random.shuffle(candidates)
        
        S = []
        S_np = np.array([], dtype=int)
        sum_flags = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            if can_add_correct(S_np, x, sum_flags):
                S.append(x)
                S_np = np.append(S_np, x)
                S_np.sort()
                add_correct(S_np, x, sum_flags)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    if best_size == TARGET_SIZE:
                        return best_set
                        
        # Optional: Local search on the best set found so far?
        # If we have time, we can try to swap elements in best_set to allow adding more.
        # This is complex. Let's stick to restarts first.
        
    return best_set
# EVOLVE-BLOCK-END
