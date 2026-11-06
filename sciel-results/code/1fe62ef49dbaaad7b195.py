import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for 60s limit
    
    # Pre-generate all inputs as bits
    # Input k has bits (k >> i) & 1 at position i
    # We use a 2D array: [NUM_INPUTS, N]
    # State[i, j] is the bit at wire j for input i
    # To save memory and time, we can use uint8
    
    # Initialize State
    # k in 0..2^N-1
    # We can generate this efficiently
    # State = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    # for j in range(N):
    #     State[:, j] = (np.arange(NUM_INPUTS) >> j) & 1
    
    # Faster generation
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    State = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        State[:, j] = (indices >> j) & 1
        
    # Identify sorted inputs
    # An input is sorted if State[i, 0] <= State[i, 1] <= ... <= State[i, N-1]
    # Since bits are 0/1, this means all 0s are before all 1s.
    # We can check this by looking at the first 1.
    # Or simply: np.all(State[:, :-1] <= State[:, 1:], axis=1)
    
    is_sorted = np.all(State[:, :-1] <= State[:, 1:], axis=1)
    unsolved_mask = ~is_sorted
    num_unsolved = np.sum(unsolved_mask)
    
    network = []
    
    # Candidate comparators
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while num_unsolved > 0:
        if time.time() > deadline:
            break
            
        best_comparator = None
        max_solved = 0
        best_new_unsolved_mask = unsolved_mask
        
        for i, j in candidates:
            # Check if this comparator is already in the network?
            # Greedy usually doesn't repeat, but sometimes it might be beneficial?
            # No, repeating a comparator is useless if the network is deterministic and monotonic?
            # Actually, compare-exchange is idempotent. So (i,j) then (i,j) is same as (i,j).
            # So we should not add duplicates.
            if (i, j) in network:
                continue
                
            # Apply comparator (i, j) to a copy of the state?
            # We don't want to modify the global State until we pick the best one.
            # But copying State (262k x 18) 153 times is expensive.
            
            # Optimization:
            # We only care about the inputs that are currently UNSOLVED.
            # Let S be the indices where unsolved_mask is True.
            # We can operate on State[S, :]
            
            # However, applying (i,j) changes the state of S.
            # We need to check if the *result* is sorted.
            
            # Let's extract the columns for i and j for the unsolved inputs
            # But we need the full state to check if it's sorted?
            # Yes, we need to check the whole row.
            
            # Alternative:
            # Just apply to the whole State, check, and revert if not chosen?
            # Reverting is also expensive.
            
            # Better:
            # Create a temporary copy of the columns i and j for the unsolved inputs?
            # No, the sorting condition depends on all columns.
            
            # Let's just copy the whole State for the candidate evaluation?
            # 153 * 262144 * 18 * 1 byte = ~700 MB of memory traffic per step?
            # That might be too slow.
            
            # Let's try a different approach:
            # We maintain the State.
            # For each candidate, we simulate the effect on the unsolved inputs only?
            # But the "sorted" check requires the whole row.
            
            # Let's assume the cost of copying State is acceptable for a few steps.
            # 262144 * 18 * 1 byte = 4.7 MB.
            # 153 copies = 700 MB.
            # Memory bandwidth is ~10 GB/s. 0.07s per step.
            # If we need 100 steps, that's 7s. Acceptable.
            
            State_temp = State.copy()
            
            # Apply comparator
            col_i = State_temp[:, i]
            col_j = State_temp[:, j]
            
            # min to i, max to j
            min_val = np.minimum(col_i, col_j)
            max_val = np.maximum(col_i, col_j)
            
            State_temp[:, i] = min_val
            State_temp[:, j] = max_val
            
            # Check which unsolved inputs are now sorted
            # We only need to check the ones that were unsolved
            # is_sorted_temp = np.all(State_temp[:, :-1] <= State_temp[:, 1:], axis=1)
            # This is expensive.
            
            # Optimization:
            # We know that only the rows where unsolved_mask is True can become sorted.
            # So we only check those rows.
            
            unsolved_indices = np.where(unsolved_mask)[0]
            if len(unsolved_indices) == 0:
                break
                
            # Check sortedness for these indices
            # We need to check if State_temp[unsolved_indices, :] is sorted
            # This is a sub-array of shape (num_unsolved, N)
            
            # To avoid creating a new array for the check, we can use the full check
            # and mask it.
            
            # Actually, np.all on the sub-array is fine.
            sub_state = State_temp[unsolved_indices, :]
            is_sorted_sub = np.all(sub_state[:, :-1] <= sub_state[:, 1:], axis=1)
            
            num_solved = np.sum(is_sorted_sub)
            
            if num_solved > max_solved:
                max_solved = num_solved
                best_comparator = (i, j)
                # We need to update unsolved_mask later
                # We can store the new unsolved mask
                # new_unsolved_mask = unsolved_mask.copy()
                # new_unsolved_mask[unsolved_indices[is_sorted_sub]] = False
                # But we need to apply the comparator to the global State first.
                
        if best_comparator is None or max_solved == 0:
            # No single comparator solves any new input?
            # This can happen if the greedy choice is bad.
            # In that case, we might need to pick a random one or break.
            # For now, break to avoid infinite loop.
            # Or pick the one that minimizes "disorder"?
            # Let's just break.
            break
            
        # Apply the best comparator to the global State
        i, j = best_comparator
        col_i = State[:, i]
        col_j = State[:, j]
        min_val = np.minimum(col_i, col_j)
        max_val = np.maximum(col_i, col_j)
        State[:, i] = min_val
        State[:, j] = max_val
        
        # Update unsolved mask
        # We need to re-calculate which inputs are sorted now
        # Only the ones that were unsolved can change status
        unsolved_indices = np.where(unsolved_mask)[0]
        sub_state = State[unsolved_indices, :]
        is_sorted_sub = np.all(sub_state[:, :-1] <= sub_state[:, 1:], axis=1)
        
        unsolved_mask[unsolved_indices[is_sorted_sub]] = False
        num_unsolved = np.sum(unsolved_mask)
        
        network.append(best_comparator)
        
    return network

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    import numpy as np
    import time

    N = 18
    NUM_INPUTS = 1 << N
    
    start_time = time.time()
    deadline = start_time + 50

    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    State = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        State[:, j] = (indices >> j) & 1
        
    is_sorted = np.all(State[:, :-1] <= State[:, 1:], axis=1)
    unsolved_mask = ~is_sorted
    num_unsolved = np.sum(unsolved_mask)
    
    network = []
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while num_unsolved > 0:
        if time.time() > deadline:
            break
            
        best_comparator = None
        max_solved = 0
        
        for i, j in candidates:
            if (i, j) in network:
                continue
                
            State_temp = State.copy()
            
            col_i = State_temp[:, i]
            col_j = State_temp[:, j]
            
            min_val = np.minimum(col_i, col_j)
            max_val = np.maximum(col_i, col_j)
            
            State_temp[:, i] = min_val
            State_temp[:, j] = max_val
            
            unsolved_indices = np.where(unsolved_mask)[0]
            if len(unsolved_indices) == 0:
                break
                
            sub_state = State_temp[unsolved_indices, :]
            is_sorted_sub = np.all(sub_state[:, :-1] <= sub_state[:, 1:], axis=1)
            
            num_solved = np.sum(is_sorted_sub)
            
            if num_solved > max_solved:
                max_solved = num_solved
                best_comparator = (i, j)
                
        if best_comparator is None or max_solved == 0:
            break
            
        i, j = best_comparator
        col_i = State[:, i]
        col_j = State[:, j]
        min_val = np.minimum(col_i, col_j)
        max_val = np.maximum(col_i, col_j)
        State[:, i] = min_val
        State[:, j] = max_val
        
        unsolved_indices = np.where(unsolved_mask)[0]
        sub_state = State[unsolved_indices, :]
        is_sorted_sub = np.all(sub_state[:, :-1] <= sub_state[:, 1:], axis=1)
        
        unsolved_mask[unsolved_indices[is_sorted_sub]] = False
        num_unsolved = np.sum(unsolved_mask)
        
        network.append(best_comparator)
        
    return network
