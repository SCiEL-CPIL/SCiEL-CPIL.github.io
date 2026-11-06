import time
import numpy as np

N = 16
MAX_TIME = 58.0

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # 1. Pre-compute all 65536 Boolean inputs
    # bits[i, j] is the j-th bit of i. Wire j corresponds to bit j.
    # Input i is sorted if bits[i] is non-decreasing (which for binary means 0...01...1).
    # Actually, for binary inputs, "sorted" means all 0s come before all 1s.
    # A violation exists if there is a 1 before a 0.
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)
    bits = (indices[:, None] >> np.arange(N, dtype=np.uint32)) & 1 # Shape: (65536, 16)
    
    # Pre-compute the "target" sorted state for each input?
    # No, we just need to know if the current state is sorted.
    # For binary, a vector v is sorted iff v is of the form 0...01...1.
    # This is equivalent to: no index k exists such that v[k]=1 and v[k+1]=0.
    # Or simply: np.diff(v) >= 0 for all k.
    
    # To speed up "is this input sorted?", we can precompute a "sortedness mask"
    # but since the state changes, we must re-evaluate.
    # However, we can represent the set of "currently unsorted inputs" as a boolean mask.
    
    # Initial state: all inputs are the raw bits.
    # Which inputs are initially sorted?
    # An input is initially sorted if it is of the form 0...01...1.
    # There are N+1 = 17 such inputs (all 0s, 1 zero 15 ones, ..., 15 zeros 1 one, all 1s).
    # But wait, the input is a specific permutation of bits.
    # The "unsorted" set is the set of inputs that are NOT of the form 0...01...1.
    
    # Let's identify which inputs are initially unsorted.
    # An input vector v is sorted if v = [0]*k + [1]*(N-k) for some k.
    # These correspond to specific integer indices.
    # Index for k zeros: sum(2^j for j in range(k, N)) = 2^N - 2^k.
    # So the sorted indices are {0, 2^15, 2^15+2^14, ..., 2^15+...+2^0}.
    # Actually, let's just compute the initial unsorted mask.
    
    def is_sorted_mask(vec):
        # vec is (N_inputs, N)
        # Check if any 1 is followed by 0
        # diff < 0 means 1 -> 0
        diffs = np.diff(vec, axis=1)
        return np.all(diffs >= 0, axis=1)
    
    current_state = bits.copy()
    sorted_mask = is_sorted_mask(current_state)
    unsorted_mask = ~sorted_mask
    
    # If all are sorted, return empty (unlikely for N=16)
    if np.all(sorted_mask):
        return []
        
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # We want to select a sequence of comparators.
    # Greedy Set Cover approach:
    # At each step, find the comparator (i, j) that "fixes" the most currently unsorted inputs.
    # "Fixing" an input means that after applying the comparator, that specific input becomes sorted.
    # Note: An input might become unsorted later? No, sorting networks are monotonic?
    # Actually, applying a comparator to a sorted vector keeps it sorted.
    # Proof: If v is sorted (0...01...1), and we swap (i, j) with i<j:
    #   Case 1: v[i]=0, v[j]=0 -> no change.
    #   Case 2: v[i]=1, v[j]=1 -> no change.
    #   Case 3: v[i]=0, v[j]=1 -> no change (already min/max).
    #   Case 4: v[i]=1, v[j]=0 -> This is a violation in a "sorted" vector?
    #           If v is sorted, there is no 1 before 0. So v[i]=1 implies v[j]=1 for j>i.
    #           So Case 4 cannot happen for a sorted vector.
    # Therefore, once an input is sorted, it remains sorted under any subsequent comparators.
    # This is a crucial property! It means the set of sorted inputs only grows.
    # We can greedily pick the comparator that adds the most newly sorted inputs.
    
    current_comps = []
    
    while not np.all(sorted_mask) and time.time() < deadline - 2.0:
        best_comp = None
        best_newly_sorted_count = 0
        best_new_state = None
        best_new_sorted_mask = None
        
        # For each candidate comparator, apply it to the current state
        # and count how many previously unsorted inputs become sorted.
        
        # Optimization: We can vectorize this over candidates?
        # It's hard to vectorize over the 120 candidates easily because the state update
        # depends on the specific (i, j).
        # But 120 iterations * (65536 * 16 ops) is ~ 120 * 1M = 120M ops.
        # In Numpy, this is very fast (< 0.1s).
        
        for comp in all_comps:
            i, j = comp
            # Apply comp to current_state
            # We only care about the inputs that are currently unsorted.
            # Let's create a temporary state for the unsorted inputs only?
            # No, just apply to whole state, it's fast enough.
            
            temp_state = current_state.copy()
            # min to i, max to j
            min_val = np.minimum(temp_state[:, i], temp_state[:, j])
            max_val = np.maximum(temp_state[:, i], temp_state[:, j])
            temp_state[:, i] = min_val
            temp_state[:, j] = max_val
            
            new_sorted_mask = is_sorted_mask(temp_state)
            
            # Count newly sorted: was unsorted, now sorted
            newly_sorted = np.sum((~sorted_mask) & new_sorted_mask)
            
            if newly_sorted > best_newly_sorted_count:
                best_newly_sorted_count = newly_sorted
                best_comp = comp
                best_new_state = temp_state
                best_new_sorted_mask = new_sorted_mask
        
        if best_comp is None or best_newly_sorted_count == 0:
            # No single comparator can fix any remaining unsorted inputs?
            # This shouldn't happen if the network is solvable, but if we get stuck,
            # we might need a different strategy or accept invalidity.
            # For safety, if we can't improve, break.
            break
        
        # Commit the move
        current_comps.append(best_comp)
        current_state = best_new_state
        sorted_mask = best_new_sorted_mask
        
        # Early exit if all sorted
        if np.all(sorted_mask):
            break

    # Verification: Ensure the final network is valid on ALL inputs
    # The greedy process guarantees that each step only adds sorted inputs,
    # and once sorted, they stay sorted.
    # So if we exited because all are sorted, it is valid.
    # If we exited due to time or stagnation, we must check validity.
    # If invalid, we need a fallback.
    
    # Check validity
    if not np.all(sorted_mask):
        # Fallback: Use a known valid network (e.g., Batcher or OET)
        # We'll generate a simple OET as a safe fallback.
        fallback_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    fallback_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    fallback_comps.append((j, j + 1))
        
        # Verify fallback
        vec = bits.copy()
        for i, j in fallback_comps:
            min_val = np.minimum(vec[:, i], vec[:, j])
            max_val = np.maximum(vec[:, i], vec[:, j])
            vec[:, i] = min_val
            vec[:, j] = max_val
        if np.all(np.diff(vec, axis=1) >= 0):
            return fallback_comps
        else:
            # This should never happen for OET
            return []

    return current_comps

# EVOLVE-BLOCK-END
