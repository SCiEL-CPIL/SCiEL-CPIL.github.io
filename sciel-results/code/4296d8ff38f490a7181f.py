"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires using a Greedy Layered Construction
    approach inspired by SAT formulation. We build the network layer by layer,
    selecting comparators that maximize the reduction of 'disorder' (inversions)
    on the set of currently failing boolean inputs.
    """
    import random
    import time
    import numpy as np

    start_time = time.time()
    timeout = 55.0

    # Precompute all boolean inputs as a numpy array for vectorized operations
    # Shape: (2^N, N)
    # To save memory and time, we process in chunks or use a smart verification.
    # 2^18 = 262,144. N=18. 262144 * 18 * 1 byte = ~4.7 MB. This is fine.
    all_inputs = np.array([[(mask >> i) & 1 for i in range(N)] for mask in range(1 << N)], dtype=np.int8)
    
    def check_network(comparators):
        """
        Check if the network sorts all boolean inputs.
        Returns True if sorting, False otherwise.
        Also returns the number of failing inputs for heuristic guidance.
        """
        if not comparators:
            # Check if all inputs are already sorted (impossible for N>1)
            return False, (1 << N)
        
        # Copy inputs to simulate
        states = all_inputs.copy()
        
        for i, j in comparators:
            # Vectorized compare-exchange
            # Swap if states[:, i] > states[:, j]
            mask_swap = states[:, i] > states[:, j]
            if np.any(mask_swap):
                # Swap values where mask_swap is True
                temp = states[mask_swap, i].copy()
                states[mask_swap, i] = states[mask_swap, j]
                states[mask_swap, j] = temp
        
        # Check if sorted
        # A state is sorted if states[:, k] <= states[:, k+1] for all k
        # We can check if any state has an inversion
        # Inversion exists if states[:, k] > states[:, k+1] for any k
        # Let's check the first inversion for each state
        sorted_mask = np.ones(len(states), dtype=bool)
        for k in range(N - 1):
            inverted = states[:, k] > states[:, k+1]
            sorted_mask &= ~inverted
            if not np.any(sorted_mask):
                break
        
        num_failing = np.sum(~sorted_mask)
        return num_failing == 0, num_failing

    def greedy_layered_construct():
        """
        Build the network layer by layer.
        """
        random.seed(42)
        current_comps = []
        # We track the set of inputs that are NOT yet sorted by the current prefix.
        # Initially, all inputs are unsorted (except the identity permutation if we consider 0-1, 
        # but actually for 0-1, an input is sorted if it is of the form 00...011...1).
        
        # Precompute which inputs are already sorted (0s then 1s)
        sorted_inputs_mask = np.zeros(1 << N, dtype=bool)
        for mask in range(1 << N):
            # Check if mask is sorted: 000...0111...1
            # This happens if the sequence of bits is non-decreasing
            is_sorted = True
            prev = 0
            for i in range(N):
                bit = (mask >> i) & 1
                if bit < prev:
                    is_sorted = False
                    break
                prev = bit
            sorted_inputs_mask[mask] = is_sorted
        
        # We only care about inputs that are NOT sorted initially
        # Actually, we need to process ALL inputs to see which ones get fixed.
        # But we can optimize by only tracking the "failing" set.
        
        # State of all inputs
        states = all_inputs.copy()
        
        max_layers = 50
        for layer in range(max_layers):
            if time.time() - start_time > timeout:
                break
            
            # Identify which inputs are still failing
            failing_mask = np.ones(len(states), dtype=bool)
            for k in range(N - 1):
                inverted = states[:, k] > states[:, k+1]
                failing_mask &= ~inverted
                if not np.any(failing_mask):
                    break
            
            if not np.any(failing_mask):
                break # All inputs sorted
            
            # Get indices of failing inputs
            failing_indices = np.where(failing_mask)[0]
            
            if len(failing_indices) == 0:
                break
            
            # For each potential comparator (i, j), calculate how many failing inputs it would fix
            # "Fixing" means that after applying (i,j) to the failing inputs, they become sorted?
            # No, that's too strong. It should reduce the total number of inversions.
            # Better heuristic: Choose (i,j) that maximizes the number of failing inputs 
            # for which the comparator actually swaps (i.e., state[i] > state[j]).
            # This is a proxy for progress.
            
            best_comp = None
            best_score = -1
            
            # To save time, we might not evaluate all C(18,2)=153 comparators every layer
            # if the number of failing inputs is small, we can be more selective.
            # But 153 is small. Let's evaluate all.
            
            # We need to apply the comparator to the *current* states of the failing inputs only
            # to determine the score.
            failing_states = states[failing_indices]
            
            for i in range(N):
                for j in range(i + 1, N):
                    # Count how many failing inputs have state[i] > state[j]
                    # These are the ones that would be swapped.
                    swaps = np.sum(failing_states[:, i] > failing_states[:, j])
                    
                    # Additional heuristic: prefer comparators that are "local" or part of a pattern?
                    # Let's stick to swap count for now.
                    
                    if swaps > best_score:
                        best_score = swaps
                        best_comp = (i, j)
            
            if best_comp is None or best_score == 0:
                # No comparator can fix any failing input directly?
                # This shouldn't happen if the network is not sorted, unless we are stuck in a local minimum
                # where no single comparator reduces the inversion count for *all* failing inputs?
                # Actually, if best_score is 0, it means for all failing inputs, state[i] <= state[j] for all i<j?
                # That would imply they are sorted. Contradiction.
                # So best_score must be > 0 if failing_mask is not empty.
                break
            
            # Add the best comparator
            current_comps.append(best_comp)
            
            # Apply the comparator to the global states
            i, j = best_comp
            mask_swap = states[:, i] > states[:, j]
            if np.any(mask_swap):
                temp = states[mask_swap, i].copy()
                states[mask_swap, i] = states[mask_swap, j]
                states[mask_swap, j] = temp
        
        return current_comps

    # Phase 1: Greedy Construction
    best_comps = greedy_layered_construct()
    
    # Phase 2: Local Search Optimization (Removal and Swap)
    # Try to remove comparators
    if time.time() - start_time < timeout - 5:
        valid, _ = check_network(best_comps)
        if not valid:
            # If greedy failed to sort, we might need to add more or use a different strategy.
            # But let's assume greedy gets close. If not valid, we fall back to a known good network?
            # No, we must return a valid network.
            # Let's try to complete the greedy process or use a dense network as fallback.
            # For now, if greedy doesn't sort, we try to add random comparators until it does.
            while not valid and time.time() - start_time < timeout - 5:
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = random.randint(0, len(best_comps))
                new_comps = best_comps[:pos] + [(i, j)] + best_comps[pos:]
                valid, _ = check_network(new_comps)
                if valid:
                    best_comps = new_comps
                    break
            if not valid:
                # Emergency fallback: Odd-Even Transposition
                best_comps = []
                for phase in range(N):
                    for i in range(phase % 2, N - 1, 2):
                        best_comps.append((i, i + 1))
                valid, _ = check_network(best_comps)
                # This should be valid

        # Now try to remove
        random.seed(42)
        while time.time() - start_time < timeout - 2:
            removed = False
            indices = list(range(len(best_comps)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > timeout - 2:
                    break
                new_comps = best_comps[:idx] + best_comps[idx+1:]
                valid, _ = check_network(new_comps)
                if valid:
                    best_comps = new_comps
                    removed = True
            
            if not removed:
                # Try swapping comparators? Or just break.
                # Swapping is complex. Let's try adding a random one and then removing more.
                if len(best_comps) < 100 and random.random() < 0.2:
                    i, j = random.sample(range(N), 2)
                    if i > j: i, j = j, i
                    pos = random.randint(0, len(best_comps))
                    new_comps = best_comps[:pos] + [(i, j)] + best_comps[pos:]
                    valid, _ = check_network(new_comps)
                    if valid:
                        best_comps = new_comps
                else:
                    break
    
    return best_comps
# EVOLVE-BLOCK-END
