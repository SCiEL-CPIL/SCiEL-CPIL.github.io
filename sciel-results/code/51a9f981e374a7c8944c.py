# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # 1. Deterministic Greedy Baseline (ensures score >= 1.0)
    def _greedy_numpy(order, counts_arr):
        chosen = []
        for x in order:
            if counts_arr[2 * x] >= 2:
                continue
            valid = True
            for y in chosen:
                s = x + y
                if counts_arr[s] >= 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts_arr[2 * x] += 1
                for y in chosen[:-1]:
                    s = x + y
                    counts_arr[s] += 1
        return chosen

    order1 = list(range(1, 1001))
    counts1 = np.zeros(2001, dtype=np.int32)
    best_set = _greedy_numpy(order1, counts1)
    best_size = len(best_set)

    # 2. Simulated Annealing Strategy
    # State: binary mask of length 1000 (index i represents integer i+1)
    # Objective: Maximize size - Penalty(violations)
    
    def _get_violations(mask):
        """
        Calculate the number of sum representations that exceed the limit (2).
        Returns a penalty value. 0 means valid B2[2] set.
        """
        # Sum of mask is the size of the set
        # We need counts of sums a+b for a<=b in the set.
        # Efficient way: 
        # 1. Compute all pair sums (including a+a)
        # 2. Count frequencies
        # 3. Penalty = sum(max(0, count[s] - 2))
        
        # To keep it fast, we only update locally or use vectorized ops if possible.
        # For SA, we need fast evaluation of *changes*.
        # However, a full recalculation is O(N^2). With N=1000, N^2=1e6.
        # 1e6 operations in numpy is fast (~1ms).
        # We can afford ~1000 steps/sec if we are efficient.
        
        # Let's use a more optimized local update strategy in the loop,
        # but for the initial state and occasional full checks, we use this.
        
        # Actually, for SA, we need to evaluate delta_penalty.
        # Let's maintain a counts array.
        
        pass # Placeholder, logic implemented in main loop

    def _init_counts_from_mask(mask):
        """Compute sum counts from a binary mask."""
        # Get indices of 1s
        idxs = np.nonzero(mask)[0] + 1 # 1-based indices
        if len(idxs) == 0:
            return np.zeros(2001, dtype=np.int32)
        
        # Compute all pair sums
        # This is O(k^2). If k is large (e.g. 500), 250k sums.
        # We can use np.add.outer and then bincount?
        # np.add.outer creates a k x k matrix. 
        # For k=500, 500x500 = 250,000 elements.
        # Then we take upper triangle (a<=b).
        # Then bincount.
        
        # But we need to be careful about memory and speed.
        # Alternative: Loop over elements and update counts? 
        # Adding one element x: 
        #   count[2x] += 1
        #   count[x+y] += 1 for all y in set
        # This is O(k).
        
        # For SA, we will maintain the counts array incrementally.
        
        counts = np.zeros(2001, dtype=np.int32)
        # Use the greedy-like addition logic to build counts from scratch
        # This is O(k^2) total.
        for i, x in enumerate(idxs):
            counts[2*x] += 1
            for y in idxs[i+1:]:
                counts[x+y] += 1
        return counts

    def _calc_penalty(counts):
        """Calculate penalty: sum of (count[s] - 2) for count[s] > 2."""
        # Only sums up to 2000 are relevant
        over = counts[2:2001] > 2
        if np.any(over):
            return np.sum(counts[over] - 2)
        return 0

    # SA Parameters
    T_start = 100.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # Initial state: Random mask or Greedy?
    # Start with a dense random mask to explore, or Greedy to refine?
    # Let's start with a random mask with ~50% density, or maybe Greedy + noise.
    # Greedy gives a valid set. SA can try to grow it by allowing temporary violations.
    
    # Let's run multiple SA restarts.
    
    rng = np.random.default_rng(42)
    
    for restart in range(20): # Max 20 restarts
        if time.time() > deadline - 10:
            break
            
        # Initialize state
        # Option A: Start from best known valid set, randomize some bits?
        # Option B: Start from random mask.
        # Let's start from a random mask with density ~0.5, or slightly higher.
        # Actually, starting from the greedy solution and trying to ADD more via SA
        # is a good "repair and grow" strategy.
        
        # Let's try: Start with greedy solution, then SA tries to flip 0->1 (add) 
        # and 1->0 (remove) to maximize size - penalty.
        
        mask = np.zeros(1000, dtype=np.int8)
        for x in best_set:
            mask[x-1] = 1
            
        counts = _init_counts_from_mask(mask)
        current_penalty = _calc_penalty(counts)
        current_size = np.sum(mask)
        current_score = current_size - current_penalty * 10 # Weight penalty heavily
        
        T = T_start
        steps = 0
        max_steps = 5000 # Adjust based on time
        
        while time.time() < deadline - 5 and steps < max_steps:
            steps += 1
            
            # Choose a random index to flip
            idx = rng.integers(0, 1000)
            x = idx + 1
            
            if mask[idx] == 1:
                # Try to remove x
                # Update counts: decrement contributions of x
                old_penalty = current_penalty
                old_size = current_size
                
                # Decrement counts
                counts[2*x] -= 1
                # Need to know other elements to decrement x+y
                # This is the bottleneck: we need the list of other elements.
                # Maintaining a list of active indices is O(1) update but O(N) search if not careful.
                # Let's maintain a list `active` of indices in mask.
                
                # For simplicity in this prototype, let's just re-calculate penalty 
                # using the counts array after update, but we need to update counts correctly.
                # Updating counts for removal of x:
                # counts[2x] -= 1
                # for y in active (y != x): counts[x+y] -= 1
                
                # We need `active` list.
                # Let's refactor to maintain `active` list.
                
                # Since I'm writing this from scratch, I'll implement the loop with `active` list.
                # But I already started the code above. I will rewrite the SA loop properly.
                
                break # Break out of this dummy loop to rewrite properly below

    # --- Proper SA Implementation ---
    
    def _run_sa(initial_mask, time_limit):
        mask = initial_mask.copy()
        active = np.nonzero(mask)[0] + 1 # 1-based
        counts = _init_counts_from_mask(mask)
        current_penalty = _calc_penalty(counts)
        current_size = len(active)
        
        # Score = Size - Penalty * Lambda
        # We want to maximize this.
        # Let's use a dynamic lambda or just track penalty separately.
        # If penalty == 0, we want max size.
        # If penalty > 0, we might accept to escape local minima.
        
        T = 50.0
        cooling = 0.9995
        
        best_mask = mask.copy()
        best_size = current_size
        best_penalty = current_penalty
        
        while time.time() < time_limit:
            # Flip a random bit
            idx = np.random.randint(0, 1000)
            x = idx + 1
            
            if mask[idx] == 1:
                # Remove x
                # 1. Update counts
                counts[2*x] -= 1
                # Find other active elements
                # This is O(N) to iterate active. 
                # To make it faster, we can just iterate over all 1..1000 and check mask? 
                # No, iterating active is better if sparse, but active is ~500.
                # Let's iterate over active list.
                
                # Remove x from active list
                # active is a numpy array. Removing an element is slow.
                # Let's use a Python list for active for easier removal, or just recompute?
                # Recomputing counts from scratch is O(N^2) ~ 1ms.
                # If we do 1000 steps/sec, that's 1s per 1000 steps. 
                # 120s -> 120,000 steps.
                # 120,000 * 1ms = 120s. Just barely.
                # Incremental is better.
                
                # Let's maintain `active` as a Python list for O(1) removal (if we have index) 
                # or O(N) removal.
                # Actually, we can just iterate over the `active` list to update counts.
                
                # Remove x from active
                try:
                    active.remove(x)
                except ValueError:
                    pass # Should not happen
                
                # Update counts for pairs with x
                for y in active:
                    counts[x + y] -= 1
                
                new_penalty = _calc_penalty(counts)
                new_size = len(active)
                
                # Delta in "Energy" (we minimize Penalty - Size? Or maximize Size - Penalty?)
                # Let's define Energy = Penalty * 100 - Size. We want to minimize Energy.
                # Or Score = Size - Penalty * 100. We want to maximize Score.
                # Let's use Score.
                
                old_score = current_size - current_penalty * 100
                new_score = new_size - new_penalty * 100
                delta_score = new_score - old_score
                
                if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
                    # Accept
                    mask[idx] = 0
                    current_penalty = new_penalty
                    current_size = new_size
                    
                    if current_penalty == 0 and current_size > best_size:
                        best_size = current_size
                        best_mask = mask.copy()
                else:
                    # Reject: Restore state
                    # We already modified counts and active. Need to restore.
                    # This is inefficient. 
                    # Better: Calculate delta, decide, then update state only if accepted.
                    # To do that, we need to compute new_penalty without modifying global state.
                    # That requires copying counts (O(N)) or recalculating penalty locally.
                    
                    # Let's just restore.
                    mask[idx] = 1
                    active.append(x)
                    # Restore counts
                    counts[2*x] += 1
                    for y in active:
                        if y != x:
                            counts[x + y] += 1
                    # Note: active now contains x again.
                    # The loop `for y in active` in restoration includes x if not careful.
                    # In removal: we removed x, then iterated remaining.
                    # In restoration: we add x back. We need to add back pairs (x,y) for y in active (which now includes x? No, we append after).
                    # Let's be precise.
                    
                    # Actually, the "restore" logic is error-prone and slow.
                    # Let's use the "copy counts" approach for evaluation?
                    # Copying 2001 int32s is very fast (~10us).
                    # So:
                    # 1. Copy counts to temp_counts
                    # 2. Apply change to temp_counts
                    # 3. Calc penalty on temp_counts
                    # 4. Decide
                    # 5. If accept, assign temp_counts to counts, update mask/active.
                    
                    # This is much cleaner.
                    pass # This logic will be implemented in the final loop below.
            
            else:
                # Add x
                pass # Same logic
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_mask, best_size, best_penalty

    # Let's implement the cleaner SA loop
    
    def _sa_optimize(initial_mask, time_limit):
        mask = initial_mask.copy()
        active = list(np.nonzero(mask)[0] + 1)
        counts = _init_counts_from_mask(mask)
        current_penalty = _calc_penalty(counts)
        current_size = len(active)
        
        best_mask = mask.copy()
        best_size = current_size
        
        T = 50.0
        cooling = 0.9999
        
        while time.time() < time_limit:
            idx = np.random.randint(0, 1000)
            x = idx + 1
            
            temp_counts = counts.copy()
            
            if mask[idx] == 1:
                # Remove x
                temp_counts[2*x] -= 1
                for y in active:
                    if y != x:
                        temp_counts[x + y] -= 1
                new_size = current_size - 1
            else:
                # Add x
                temp_counts[2*x] += 1
                for y in active:
                    temp_counts[x + y] += 1
                new_size = current_size + 1
            
            new_penalty = _calc_penalty(temp_counts)
            
            # Score: Maximize Size - Penalty * Weight
            # Weight should be high enough to prefer valid sets, but low enough to allow exploration.
            # If Penalty > 0, we are in an invalid state.
            # Let's use Weight = 10.
            old_score = current_size - current_penalty * 10
            new_score = new_size - new_penalty * 10
            delta = new_score - old_score
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                # Accept
                mask[idx] = 1 - mask[idx]
                if mask[idx] == 1:
                    active.append(x)
                else:
                    active.remove(x)
                
                counts = temp_counts
                current_penalty = new_penalty
                current_size = new_size
                
                if current_penalty == 0 and current_size > best_size:
                    best_size = current_size
                    best_mask = mask.copy()
            
            T *= cooling
            if T < 0.001:
                break
        
        return best_mask, best_size

    # Run SA
    # Start from Greedy solution
    base_mask = np.zeros(1000, dtype=np.int8)
    for x in best_set:
        base_mask[x-1] = 1
        
    # Try a few SA runs with different initial temperatures or random perturbations
    for i in range(5):
        if time.time() > deadline - 5:
            break
            
        # Perturb initial mask slightly to escape local minima of the greedy solution
        perturbed_mask = base_mask.copy()
        # Flip 10 random bits
        for _ in range(10):
            idx = np.random.randint(0, 1000)
            perturbed_mask[idx] = 1 - perturbed_mask[idx]
            
        sa_time_limit = min(time.time() + 10, deadline - 5)
        sa_mask, sa_size = _sa_optimize(perturbed_mask, sa_time_limit)
        
        if sa_size > best_size:
            best_size = sa_size
            # Reconstruct best_set from mask
            best_set = list(np.nonzero(sa_mask)[0] + 1)

    # Final check: if SA found a better valid set, use it.
    # If not, return the original greedy set (which is valid).
    
    return best_set
# EVOLVE-BLOCK-END
