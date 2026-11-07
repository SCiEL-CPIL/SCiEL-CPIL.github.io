import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Precompute bits for quick access? Not strictly necessary, shifts are fast.
    
    def is_valid(mask):
        """
        Check if the set represented by mask is valid.
        Valid means: no distinct a, b, c in S such that a+b, a+c, b+c are in S.
        """
        # Iterate over all pairs a < b in S
        # If a+b is in S, check if there exists c > b in S such that a+c and b+c are in S.
        # To optimize, we can break early.
        
        for a in range(1, N + 1):
            if not (mask >> a) & 1:
                continue
            
            # b must be > a
            for b in range(a + 1, N + 1):
                if not (mask >> b) & 1:
                    continue
                
                s_ab = a + b
                if s_ab > N:
                    break
                if not (mask >> s_ab) & 1:
                    continue
                
                # If a+b is in S, we have a potential conflict.
                # We need to find c > b such that c in S, a+c in S, b+c in S.
                # Note: a+c and b+c must be <= N.
                # Max c is N - max(a, b) = N - b.
                max_c = N - b
                if max_c <= b:
                    continue
                
                for c in range(b + 1, max_c + 1):
                    if not (mask >> c) & 1:
                        continue
                    s_ac = a + c
                    s_bc = b + c
                    # s_ac and s_bc are guaranteed <= N because c <= N-b => b+c <= N.
                    # And a < b => a+c < b+c <= N.
                    if (mask >> s_ac) & 1 and (mask >> s_bc) & 1:
                        return False
        return True

    def get_elements(mask):
        """Return list of elements in the set."""
        return [i for i in range(1, N + 1) if (mask >> i) & 1]

    def set_to_mask(s_set):
        mask = 0
        for x in s_set:
            mask |= (1 << x)
        return mask

    # Initial solution: Upper half
    # {65, ..., 128} is valid. Size 64.
    best_mask = 0
    for i in range(N // 2 + 1, N + 1):
        best_mask |= (1 << i)
    best_size = N // 2
    
    # Verify best_mask is valid (it should be)
    if not is_valid(best_mask):
        # Fallback if upper half somehow invalid (shouldn't happen)
        best_mask = 0
        best_size = 0

    random.seed(42)

    # Local Search Function
    def local_search(start_mask):
        current_mask = start_mask
        current_size = bin(current_mask).count("1")
        
        # Try to improve by adding elements first
        while True:
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding any element not in current_mask
            # Randomize order to escape deterministic traps
            candidates = [i for i in range(1, N + 1) if not (current_mask >> i) & 1]
            random.shuffle(candidates)
            
            for elem in candidates:
                trial_mask = current_mask | (1 << elem)
                if is_valid(trial_mask):
                    current_mask = trial_mask
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no addition works, try swaps (remove 1, add 1)
            # This allows moving to a different local optimum of the same size
            # which might allow future additions.
            elements = get_elements(current_mask)
            if not elements:
                break
            
            to_remove = random.choice(elements)
            base_mask = current_mask & ~(1 << to_remove)
            
            # Try to add one element to base_mask
            # Candidates are those not in base_mask and != to_remove
            swap_candidates = [i for i in range(1, N + 1) if i != to_remove and not (base_mask >> i) & 1]
            random.shuffle(swap_candidates)
            
            swap_found = False
            for elem in swap_candidates:
                trial_mask = base_mask | (1 << elem)
                if is_valid(trial_mask):
                    # Accept swap to change the landscape
                    current_mask = trial_mask
                    swap_found = True
                    break
            
            if not swap_found:
                # Try 2-for-1 swap? Remove 1, add 2.
                # This increases size by 1 if successful.
                # This is a more powerful move.
                # Pick 2 candidates to add
                if len(swap_candidates) >= 2:
                    # Try adding 2 elements to base_mask
                    # We need to check validity incrementally or check the final set.
                    # Checking the final set is O(N^3).
                    # Let's try random pairs of additions
                    pair_found = False
                    # Limit attempts to avoid slowdown
                    for _ in range(10):
                        c1, c2 = random.sample(swap_candidates, 2)
                        trial_mask = base_mask | (1 << c1) | (1 << c2)
                        if is_valid(trial_mask):
                            current_mask = trial_mask
                            current_size += 1
                            pair_found = True
                            break
                    if pair_found:
                        continue
                else:
                    # No swaps or 2-for-1 found, stuck
                    break
        
        return current_mask, current_size

    # Phase 1: Greedy addition from upper half
    current_mask = best_mask
    current_size = best_size
    
    # Try to add small numbers greedily
    for x in range(1, N // 2 + 1):
        if time.time() > deadline - 20:
            break
        trial_mask = current_mask | (1 << x)
        if is_valid(trial_mask):
            current_mask = trial_mask
            current_size += 1
    
    if current_size > best_size:
        best_mask = current_mask
        best_size = current_size

    # Phase 2: Local Search with multiple restarts
    # We will run local search on the current best, and also on perturbed versions.
    
    # Run local search on current best
    improved_mask, improved_size = local_search(current_mask)
    if improved_size > best_size:
        best_mask = improved_mask
        best_size = improved_size

    # Restart loop
    restarts = 0
    while time.time() < deadline - 10 and restarts < 50:
        restarts += 1
        
        # Create a perturbed start
        # Remove 1-5 random elements from best_mask
        start_mask = best_mask
        elements = get_elements(start_mask)
        if elements:
            num_remove = random.randint(1, min(5, len(elements)))
            to_remove = random.sample(elements, num_remove)
            for x in to_remove:
                start_mask &= ~(1 << x)
        
        # Run local search
        improved_mask, improved_size = local_search(start_mask)
        
        if improved_size > best_size:
            best_mask = improved_mask
            best_size = improved_size
    
    # Final polish: try to add any remaining element to best_mask
    if time.time() < deadline - 2:
        remaining = [i for i in range(1, N + 1) if not (best_mask >> i) & 1]
        random.shuffle(remaining)
        for elem in remaining:
            trial_mask = best_mask | (1 << elem)
            if is_valid(trial_mask):
                best_mask = trial_mask
                best_size += 1
    
    # Convert best_mask to list of ints
    result = get_elements(best_mask)
    
    # Ensure types are Python int
    return [int(x) for x in result]

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Validate
    s_set = set(res)
    valid = True
    for a in s_set:
        for b in s_set:
            if a >= b: continue
            if a + b in s_set:
                for c in s_set:
                    if c <= b: continue
                    if a + c in s_set and b + c in s_set:
                        valid = False
                        break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
