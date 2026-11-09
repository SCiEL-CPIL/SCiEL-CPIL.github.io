import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safe margin
    
    # Precompute conflicts using NumPy
    # For each element x, store the indices of the other 3 elements in each 4-AP containing x
    # others[x] will be a numpy array of shape (num_aps, 3)
    others = [None] * (N + 1)
    
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            # For each position in the AP, the other 3 are the conflict
            # Position 0: a, others are b, c, d
            # Position 1: b, others are a, c, d
            # Position 2: c, others are a, b, d
            # Position 3: d, others are a, b, c
            
            # We will append to lists first for efficiency, then convert to numpy
            if others[ap[0]] is None: others[ap[0]] = []
            if others[ap[1]] is None: others[ap[1]] = []
            if others[ap[2]] is None: others[ap[2]] = []
            if others[ap[3]] is None: others[ap[3]] = []
            
            others[ap[0]].append([ap[1]-1, ap[2]-1, ap[3]-1]) # 0-indexed for numpy
            others[ap[1]].append([ap[0]-1, ap[2]-1, ap[3]-1])
            others[ap[2]].append([ap[0]-1, ap[1]-1, ap[3]-1])
            others[ap[3]].append([ap[0]-1, ap[1]-1, ap[2]-1])
            
    # Convert to numpy arrays
    others_np = [None] * (N + 1)
    for x in range(1, N + 1):
        if others[x]:
            others_np[x] = np.array(others[x], dtype=np.int32)
        else:
            others_np[x] = np.empty((0, 3), dtype=np.int32)
            
    def can_add(x, S):
        """Check if x (1-indexed) can be added to set S (numpy bool array, 0-indexed)"""
        idx = x - 1
        if S[idx]:
            return False
        confs = others_np[x]
        if len(confs) == 0:
            return True
        # Check if any of the conflict triples are fully present in S
        # S[confs] has shape (num_aps, 3)
        # We want to know if any row is all True
        return not np.any(np.all(S[confs], axis=1))
    
    def get_size(S):
        return np.sum(S)
        
    def greedy_construct(seed):
        rng = random.Random(seed)
        S = np.zeros(N, dtype=bool)
        elements = list(range(1, N + 1))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S):
                S[x-1] = True
                
        return S
    
    def local_search(S, rng, time_budget):
        S = S.copy()
        current_size = get_size(S)
        best_size = current_size
        best_S = S.copy()
        
        # Precompute list of all elements for random selection
        all_elements = list(range(1, N + 1))
        
        while time.time() - start_time < time_budget:
            # Randomly pick an element to remove
            if current_size == 0:
                break
            x_remove = rng.choice(all_elements)
            if not S[x_remove-1]:
                continue
                
            S[x_remove-1] = False
            current_size -= 1
            
            # Try to add any element
            # To speed up, we can just iterate through a shuffled list of all elements
            # Or just try the removed one and a few random ones?
            # Better: Try all elements that were not in S, plus the removed one.
            # Since N=512, iterating all 512 is cheap if vectorized or fast.
            
            # Let's just iterate through all elements in random order
            candidates = all_elements[:]
            rng.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                if can_add(x, S):
                    S[x-1] = True
                    current_size += 1
                    added_any = True
                    
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
                
            # If we didn't add anything, or size didn't improve, we might want to revert?
            # Standard Simulated Annealing or Hill Climbing:
            # If current_size >= best_size, keep. Else revert?
            # Actually, if we removed x and added nothing, size dropped by 1.
            # If we removed x and added y, size same.
            # If we removed x and added y, z, size increased.
            
            # If current_size < best_size, we should probably revert to best_S?
            # Or just keep the current state if it's a local optimum?
            # Let's use a simple hill climbing: keep if size >= previous size.
            # But we already tracked best_S.
            
            # To avoid getting stuck, if no improvement in size, we might revert the removal
            # if it didn't lead to a better state.
            if current_size < best_size:
                # Revert to the state before this move?
                # We don't have a stack. Let's just reset to best_S if we are doing well?
                # No, local search should explore.
                # Let's just keep S as is. It's a random walk.
                pass
                
        return best_S

    # Main
    best_S = np.zeros(N, dtype=bool)
    best_size = 0
    rng = random.Random(42)
    
    # Initial greedy
    S = greedy_construct(0)
    size = get_size(S)
    if size > best_size:
        best_size = size
        best_S = S.copy()
        
    # Local search
    if time.time() - start_time < time_limit - 5:
        S = local_search(S, rng, time_limit - 5)
        size = get_size(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    # More restarts
    seed = 1
    while time.time() - start_time < time_limit - 2:
        S = greedy_construct(seed)
        size = get_size(S)
        
        # Only do local search if promising or if we have time
        if size >= best_size - 10:
            S = local_search(S, rng, time_limit - 2)
            size = get_size(S)
            
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
        seed += 1
        
    # Convert numpy bool array to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_S[i]]
    return result

# EVOLVE-BLOCK-END
