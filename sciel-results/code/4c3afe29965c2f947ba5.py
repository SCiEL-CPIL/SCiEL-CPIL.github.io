import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safe margin (90s limit, 32s buffer)
    
    # Precompute conflicts using NumPy
    # For each element x, store the indices of the other 3 elements in each 4-AP containing x
    others = [None] * (N + 1)
    
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            
            if others[ap[0]] is None: others[ap[0]] = []
            if others[ap[1]] is None: others[ap[1]] = []
            if others[ap[2]] is None: others[ap[2]] = []
            if others[ap[3]] is None: others[ap[3]] = []
            
            others[ap[0]].append([ap[1]-1, ap[2]-1, ap[3]-1])
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
        
        all_elements = list(range(1, N + 1))
        
        while time.time() - start_time < time_budget:
            if current_size == 0:
                break
            
            # Try removing an element and adding others
            x_remove = rng.choice(all_elements)
            if not S[x_remove-1]:
                continue
                
            S[x_remove-1] = False
            current_size -= 1
            
            # Try to add elements in random order
            candidates = all_elements[:]
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S):
                    S[x-1] = True
                    current_size += 1
                    
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
            else:
                # Revert if no improvement (simple hill climbing)
                S = best_S.copy()
                current_size = best_size
                
        return best_S

    def polish(S, time_budget):
        """Final polish pass: try all orderings of addition"""
        S = S.copy()
        current_size = get_size(S)
        best_size = current_size
        best_S = S.copy()
        
        all_elements = list(range(1, N + 1))
        
        # Try multiple random orderings
        while time.time() - start_time < time_budget:
            candidates = all_elements[:]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S):
                    S[x-1] = True
                    current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
        
        return best_S

    # Main
    best_S = np.zeros(N, dtype=bool)
    best_size = 0
    rng = random.Random(42)
    
    # Initial greedy with multiple seeds
    for seed in range(10):
        if time.time() - start_time > time_limit - 5:
            break
        S = greedy_construct(seed)
        size = get_size(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
        
    # Local search on best
    if time.time() - start_time < time_limit - 10:
        S = local_search(best_S, rng, time_limit - 10)
        size = get_size(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    # More restarts with local search
    seed = 10
    while time.time() - start_time < time_limit - 5:
        S = greedy_construct(seed)
        size = get_size(S)
        
        # Only do local search if promising
        if size >= best_size - 5:
            S = local_search(S, rng, time_limit - 5)
            size = get_size(S)
            
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
        seed += 1
        
    # Final polish pass
    if time.time() - start_time < time_limit - 2:
        best_S = polish(best_S, time_limit - 2)
        best_size = get_size(best_S)
        
    # Convert numpy bool array to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_S[i]]
    return result

# EVOLVE-BLOCK-END
