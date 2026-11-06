import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_val = 4096
    max_sum = 3 * max_val + 1
    used_sums = np.zeros(max_sum, dtype=bool)
    
    selected = []
    S = np.array([], dtype=np.int32)
    
    def get_new_sums(x, S):
        k = len(S)
        if k == 0:
            return np.array([3*x], dtype=np.int32)
        
        s1 = np.array([3*x], dtype=np.int32)
        s2 = 2*x + S
        i_idx, j_idx = np.triu_indices(k)
        s3 = x + S[i_idx] + S[j_idx]
        return np.concatenate((s1, s2, s3))
    
    def try_add(x, S, used_sums):
        """Try to add x to set S. Returns True if successful, and updates S and used_sums."""
        k = len(S)
        if k == 0:
            new_sums = np.array([3*x], dtype=np.int32)
        else:
            s1 = np.array([3*x], dtype=np.int32)
            s2 = 2*x + S
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            new_sums = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        if len(np.unique(new_sums)) != len(new_sums):
            return False, S, used_sums
            
        # Check against used_sums
        if np.any(used_sums[new_sums]):
            return False, S, used_sums
            
        # If valid, update
        S_new = np.append(S, x)
        used_sums[new_sums] = True
        return True, S_new, used_sums
    
    def greedy_with_ordering(order):
        """Run greedy algorithm with given ordering of candidates."""
        S = np.array([], dtype=np.int32)
        used_sums = np.zeros(max_sum, dtype=bool)
        selected = []
        
        for x in order:
            if time.time() > deadline - 2:
                break
            success, S, used_sums = try_add(x, S, used_sums)
            if success:
                selected.append(x)
        
        return selected
    
    # Strategy 1: Deterministic greedy from 1
    best_set = greedy_with_ordering(range(1, max_val + 1))
    best_size = len(best_set)
    
    # Strategy 2: Multiple randomized greedy restarts
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Different randomization strategies
        if restart % 3 == 0:
            # Random shuffle
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        elif restart % 3 == 1:
            # Biased toward smaller values (like original but shuffled slightly)
            candidates = list(range(1, max_val + 1))
            # Partial shuffle: swap random pairs
            for _ in range(max_val // 2):
                i, j = random.randint(0, max_val - 1), random.randint(0, max_val - 1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        else:
            # Reverse biased: try larger values first, then fill in
            candidates = list(range(max_val, 0, -1))
            random.shuffle(candidates)
        
        current_set = greedy_with_ordering(candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Strategy 3: Local search on best set - try to swap elements
    if time.time() < deadline - 10:
        current_set = sorted(best_set)
        S = np.array(current_set, dtype=np.int32)
        k = len(S)
        
        # Rebuild used_sums for current set
        used_sums = np.zeros(max_sum, dtype=bool)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    used_sums[S[i] + S[j] + S[l]] = True
        
        # Try to improve by swapping elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # For each element in the set, try to replace with a larger value
            for idx in range(k):
                if time.time() > deadline - 5:
                    break
                removed = S[idx]
                
                # Try candidate values larger than removed
                for new_val in range(removed + 1, max_val + 1, 7):  # Step by 7 to reduce candidates
                    if time.time() > deadline - 5:
                        break
                    
                    # Build candidate set
                    S_candidate = np.delete(S, idx)
                    S_candidate = np.append(S_candidate, new_val)
                    S_candidate = np.sort(S_candidate)
                    
                    # Check if valid
                    valid = True
                    temp_used = np.zeros(max_sum, dtype=bool)
                    n_c = len(S_candidate)
                    for i in range(n_c):
                        for j in range(i, n_c):
                            for l in range(j, n_c):
                                s = S_candidate[i] + S_candidate[j] + S_candidate[l]
                                if temp_used[s]:
                                    valid = False
                                    break
                                temp_used[s] = True
                        if not valid:
                            break
                    if not valid:
                        break
                    
                    if valid:
                        # Accept the swap
                        S = S_candidate
                        current_set = list(S)
                        # Rebuild used_sums
                        used_sums = np.zeros(max_sum, dtype=bool)
                        for i in range(k):
                            for j in range(i, k):
                                for l in range(j, k):
                                    used_sums[S[i] + S[j] + S[l]] = True
                        improved = True
                        break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Strategy 4: Try to add more elements to best_set
    if time.time() < deadline - 3:
        current_set = sorted(best_set)
        S = np.array(current_set, dtype=np.int32)
        k = len(S)
        
        used_sums = np.zeros(max_sum, dtype=bool)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    used_sums[S[i] + S[j] + S[l]] = True
        
        # Try to add new elements
        for x in range(1, max_val + 1):
            if time.time() > deadline - 3:
                break
            if x in current_set:
                continue
            success, S_new, used_sums = try_add(x, S, used_sums)
            if success:
                current_set.append(x)
                current_set.sort()
                S = S_new
                k += 1
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set
