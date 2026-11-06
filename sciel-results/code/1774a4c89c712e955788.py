import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill

    max_val = 4096
    max_sum = 3 * max_val + 1
    # used_sums[i] is True if sum i is already taken

    # Precompute triu indices for various k to save time?
    # k goes up to ~40. 40*40/2 = 800.
    # np.triu_indices is fast enough.

    def is_valid_candidate(x, S, used_sums):
        """
        Check if x can be added to S.
        """
        k = len(S)
        if k == 0:
            return True
        
        # Generate all new sums involving x
        # 1. (x, x, x)
        s1 = np.array([3 * x], dtype=np.int32)
        # 2. (s_i, x, x)
        s2 = S + 2 * x
        # 3. (s_i, s_j, x) for i <= j
        if k > 1:
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
        else:
            # k=1: S=[a]. s3 = x + a + a = x + 2a
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            
        all_new = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        # If there are duplicates, x creates a collision with itself
        # Optimization: use a set or sort? np.unique is O(n log n).
        # For small k, n is small.
        if len(np.unique(all_new)) != len(all_new):
            return False
            
        # Check against used sums
        if np.any(used_sums[all_new]):
            return False
            
        return True

    def add_element(x, S, used_sums):
        """
        Add x to S and update used_sums.
        Assumes x is valid.
        """
        k = len(S)
        if k == 0:
            S_new = np.array([x], dtype=np.int32)
            used_sums[3 * x] = True
            return S_new
        
        s1 = np.array([3 * x], dtype=np.int32)
        s2 = S + 2 * x
        i_idx, j_idx = np.triu_indices(k)
        s3 = x + S[i_idx] + S[j_idx]
        all_new = np.concatenate((s1, s2, s3))
        
        used_sums[all_new] = True
        S_new = np.append(S, x)
        return S_new

    def remove_element(x, S, used_sums):
        """
        Remove x from S and update used_sums.
        Returns new S.
        """
        k = len(S)
        S_old = np.delete(S, np.where(S == x)[0])
        
        s1_clear = np.array([3 * x], dtype=np.int32)
        s2_clear = S_old + 2 * x
        if len(S_old) > 0:
             i_idx, j_idx = np.triu_indices(len(S_old))
             s3_clear = x + S_old[i_idx] + S_old[j_idx]
        else:
             s3_clear = np.array([], dtype=np.int32)
             
        all_clear = np.concatenate((s1_clear, s2_clear, s3_clear))
        
        # Clear only if not used by others? 
        # Since S is valid, sums are unique. So clearing is safe.
        used_sums[all_clear] = False
        return S_old

    # Phase 1: Deterministic Greedy
    used_sums_init = np.zeros(max_sum, dtype=bool)
    S_init = np.array([], dtype=np.int32)
    for x in range(1, max_val + 1):
        if is_valid_candidate(x, S_init, used_sums_init):
            S_init = add_element(x, S_init, used_sums_init)
    
    best_set = S_init.tolist()
    best_len = len(best_set)

    # Phase 2: Local Search with Random Restarts
    # We will run multiple local search iterations.
    # Strategy:
    # 1. Start with a set (either deterministic greedy, or random greedy).
    # 2. Try to add elements.
    # 3. Try to swap elements to improve "density" or allow additions.
    # 4. Use a "soft" acceptance or simulated annealing? 
    #    Given the hard constraint, we can only accept valid moves.
    #    The objective is just to maximize size.
    #    So any valid add is an improvement.
    #    Swaps are neutral in size, but might enable future adds.
    
    num_restarts = 20
    # We have ~55s. 
    # One greedy pass is fast (~50ms).
    # Local search steps: each step involves is_valid_candidate which is O(k^2).
    # k~40. 40^2 = 1600 ops. Very fast.
    # We can do many steps.
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Random Restart Greedy
        random.seed(r + 123)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        used_sums_ls = np.zeros(max_sum, dtype=bool)
        S_ls = np.array([], dtype=np.int32)
        
        # Greedy on shuffled candidates
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if is_valid_candidate(x, S_ls, used_sums_ls):
                S_ls = add_element(x, S_ls, used_sums_ls)
                
        # Local Search Phase
        # Goal: Maximize len(S_ls)
        # Moves:
        # 1. Add a random valid x not in S.
        # 2. Swap u in S with v not in S if valid.
        
        # We will iterate for a fixed number of steps or time.
        steps_per_restart = 5000
        for step in range(steps_per_restart):
            if time.time() - start_time > DEADLINE:
                break
                
            current_len = len(S_ls)
            
            added = False
            swapped = False
            
            # Try Add
            # Pick a random x from 1..4096 not in S_ls.
            # To avoid picking from S_ls too often, we can maintain a list of availables?
            # S_ls is small. Random pick and check is fine.
            
            x_new = random.randint(1, max_val)
            if x_new not in S_ls:
                if is_valid_candidate(x_new, S_ls, used_sums_ls):
                    S_ls = add_element(x_new, S_ls, used_sums_ls)
                    added = True
                    
            if not added:
                # Try Swap
                # Pick u in S_ls, v not in S_ls
                if len(S_ls) > 0:
                    u = random.choice(S_ls.tolist())
                    v = random.randint(1, max_val)
                    if v != u and v not in S_ls:
                        # Remove u
                        S_temp = remove_element(u, S_ls, used_sums_ls)
                        # Check if v is valid with S_temp
                        if is_valid_candidate(v, S_temp, used_sums_ls):
                            # Accept swap
                            S_ls = add_element(v, S_temp, used_sums_ls)
                            swapped = True
                        else:
                            # Revert: Add u back to S_temp
                            S_ls = add_element(u, S_temp, used_sums_ls)
                            
            if len(S_ls) > best_len:
                best_len = len(S_ls)
                best_set = S_ls.tolist()
                # If we reached the theoretical max (40), we can stop early?
                # Max is 40.
                if best_len >= 40:
                    return best_set

    return best_set
