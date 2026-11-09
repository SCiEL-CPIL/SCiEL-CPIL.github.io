import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 129  # Range [0, 128]
    min_sum = 128
    max_sum = 256
    
    best_A = None
    best_max_rep = float('inf')
    
    # Helper to compute max rep and validity
    def evaluate(A_set):
        if not A_set:
            return float('inf'), False
        A_list = list(A_set)
        # Use numpy for speed
        A_arr = np.array(A_list)
        # Compute all pair sums
        # Unordered pairs: i <= j
        # We can just compute all i, j and then adjust? 
        # The problem says "Pair representations are unordered and permit equal summands".
        # So (a,b) and (b,a) are the same representation.
        # If a != b, there is 1 representation.
        # If a == b, there is 1 representation.
        # So we just need to count the number of pairs {a,b} with a<=b such that a+b=s.
        
        # Efficient way:
        # convolve the indicator vector with itself?
        # If A is a set, let v be the indicator vector of A.
        # The number of ordered pairs (a,b) with a+b=s is (v*v)[s].
        # Let C_ordered[s] be this count.
        # For a != b, the unordered pair {a,b} corresponds to 2 ordered pairs (a,b) and (b,a).
        # For a == b, the unordered pair {a,a} corresponds to 1 ordered pair (a,a).
        # Let k be the number of elements a in A such that 2a = s. (0 or 1)
        # Then C_unordered[s] = (C_ordered[s] - k) / 2 + k = (C_ordered[s] + k) / 2.
        
        v = np.zeros(N, dtype=int)
        v[A_arr] = 1
        
        # Convolution
        conv = np.convolve(v, v, mode='full')
        
        # Check validity and max rep
        max_rep = 0
        valid = True
        for s in range(min_sum, max_sum + 1):
            c_ord = conv[s]
            # k is 1 if s is even and s//2 is in A, else 0
            if s % 2 == 0:
                a = s // 2
                if 0 <= a < N and v[a] == 1:
                    k = 1
                else:
                    k = 0
            else:
                k = 0
            
            c_unord = (c_ord + k) // 2
            
            if c_unord == 0:
                valid = False
                break
            if c_unord > max_rep:
                max_rep = c_unord
                
        return max_rep, valid

    # Local search function
    def local_search(A_init, time_limit):
        A_set = set(A_init)
        A_list = list(A_set)
        
        # Initial evaluation
        cur_max_rep, cur_valid = evaluate(A_set)
        if not cur_valid:
            return None, cur_max_rep
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding elements
            candidates_to_add = [i for i in range(N) if i not in A_set]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add:
                if time.time() > time_limit: break
                A_test = A_set | {x}
                new_max_rep, new_valid = evaluate(A_test)
                if new_valid and new_max_rep < cur_max_rep:
                    A_set = A_test
                    cur_max_rep = new_max_rep
                    improved = True
                    break # Restart search after improvement
            
            # Try removing elements
            candidates_to_remove = list(A_set)
            random.shuffle(candidates_to_remove)
            for y in candidates_to_remove:
                if time.time() > time_limit: break
                A_test = A_set - {y}
                if not A_test: continue
                new_max_rep, new_valid = evaluate(A_test)
                if new_valid and new_max_rep < cur_max_rep:
                    A_set = A_test
                    cur_max_rep = new_max_rep
                    improved = True
                    break
                    
            # Try swapping elements
            if time.time() > time_limit: break
            A_list = list(A_set)
            candidates_to_add = [i for i in range(N) if i not in A_set]
            if A_list and candidates_to_add:
                random.shuffle(A_list)
                random.shuffle(candidates_to_add)
                # Limit swaps to check to keep it fast
                swaps_to_check = min(50, len(A_list) * len(candidates_to_add))
                for _ in range(swaps_to_check):
                    if time.time() > time_limit: break
                    y = random.choice(A_list)
                    x = random.choice(candidates_to_add)
                    A_test = (A_set - {y}) | {x}
                    new_max_rep, new_valid = evaluate(A_test)
                    if new_valid and new_max_rep < cur_max_rep:
                        A_set = A_test
                        cur_max_rep = new_max_rep
                        improved = True
                        break
                        
        return A_set, cur_max_rep

    # Multiple restarts with different set sizes
    # Baseline max rep is ~65.
    # Random sets of size ~40-60 might do better.
    
    sizes_to_try = [40, 45, 50, 55, 60]
    
    for k in sizes_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this size
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Random subset of size k
            A_init = set(random.sample(range(N), k))
            
            # Run local search with a time slice
            # Give each restart a portion of the remaining time
            remaining = deadline - time.time()
            time_slice = remaining / (5 * (len(sizes_to_try) - sizes_to_try.index(k))) # Rough estimate
            
            A_res, max_rep = local_search(A_init, start_time + time_slice + 10) # Add buffer
            
            if A_res is not None and max_rep < best_max_rep:
                best_max_rep = max_rep
                best_A = A_res

    # If no valid set found (unlikely), fallback to seed
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)
