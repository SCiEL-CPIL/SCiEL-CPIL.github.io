"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = []
    best_size = 0
    
    N = 4096
    # Max possible sum is 3*4096 = 12288
    MAX_SUM = 12288
    
    # Precompute all triple sums for a given set using numpy
    def compute_triple_sums(arr):
        """Compute all unordered triple sums for array arr (sorted, distinct)."""
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.int32)
        if n == 1:
            return np.array([3 * arr[0]], dtype=np.int32)
        
        # Use numpy to compute all pairs
        # For unordered triples (a,b,c) with a<=b<=c:
        # Type 1: (a,a,a) -> 3a
        # Type 2: (a,a,b) with a<b -> 2a+b
        # Type 3: (a,b,b) with a<b -> a+2b
        # Type 4: (a,b,c) with a<b<c -> a+b+c
        
        sums_list = []
        
        # Type 1: (a,a,a)
        sums_list.append(3 * arr)
        
        # Type 2: (a,a,b) with a<b
        if n >= 2:
            # For each a, b > a
            # 2*arr[i] + arr[j] for i < j
            idx_i = np.arange(n)[:, None]
            idx_j = np.arange(n)[None, :]
            mask_ij = idx_i < idx_j
            ii = idx_i[mask_ij]
            jj = idx_j[mask_ij]
            if len(ii) > 0:
                sums_list.append(2 * arr[ii] + arr[jj])
        
        # Type 3: (a,b,b) with a<b
        if n >= 2:
            # arr[i] + 2*arr[j] for i < j
            idx_i = np.arange(n)[:, None]
            idx_j = np.arange(n)[None, :]
            mask_ij = idx_i < idx_j
            ii = idx_i[mask_ij]
            jj = idx_j[mask_ij]
            if len(ii) > 0:
                sums_list.append(arr[ii] + 2 * arr[jj])
        
        # Type 4: (a,b,c) with a<b<c
        if n >= 3:
            # arr[i] + arr[j] + arr[k] for i < j < k
            # This is more complex; use itertools or a different approach
            # For efficiency, we'll use a different method
            pass
        
        if sums_list:
            all_sums = np.concatenate(sums_list)
            return all_sums
        return np.array([], dtype=np.int32)
    
    # More efficient: use a set-based approach with numpy for speed
    def can_add_fast(selected_arr, x):
        """Check if x can be added to selected_arr maintaining B_3[1]."""
        n = len(selected_arr)
        if n == 0:
            return True
        
        # Check 3x
        # Check y + 2x for all y in selected
        # Check 2y + x for all y in selected
        # Check y + z + x for all y <= z in selected
        
        # Use numpy for vectorized checks
        # We need to check against existing sums
        
        # For efficiency with small n, use a hybrid approach
        # Precompute existing sums as a set for O(1) lookup
        
        # This function assumes existing_sums is passed or we recompute
        # For now, let's use a different strategy: build incrementally
        
        return True  # Placeholder
    
    # Better approach: Use a greedy algorithm with numpy-accelerated collision detection
    # Key insight: for a candidate x, the new sums are:
    # {3x} ∪ {y+2x : y in S} ∪ {2y+x : y in S} ∪ {y+z+x : y<=z in S}
    # We need to check if any of these collide with existing sums
    
    # Let's use a different strategy: 
    # 1. Start with a good initial set using geometric progression
    # 2. Use local search / simulated annealing to improve
    
    # Initial construction using powers
    def initial_construct():
        # Use a combination of strategies
        candidates = []
        
        # Powers of 3
        p = 1
        while p <= 4096:
            candidates.append(p)
            p *= 3
        
        # Add some variations
        for base in [2, 3, 4, 5]:
            p = 1
            while p <= 4096:
                candidates.append(p)
                p *= base
        
        # Sparse arithmetic progressions
        for start in [1, 2, 3]:
            for step in [10, 20, 50, 100]:
                val = start
                while val <= 4096:
                    candidates.append(val)
                    val += step
        
        # Random candidates
        for _ in range(200):
            candidates.append(random.randint(1, 4096))
        
        # Deduplicate and sort
        candidates = sorted(set(candidates))
        return candidates
    
    # Greedy construction with numpy acceleration
    def greedy_construct(candidate_list, max_iter=5000):
        selected = []
        existing_sums = set()
        in_selected = set()
        
        for x in candidate_list:
            if x in in_selected:
                continue
            
            # Check if x can be added
            n = len(selected)
            
            # Check 3x
            if 3 * x in existing_sums:
                continue
            
            # Check y + 2x and 2y + x
            valid = True
            for y in selected:
                if (y + 2 * x) in existing_sums:
                    valid = False
                    break
                if (2 * y + x) in existing_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check y + z + x for y <= z
            for i in range(n):
                y = selected[i]
                for j in range(i, n):
                    z = selected[j]
                    if (y + z + x) in existing_sums:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Add x
            selected.append(x)
            in_selected.add(x)
            existing_sums.add(3 * x)
            for y in selected[:-1]:
                existing_sums.add(y + 2 * x)
                existing_sums.add(2 * y + x)
            n = len(selected)
            for i in range(n - 1):
                y = selected[i]
                for j in range(i, n - 1):
                    z = selected[j]
                    existing_sums.add(y + z + x)
        
        return selected
    
    # Try multiple initializations and greedy passes
    for seed in range(1000):
        if time.time() > deadline - 3.0:
            break
        
        random.seed(seed)
        
        # Generate candidates
        candidates = initial_construct()
        
        # Shuffle for diversity
        if seed % 2 == 0:
            random.shuffle(candidates)
        
        # Greedy construction
        result = greedy_construct(candidates)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
        
        # Early stop
        if best_size >= 30:
            break
    
    # Local search improvement
    if best_size > 0 and time.time() < deadline - 5.0:
        # Try to improve by swapping elements
        current_set = set(best_set)
        current_arr = sorted(best_set)
        
        # Compute existing sums
        existing_sums = set()
        n = len(current_arr)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(current_arr[i] + current_arr[j] + current_arr[k])
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_set_copy = set(current_set)
            
            # Try removing one element and adding two
            for remove in list(current_set_copy):
                if time.time() > deadline - 2.0:
                    break
                
                temp_set = current_set_copy - {remove}
                temp_arr = sorted(temp_set)
                
                # Recompute sums for temp_set
                temp_sums = set()
                tn = len(temp_arr)
                for i in range(tn):
                    for j in range(i, tn):
                        for k in range(j, tn):
                            temp_sums.add(temp_arr[i] + temp_arr[j] + temp_arr[k])
                
                # Try to add two new elements
                added = 0
                for x in range(1, 4097):
                    if x in temp_set or added >= 2:
                        continue
                    
                    # Check if x can be added
                    n2 = len(temp_arr)
                    if 3 * x in temp_sums:
                        continue
                    valid = True
                    for y in temp_arr:
                        if (y + 2 * x) in temp_sums:
                            valid = False
                            break
                        if (2 * y + x) in temp_sums:
                            valid = False
                            break
                    if not valid:
                        continue
                    for i in range(n2):
                        y = temp_arr[i]
                        for j in range(i, n2):
                            z = temp_arr[j]
                            if (y + z + x) in temp_sums:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Add x
                    temp_set.add(x)
                    temp_arr = sorted(temp_set)
                    n2 = len(temp_arr)
                    temp_sums.add(3 * x)
                    for y in temp_arr[:-1]:
                        temp_sums.add(y + 2 * x)
                        temp_sums.add(2 * y + x)
                    for i in range(n2 - 1):
                        y = temp_arr[i]
                        for j in range(i, n2 - 1):
                            z = temp_arr[j]
                            temp_sums.add(y + z + x)
                    added += 1
                
                if added == 2 and len(temp_set) > len(current_set):
                    current_set = temp_set
                    current_arr = sorted(temp_set)
                    improved = True
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    return best_set
# EVOLVE-BLOCK-END
