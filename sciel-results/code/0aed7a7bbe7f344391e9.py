import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    n = 20
    max_val = 10000

    def compute_objective(A):
        """
        Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
        Uses NumPy for speed.
        """
        A_arr = np.array(A, dtype=np.int64)
        n_local = len(A_arr)
        
        # Generate indices for i <= j pairs
        i_idx, j_idx = np.triu_indices(n_local)
        
        sums = A_arr[i_idx] + A_arr[j_idx]
        prods = A_arr[i_idx] * A_arr[j_idx]
        
        sum_size = len(np.unique(sums))
        prod_size = len(np.unique(prods))
        
        return max(sum_size, prod_size)

    # Generate candidate pool for replacements
    # 1. Small integers (good for sum collisions)
    pool_small = list(range(1, 100))
    # 2. 2,3-smooth numbers (Hamming numbers) - good for product collisions
    smooths = []
    for a in range(0, 15):
        for b in range(0, 15):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= max_val:
                smooths.append(val)
    smooths = list(set(smooths))
    smooths.sort()
    
    # 3. Powers of 2, 3, 5
    powers = []
    for base in [2, 3, 5]:
        val = 1
        while val <= max_val:
            powers.append(val)
            val *= base
    powers = list(set(powers))
    powers.sort()
    
    # Combine pool
    candidate_pool = list(set(pool_small + smooths + powers))
    candidate_pool.sort()
    
    def generate_gp(ratio, start, length):
        A = []
        val = start
        for _ in range(length):
            A.append(val)
            val *= ratio
        return A

    # Strategy 1: Generate structured initial candidates
    initial_candidates = []
    
    # 1. Single GP
    for ratio in [2, 3]:
        for start in [1, 2]:
            A = generate_gp(ratio, start, n)
            if max(A) <= max_val:
                A = sorted(A)
                if len(set(A)) == n:
                    initial_candidates.append(A)
                    
    # 2. Two-cluster GP: {start1 * ratio^i} U {start2 * ratio^i}
    # This often yields smaller product sets because products stay within the "grid"
    for ratio in [2, 3, 4]:
        for start1 in [1, 2]:
            for start2 in [3, 4, 5, 10, 15]:
                if start1 == start2: continue
                A1 = generate_gp(ratio, start1, 10)
                A2 = generate_gp(ratio, start2, 10)
                A = A1 + A2
                # Ensure distinct and within bounds
                A = sorted(A)
                if len(set(A)) == n and max(A) <= max_val:
                    initial_candidates.append(A)
    
    # 3. Small consecutive integers
    initial_candidates.append(list(range(1, 21)))
    initial_candidates.append(list(range(2, 22)))
    
    # 4. Random samples from smooths
    random.seed(42)
    for _ in range(20):
        if len(smooths) >= n:
            A = random.sample(smooths, n)
            A.sort()
            initial_candidates.append(A)
            
    # 5. Random samples from small range
    for _ in range(20):
        A = random.sample(range(1, 500), n)
        A.sort()
        initial_candidates.append(A)

    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    
    for cand in initial_candidates:
        if len(cand) != n or len(set(cand)) != n:
            continue
        if max(cand) > max_val:
            continue
        score = compute_objective(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(1, n+1))
        best_score = compute_objective(best_set)
        
    # Simulated Annealing Refinement
    current_set = best_set[:]
    current_score = best_score
    best_set_sa = current_set[:]
    best_score_sa = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    # Precompute a set for fast membership check
    current_set_for_membership = set(current_set)
    
    # Time budget for SA
    sa_deadline = deadline - 5.0
    
    while time.time() < sa_deadline:
        if T < T_min:
            T = T_min
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Single replacement
            i = random.randint(0, n - 1)
            # Pick new value from candidate pool or random
            if random.random() < 0.8:
                new_val = random.choice(candidate_pool)
            else:
                new_val = random.randint(1, max_val)
                
            if not (1 <= new_val <= max_val) or new_val in current_set_for_membership:
                T *= cooling
                continue
                
            old_val = current_set[i]
            current_set[i] = new_val
            current_set_for_membership.remove(old_val)
            current_set_for_membership.add(new_val)
            
            new_score = compute_objective(current_set)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_set_sa = current_set[:]
            else:
                # Revert
                current_set[i] = old_val
                current_set_for_membership.remove(new_val)
                current_set_for_membership.add(old_val)
                
        elif move_type < 0.8:
            # Pairwise swap
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            # Try replacing A[i] with a neighbor of A[j] or random
            # Simple approach: swap values if they don't conflict? No, we need new values.
            # Let's do: Replace A[i] with random, Replace A[j] with random
            new_val_i = random.choice(candidate_pool)
            new_val_j = random.choice(candidate_pool)
            
            # Ensure distinctness and bounds
            if new_val_i == new_val_j:
                T *= cooling
                continue
            if not (1 <= new_val_i <= max_val) or not (1 <= new_val_j <= max_val):
                T *= cooling
                continue
            if new_val_i in current_set_for_membership and new_val_i != current_set[i]:
                T *= cooling
                continue
            if new_val_j in current_set_for_membership and new_val_j != current_set[j]:
                T *= cooling
                continue
            # Check if new_val_i is current_set[j] and new_val_j is current_set[i] (swap)
            # Or if they collide with other elements
            # Simple check:
            temp_set = set(current_set)
            temp_set.remove(current_set[i])
            temp_set.remove(current_set[j])
            if new_val_i in temp_set or new_val_j in temp_set or new_val_i == new_val_j:
                T *= cooling
                continue
                
            old_i = current_set[i]
            old_j = current_set[j]
            
            current_set[i] = new_val_i
            current_set[j] = new_val_j
            current_set_for_membership.remove(old_i)
            current_set_for_membership.remove(old_j)
            current_set_for_membership.add(new_val_i)
            current_set_for_membership.add(new_val_j)
            
            new_score = compute_objective(current_set)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_set_sa = current_set[:]
            else:
                # Revert
                current_set[i] = old_i
                current_set[j] = old_j
                current_set_for_membership.remove(new_val_i)
                current_set_for_membership.remove(new_val_j)
                current_set_for_membership.add(old_i)
                current_set_for_membership.add(old_j)
                
        else:
            # Local perturbation: change one element by a small delta
            i = random.randint(0, n - 1)
            delta_val = random.randint(-10, 10)
            new_val = current_set[i] + delta_val
            if not (1 <= new_val <= max_val) or new_val in current_set_for_membership:
                T *= cooling
                continue
                
            old_val = current_set[i]
            current_set[i] = new_val
            current_set_for_membership.remove(old_val)
            current_set_for_membership.add(new_val)
            
            new_score = compute_objective(current_set)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score_sa:
                    best_score_sa = new_score
                    best_set_sa = current_set[:]
            else:
                current_set[i] = old_val
                current_set_for_membership.remove(new_val)
                current_set_for_membership.add(old_val)
                
        T *= cooling
        
    # Final local search polish
    if time.time() < deadline - 1:
        current_set = best_set_sa[:]
        current_set_for_membership = set(current_set)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(n):
                for delta_val in range(-5, 6):
                    if delta_val == 0: continue
                    new_val = current_set[i] + delta_val
                    if not (1 <= new_val <= max_val) or new_val in current_set_for_membership:
                        continue
                    old_val = current_set[i]
                    current_set[i] = new_val
                    current_set_for_membership.remove(old_val)
                    current_set_for_membership.add(new_val)
                    
                    new_score = compute_objective(current_set)
                    if new_score < best_score_sa:
                        best_score_sa = new_score
                        best_set_sa = current_set[:]
                        improved = True
                    else:
                        current_set[i] = old_val
                        current_set_for_membership.remove(new_val)
                        current_set_for_membership.add(old_val)
                        
    return sorted(best_set_sa)

# EVOLVE-BLOCK-END
